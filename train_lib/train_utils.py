"""Common train utils."""
import functools
import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from absl import logging
from flax import struct
from flax.training import checkpoints
from tensorflow.io import gfile

import dataset_utils
import input_pipeline
from common_lib import debug_utils, tree_utils


@struct.dataclass
class TrainState:
  """Dataclass to keep track of state of training.

  The state of training is structured as a struct.dataclass, which enables
  instances of this class to be passed into jax transformations like tree_map
  and pmap.
  """

  tx: Optional[optax.GradientTransformation] = struct.field(default=None,
                                                            pytree_node=False)
  opt_state: Optional[optax.OptState] = None
  params: Optional[Any] = struct.field(default_factory=dict)
  global_step: Optional[int] = 0
  model_state: Optional[Any] = struct.field(default_factory=dict)
  rng: Optional[jnp.ndarray] = None
  metadata: Optional[Dict[str, Any]] = None

  # NOTE: When using the raw TrainState as the target for checkpoint restoration
  #  in Flax, you should provide the pytree structure, otherwise it might just
  #  silenty ignore restoring the checkpoint subtree if you use with an empty
  #  dict when setting `allow_partial_mpa_restoration=True` and if you set it
  #  to None (e.g., for `metadata`` above), Flax replaces it with a state dict.

  def __getitem__(self, item):
    """Make TrainState a subscriptable object."""
    return getattr(self, item)

  def get(self, keyname: str, default: Optional[Any] = None) -> Any:
    """Return the value for key if it exists otherwise the default."""
    try:
      return self[keyname]
    except KeyError:
      return default


class TrainingDivergedError(Exception):
  pass


def bind_rng_to_host_device(
    rng: jnp.ndarray,
    axis_name: Union[str, Tuple[str, ...]],
    bind_to: Optional[str] = None,
) -> jnp.ndarray:
  """Binds a rng to the host/device we are on.

  Must be called from within a pmapped function. Note that when binding to
  "device", we also bind the rng to hosts, as we fold_in the rng with axis_index
  which is unique for devices across all hosts.

  Args:
    rng: A jax.random.PRNGKey.
    axis_name: The axis of the devices we are binding rng across.
    bind_to: Must be one of the 'host' or 'device'. None means no binding.

  Returns:
    jax.random.PRNGKey specialized to host/device.
  """
  if bind_to is None:
    return rng
  if bind_to == 'host':
    return jax.random.fold_in(rng, jax.process_index())
  elif bind_to == 'device':
    return jax.random.fold_in(rng, jax.lax.axis_index(axis_name))
  else:
    raise ValueError(
        "`bind_to` should be one of the `[None, 'host', 'device']`")


def initialize_model(*,
                     model: Any,
                     input_spec: Sequence,
                     config: ml_collections.ConfigDict,
                     rngs: jnp.ndarray,
                     train: Optional[bool] = False,
                     **model_kwargs):
  """Initializes parameters and model state.
  
  Args:
    model: Definition of the model (flax model).
    input_spec: An iterable of (shape, dtype) pairs specifying the shape and
      dtype of the inputs. If unspecified, dtype is assumed float32.
    config: Configuration for init.
    rngs: Jax rng key.
    train: If the model should be initialized in train mode.
    model_kwargs: Other model keyword arguments.
  
  Returns:
    Initial params, Init model state, number of trainable params, GFLOPs.
  """
  batch_size = ((config.batch_size //
                 jax.device_count()) if config.get('batch_size') else None)
  dummy_input = []
  for spec in input_spec:
    if spec is not None:
      in_st = debug_utils.input_spec_to_jax_shape_dtype_struct(
          spec, batch_size=batch_size)
      dummy_input.append(jnp.zeros(in_st.shape, in_st.dtype))
    else:
      dummy_input.append(None)

  @functools.partial(jax.jit, backend='cpu')
  def _initialize_model(rngs):
    variables = flax.core.freeze(
        model.init(rngs, *dummy_input, train=train, **model_kwargs))
    init_model_state, init_params = flax.core.pop(variables, 'params')
    # Set initial head bias
    if config.get('init_head_bias'):

      def _maybe_constant_fill(name, p: jnp.ndarray):
        if 'bias' in name:
          return jnp.full_like(p, config.init_head_bias)

      init_params = flax.core.unfreeze(init_params)
      init_params['output_projection'] = tree_utils.tree_map_with_names(
          _maybe_constant_fill, init_params['output_projection'])
      init_params = flax.core.freeze(init_params)
    return init_params, init_model_state

  if not isinstance(rngs, dict):
    rngs = {'params': rngs}
  init_params, init_model_state = _initialize_model(rngs)
  rngs.pop('params')

  # Count number of trainable parameters.
  num_trainable_params = debug_utils.log_param_shapes(init_params)

  # Count FLOPs
  variables = {'params': init_params, **init_model_state}
  flops = debug_utils.compute_flops(
      flax_model_apply_fn=functools.partial(
          model.apply,
          variables,
          train=False,
          rngs=rngs,
          **model_kwargs,
      ),
      input_spec=input_spec,
      fuse_multiply_add=True,
  )
  gflops = flops / (10**9)

  return init_params, init_model_state, num_trainable_params, gflops


def get_dataset(
    config: ml_collections.ConfigDict,
    *,
    rng: jnp.ndarray,
    dataset_configs: Optional[ml_collections.ConfigDict] = None
) -> dataset_utils.Dataset:
  """Creates dataset.

  By default, values in the config file are used.
  However if the optional `dataset_name` and `dataset_configs` are passed, 
  those are used instead.

  Args:
    config: Experiment config file.
    rng: RNG key to use for the dataset.
    dataset_configs: Configuration of the dataset, if not reading directly
      from config.

  Returns:
      A dataset_utils.Dataset object
  """
  lead_host = jax.process_index() == 0

  def info(s, *a):
    if lead_host:
      logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)

  device_count = jax.device_count()
  info('Device_count: %d', device_count)
  info('Number of hosts: %d', jax.process_count())
  info('Current host ID: %d', jax.process_index())

  # Verify batch sizes
  batch_size = config.batch_size
  if batch_size % device_count > 0:
    raise ValueError(f'Batch size ({batch_size}) must be divisible by the '
                     f'total device count ({device_count}).')

  eval_batch_size = config.get('eval_batch_size', batch_size)
  if eval_batch_size % device_count > 0:
    raise ValueError(f'Eval batch size ({eval_batch_size}) must be divisible '
                     f'by the total device count ({device_count}).')

  local_batch_size = batch_size // jax.process_count()
  eval_local_batch_size = eval_batch_size // jax.process_count()
  device_batch_size = batch_size // device_count
  info('Local Batch Size: %d', local_batch_size)
  info('Per-device Batch Size: %d', device_batch_size)

  dataset_configs = dataset_configs or config.get('dataset_configs', {})
  num_local_shards = jax.local_device_count()
  dataset = input_pipeline.build_pipeline(rng=rng,
                                          batch_size=local_batch_size,
                                          eval_batch_size=eval_local_batch_size,
                                          num_shards=num_local_shards,
                                          dataset_configs=dataset_configs)
  return dataset


def restore_checkpoint(checkpoint_path: str,
                       train_state: Optional[TrainState] = None,
                       assert_exist: bool = False,
                       step: Optional[int] = None) -> Tuple[TrainState, int]:
  """Restores the last checkpoint.
  
  First restores the checkpoint, which is an instance of TrainState that holds
  the state of training.

  Args:
    checkpoint_path: Directory or filename to restore the checkpoint from.
    train_state: An instance of `TrainState` that holds the state of training.
    assert_exist: Assert that there is at least one checkpoint in the given path.
    step: Step number to load or None to load latest. If specified, 
      `checkpoint_path` must be a directory.
  
  Returns:
    A tuple of training state and an integer which is the current step.
  """
  if assert_exist:
    if 'checkpoint_' in checkpoint_path.split('/')[-1]:
      glob_path = checkpoint_path
    else:
      glob_path = os.path.join(checkpoint_path, 'checkpoint_*')
    if not gfile.glob(glob_path):
      raise ValueError(f'No checkpoint found in {checkpoint_path}.')

  if train_state is None:
    raise ValueError(f'Please use `restore_pretrained_checkpoint` for loading '
                     'a checkpoint without providing a TrainState.')

  train_state = checkpoints.restore_checkpoint(checkpoint_path, train_state,
                                               step)
  return train_state, int(train_state.global_step)
