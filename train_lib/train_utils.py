"""Common train utils."""
import functools
import os
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple, Union)

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from absl import logging
from clu import metric_writers
from flax import jax_utils, struct
from flax.training import checkpoints
from tensorflow.io import gfile

import dataset_utils
import input_pipeline
from common_lib import debug_utils, tree_utils

PyTree = Any


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


@functools.partial(jax.pmap, axis_name='x')
def pmap_mean(x: PyTree) -> PyTree:
  # An axis_name is passed to pmap which can then be used by pmean.
  # In this case each device has its own version of the batch statistics and we
  # average them.
  return jax.lax.pmean(x, 'x')


def sync_model_state_across_replicas(train_state: TrainState) -> TrainState:
  """Sync the model_state (like batch statistics) across replicas.
  
  Args:
    train_state: TrainState; Current state of training.
  
  Returns:
    Updated state of training in which model_state is synced across replicas.
  """
  if jax.tree_util.tree_leaves(train_state.model_state):
    # If model_state is not empty.
    new_model_state = flax.core.copy(
        train_state.model_state,
        {'batch_stats': pmap_mean(train_state.model_state['batch_stats'])},
    )
    return train_state.replace(model_state=new_model_state)
  else:
    return train_state


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


def save_checkpoint(workdir: str,
                    train_state: TrainState,
                    max_to_keep: int = 3,
                    overwrite: bool = False,
                    **kwargs):
  """Saves a checkpoint.
  
  Args:
    workdir: Experiment directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    max_to_keep: The number of checkpoints to keep.
    overwrite: Overwrite existing checkpoint if a checkpoint at the current or
      a later step already exists (default: False).
    **kwargs: Passed on to `flax.training.checkpoints.save_checkpoint`.
  """
  if jax.process_index() == 0:
    # Get train state from the first replica.
    checkpoint_state = jax.device_get(train_state)
    checkpoints.save_checkpoint(
        workdir,
        checkpoint_state,
        int(checkpoint_state.global_step),
        overwrite=overwrite,
        max_to_keep=max_to_keep,
        **kwargs,
    )


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


def normalize_metrics_summary(metrics_summary: Dict[str, Tuple[float, int]],
                              split: str) -> Dict[str, float]:
  """Normalize the metrics in summary by its normalizer.
  
  Args:
    metrics_summary: A dictionary mapping metric name to (value, normalizer).
    split: Split for which we normalize the metrics. Used for logging.
  
  Returns:
    Normalized metrics summary.

  Raises:
    TrainingDivergedError: Due to observing a NaN in the metrics.
  """
  normalized_metrics_summary = {}
  for key, val in metrics_summary.items():
    normalized_metrics_summary[key] = val[0] / (val[1] + 1e-9)
    if np.isnan(normalized_metrics_summary[key]):
      msg = f'NaN detected in {split}_{key} (Unnormalized values: {val})'
      if split == 'train':
        raise TrainingDivergedError(msg)
      else:
        logging.error(msg)

  return normalized_metrics_summary


def stack_forest(forest: List[PyTree]) -> PyTree:
  """Transposes a list of dicts to a dict of lists.
  
  For example, given `[{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]`, the output is
  `{'a': [1, 2], 'b': [3, 4]}`.

  Args:
    A list of dicts.
  
  Returns:
    A dict of lists.
  """
  if not forest:
    return {}

  stack_args = lambda *args: np.stack(args)
  return jax.tree_util.tree_map(stack_args, *forest)


def unreplicate_and_get(x: PyTree) -> PyTree:
  return jax.device_get(jax_utils.unreplicate(x))


def log_train_summary(
    step: int,
    *,
    writer: metric_writers.MetricWriter,
    train_metrics: Sequence[Dict[str, Tuple[float, int]]],
    extra_training_logs: Optional[Sequence[Dict[str, Any]]] = None,
    metrics_normalizer_fn: Optional[Callable[
        [Dict[str, Tuple[float, int]], str], Dict[str, float]]] = None,
    prefix: str = 'train',
    key_separator: str = '_',
    flush_writer: bool = True,
) -> Dict[str, float]:
  """Computes and logs train_metrics.
  
  Args:
    step: Current step.
    writer: Summary writer.
    train_metrics: List of dictionaries of calculated metrics. Usually the
      sequence is the concatenation of the per-eval-step metrics, and every
      dictionary maps a metric name to an array of (value, normalizer) - where
      the array index is usually the batch index.
    extra_training_logs: List of dictionaries, containing additional training
      logs, from every train step, e.g. learning rate, time, num parameters,
      etc. Their mean will be logged.
    metrics_normalizer_fn: Used for normalizing metrics. The API for this
      function is: `new_metrics_dict = metrics_normalizer_fn(metrics_dict, split)`.
      If set to None, we use the normalize_metrics_summary which uses the 
      normalizer paired with each metric to normalize it.
    prefix: str; Prefix added to the name of the summaries written by this fn.
    key_separator: str; Separator added between the prefix and key.
    flush_writer: If True, flush the writer after logging.

  Returns:
    A dictionary of metrics, mapping `train_metrics` from metric name (incl.
    `prefix`) to float value.
  """
  # Get metrics from devices.
  train_metrics = stack_forest(train_metrics)
  # Compute the sum over all examples in all batches.
  train_metrics_summary = jax.tree_util.tree_map(lambda x: x.sum(),
                                                 train_metrics)
  # Normalize metrics by the total number of examples.
  metrics_normalizer_fn = metrics_normalizer_fn or normalize_metrics_summary
  train_metrics_summary = metrics_normalizer_fn(train_metrics_summary, 'train')

  # Prepare additional training logs.
  extra_training_logs = extra_training_logs or [{}]
  train_logs = stack_forest(extra_training_logs)

  # Write metrics.
  writer.write_scalars(
      step,
      {
          key_separator.join((prefix, key)): val
          for key, val in train_metrics_summary.items()
      },
  )
  writer.write_scalars(
      step,
      {
          key: val.mean() for key, val in train_logs.items()
      },
  )

  if flush_writer:
    writer.flush()

  return train_metrics_summary


def log_eval_summary(
    step: int,
    *,
    writer: metric_writers.MetricWriter,
    eval_metrics: Sequence[Dict[str, Tuple[float, int]]],
    extra_eval_summary: Optional[Mapping[str, float]] = None,
    metrics_normalizer_fn: Optional[Callable[
        [Dict[str, Tuple[str, float]], str], Dict[str, float]]] = None,
    prefix: str = 'valid',
    key_separator: str = '_',
    flush_writer: bool = True,
) -> Dict[str, float]:
  """Computes and logs eval metrics.
  
  Args:
    step: Current step.
    writer: Metric writer object.
    eval_metrics: List of dictionaries of collected metrics. Usually the 
      sequence is the concatenation of the per-eval-step metrics, and every 
      dictionary maps a metric name to an array of (value, normalizer) - where
      the array index is usually the batch index.
    extra_eval_summary: A dict containing summaries that are already ready to 
      be logged, e.g. global metrics from eval set, like precision/recall.
    metrics_normalizer_fn: Used for normalizing metrics. The API for this 
      function is: `new_metrics_dict = metrics_normalizer_fn(metrics_dict,
      split)`. If set to None, we use the `normalize_metrics_summary` which uses
      the normalizer paired with each metric to normalize it (after summing both
      metric and normalizer values).
    prefix: str; Prefix added to the name of the summaries written by this 
      function.
    key_separator: Separator added between the prefix and key.
    flush_writer: If True, flush the writer after logging.

  Returns:
    A dictionary of metrics, mapping both `eval_metrics` and `extra_eval_summary`
    from metric name (incl. `prefix`) to float value.
  """
  eval_metrics = stack_forest(eval_metrics)

  # Compute the sum over all examples in all batches.
  eval_metrics_summary = jax.tree_util.tree_map(lambda x: x.sum(), eval_metrics)
  # Normalize the metrics by the total number of examples.
  metrics_normalizer_fn = metrics_normalizer_fn or normalize_metrics_summary
  eval_metrics_summary = metrics_normalizer_fn(eval_metrics_summary, 'eval')
  # If None, set to an empty dictionary.
  extra_eval_summary = extra_eval_summary or {}

  # Adds extra_eval_summary to the returned eval_summary.
  eval_metrics_summary.update(extra_eval_summary)

  writer.write_scalars(
      step,
      {
          key_separator.join((prefix, key)): val
          for key, val in eval_metrics_summary.items()
      },
  )

  if flush_writer:
    writer.flush()

  return eval_metrics_summary
