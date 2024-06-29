"""Common train utils."""
import copy
import functools
import os
import time
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple, Union)

from absl import logging
from clu import metric_writers
import flax
from flax import jax_utils, struct
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from tensorflow.io import gfile

from common_lib import debug_utils, tree_utils
from dataset_lib import dataset_utils
from dataset_lib.datasets import DatasetRegistry

PyTree = Any


@struct.dataclass
class TrainState:
  """Dataclass to keep track of state of training.

  The state of training is structured as a struct.dataclass, which enables
  instances of this class to be passed into jax transformations like tree_map
  and pmap.
  """

  tx: Optional[optax.GradientTransformation] = struct.field(
      default=None, pytree_node=False)
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
  dataset_builder = DatasetRegistry.get(config.dataset_name)
  dataset = dataset_builder(
      rng=rng,
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
        keep=max_to_keep,
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
  if jax.process_index() == 0:
    train_metrics_summary = {
        key_separator.join((prefix, key)): val
        for key, val in train_metrics_summary.items()
    }
    train_logs = {key: val.mean() for key, val in train_logs.items()}

    writer.write_scalars(step, train_metrics_summary)
    writer.write_scalars(step, train_logs)

    # Log to stdout
    for name, value in {**train_logs, **train_metrics_summary}.items():
      logging.info(f"\u001b[35m[{step}]\u001b[0m {name} = {value:.4f}")

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

  if jax.process_index() == 0:
    eval_metrics_summary = {
        key_separator.join((prefix, key)): val
        for key, val in eval_metrics_summary.items()
    }
    writer.write_scalars(step, eval_metrics_summary)

    # Log to stdout
    for name, value in eval_metrics_summary.items():
      logging.info(f"\u001b[35m[{step}]\u001b[0m {name} = {value:.4f}")

  if flush_writer:
    writer.flush()

  return eval_metrics_summary


class Chrono:
  """Measures time and reports progress.
  
  This is a modified fork of Chrono class from big_vision codebase:
  https://github.com/google-research/big_vision/blob/main/big_vision/utils.py

  Some concepts:
  1. This differentiates between three "types" of time:
    - training time: the time spend on forward/backward passes.
    - program time: overall time the program runs, including all overheads
    - pause time: the chronometer can be paused (e.g. during evals).
  2. This handles a "warmup": the first step is skipped for training time
      purposes, as it includes significant compilation overheads, which distort
      estimates.
  3. `accumulates` (i.e. integrates) timings, and saves/loads them across
      restarts.
  """

  def __init__(self, example_type: str = 'img', warmup: int = 2):
    self.program_start_time = time.monotonic()
    self.train_start_time = None
    self.train_start_step = None  # When we started timing (after warmup)

    self.prev_time = None
    self.prev_step = None

    self.pause_start = None
    self.paused_time = 0

    self.warmup = warmup  # How many calls to `tick` to skip.
    self.load()  # Inits accum integrators.
    self.example_type = example_type

  def inform(self, first_step: int, total_steps: int, global_bs: int,
             steps_per_epoch: int):
    """Provide some extra info that's only known later in the program."""
    self.prev_step = copy.deepcopy(first_step)
    self.first_step = copy.deepcopy(first_step)
    self.total_steps = total_steps
    self.steps_per_epoch = steps_per_epoch
    self.global_bs = global_bs
    if total_steps:
      self.note = (
        f'Steps: {first_step}/{total_steps} [{first_step/total_steps:.1%}]'
      )

  def tick(self, step: int, writer: metric_writers.MetricWriter, write_note: Callable[[str], None]):
    """A chronometer tick."""
    summary = {}

    def hms(s):
      """Format time in hours/minutes/seconds."""
      if s < 60:
        return f'{s:.0f}s'
      m, s = divmod(s, 60)
      if m < 60:
        return f'{m:.0f}m {s:.0f}s'
      h, m = divmod(m, 60)
      return f'{h:.0f}h {m:.0f}m'  # Seconds intentionally omitted.
    
    now = time.monotonic()
    summary.update({'uptime': now - self.program_start_time})
    # We always count examples, regardles of the timing-related warmup that 
    # happens a few lines below.
    ds = step - self.prev_step  # Steps between ticks.
    self.prev_step = step
    self.accum_examples_seen += ds * self.global_bs
    summary.update({'examples_seen': self.accum_examples_seen})
    if self.steps_per_epoch:
      summary.update({'epoch': step / self.steps_per_epoch})
    
    # We take the start as the second time `tick` is called, so we avoid 
    # measuring the overhead of compilation and don't include it in the time
    # estimates.
    if self.warmup > 1:
      self.warmup -= 1
      write_note(self.note)  # This can help debugging.
      return
    if self.warmup == 1:
      self.train_start_time = self.prev_time = now
      self.train_start_step = step
      self.accum_program_time += now - self.program_start_time
      self.paused_time = 0
      self.warmup = 0
      write_note(self.note)  # This can help debugging.
      return
    
    # Measurements with micro-timings of current training steps speed.
    # Time between ticks (ignoring pause).
    if self.prev_time is None:
      raise ValueError('prev_time is None, possible warmup was skipped.')
    dt = now - self.prev_time - self.paused_time
    num_cores = jax.device_count()
    summary.update({
      f'{self.example_type}/sec/core': self.global_bs * ds / dt / num_cores,
      f'{self.example_type}/sec': self.global_bs * ds / dt
    })

    # Accumulate (integrate) times, good for plots.
    self.accum_train_time += dt
    self.accum_pause_time += self.paused_time
    self.accum_program_time += dt + self.paused_time

    # Convert to, and log as core-hours.
    core_hours = self.accum_train_time * num_cores / 60 / 60
    summary.update({'core_hours': core_hours})

    # Progress note with "global" full-program average timings.
    # (e.g. in program-time minus warmup)
    dt = now - self.train_start_time
    steps_timed = step - self.train_start_step
    steps_todo = self.total_steps - step
    self.note = f'Steps: {step}/{self.total_steps} [{step/self.total_steps:.1%}]'
    self.note += f'\nWalltime: {hms(self.accum_program_time)}'
    self.note += f' ({hms(self.accum_pause_time)} Not-train)'
    self.note += f'\nETA: {hms(dt / steps_timed * steps_todo)}'
    self.note += (
      f'\nTotal train time: {hms(dt / steps_timed * self.total_steps)}'
    )
    write_note(self.note)
    writer.write_scalars(step, summary)
    self.prev_time = now
    self.paused_time = 0


  def pause(self, wait_for=()):
    assert self.pause_start is None, "Don't pause twice."
    jax.block_until_ready(wait_for)
    self.pause_start = time.monotonic()
  
  def resume(self):
    assert self.pause_start is not None, "Cannot resume without pausing first."
    self.paused_time += time.monotonic() - self.pause_start
    self.pause_start = None

  def save(self):
    return dict(
        accum_program_time=self.accum_program_time,
        accum_train_time=self.accum_train_time,
        accum_pause_time=self.accum_pause_time,
        accum_examples_seen=self.accum_examples_seen,
    )

  def load(self, ckpt={}):  # pylint: disable=dangerous-default-value
    self.accum_program_time = ckpt.get('accum_program_time', 0.0)
    self.accum_train_time = ckpt.get('accum_train_time', 0.0)
    self.accum_pause_time = ckpt.get('accum_pause_time', 0.0)
    self.accum_examples_seen = ckpt.get('accum_examples_seen', 0)


def steps(prefix,
          config,
          data_size=None,
          batch_size=None,
          total_steps=None,
          default=ValueError):
  """Gets duration named `prefix` out of config and converts it to steps.
  
  Using this function to access a configuration value that denotes some kind of
  duration (eg training time, warmup, checkpoint frequency, ...) allows the
  duration to be specified in terms of steps, epochs, examples or percent of 
  training time, and coverts any of these into steps, such that the training 
  code only deals with steps.
  If the result is not an integer step number, it is rounded to the nearest one.

  Args:
    prefix: The name of the duration to query. The actual config fields can then
      be one of `prefix_steps`, `prefix_examples`, or `prefix_epochs`.
    config: The dictionary (config) from which to read the duration.
    data_size: The total number of training examples in one epoch.
    batch_size: The number of examples processed per batch.
    total_steps: The total number of training steps to run.
    default: The default value to return when no duration of the name `prefix`
      is found in the `config`. Set to `ValueError` (the default) to raise an
      error instead of returning a default value.
  
  Returns:
    The number of steps from the config, or the default value.
  
  Raises:
    ValueError if there is no such duration in the config and no default is set.
  """
  # Be helpful and make sure only match one of the following suffixes.
  suffixes = {"steps", "examples", "epochs"}
  matches = {f"{prefix}_{s}" for s in suffixes if f"{prefix}_{s}" in config}
  # Note that steps=0 is also a valid value (e.g. to only run evaluators)
  assert len(matches) <= 1, f"Only one of '{matches}' should be defined."

  # Steps are directly provided
  if f"{prefix}_steps" in config:
    return config[f"{prefix}_steps"]

  # Compute steps from total examples and batch_size
  if batch_size and f"{prefix}_examples" in config:
    return max(round(config[f"{prefix}_examples"] / batch_size), 1)

  # Compute steps from total_examples, epochs and batch_size
  if batch_size and data_size and f"{prefix}_epochs" in config:
    steps_per_epoch = data_size / batch_size
    return max(round(steps_per_epoch * config[f"{prefix}_epochs"]), 1)

  if total_steps and f"{prefix}_percent" in config:
    pct = config[f"{prefix}_percent"]
    assert 0.0 <= pct <= 1.0, (
        f"Percents should lie in [0.0, 1.0] but {prefix}_percent is {pct}")
    return max(round(pct * total_steps), 1)

  if default is ValueError:
    raise ValueError(
        f"Cannot convert {prefix} to steps, due to missing batch_size "
        f"({batch_size}), data_size ({data_size}), or corresponding entry in "
        f"config:\n" + "\n".join(config.keys()))

  return default


def create_learning_rate_schedule(total_steps,
                                  batch_size=None,
                                  data_size=None,
                                  base=1.0,
                                  decay_type="stair",
                                  scale_with_batchsize=False,
                                  **kw):
  """Creates learning rate schedule.
  
  Args:
    total_steps: The total number of steps to run.
    batch_size: The global batch-size optionally used if scaling is enabled.
    data_size: Number of examples in training data (for epoch conversion).
    base: The starting learning-rate (without warmup). 
    decay_type: 'linear' or 'cosine', 'rsqrt', 'stair'.
    scale_with_batchsize: Whether or not to scale lr automatically.
    **kw: Extra arguments specific to individual `decay_type`. Also contains
      declaration of `{warmup,cooldown}_{steps,epochs,examples}` that applies
      on top of any/all `decay_type`.
  
  Returns:
    A function learning_rate(step): float -> {"learning_rate": float}.
  """
  warmup_steps = steps(
      "warmup", kw, data_size, batch_size, total_steps, default=0)
  cooldown_steps = steps(
      "cooldown", kw, data_size, batch_size, total_steps, default=0)

  assert (total_steps <= 1) or (warmup_steps < total_steps), (
      "warmup_steps is >= total_steps")

  def step_fn(step):
    """Step -> lr function."""
    lr = base

    # This implements the linear scaling rule following
    # Goyal et. al. at arxiv.org/abs/1706.02677.
    # The reference batch size in literature is 256, so we scale lr to adjust
    # to the literature lr when batch_size changes.
    if scale_with_batchsize:
      lr = lr * batch_size / 256.

    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = jnp.clip(progress, 0.0, 1.0)
    if decay_type in ("linear", "polynomial"):
      power = kw.get("power", 1)  # Default power is linear decay
      zero = kw.get("end", kw.get("linear_end", 0))  # Ending lr
      lr = zero + (lr - zero) * (1.0 - progress)**power
    elif decay_type == "cosine":
      lr = lr * 0.5 * (1. + jnp.cos(jnp.pi * progress))
    elif decay_type == "stair":
      # Pick which step range the current step belongs to
      i = jnp.searchsorted(jnp.array(kw.get("steps", [])), step + 1)
      # Scale with the corresponding multiplier
      lr = lr * jnp.take(jnp.array([1.0] + list(kw.get("mults", []))), i)
    else:
      raise ValueError(f"Unknown lr type {decay_type}")

    if warmup_steps:
      lr = lr * jnp.minimum(1., step / warmup_steps)
    if cooldown_steps:
      lr = lr * jnp.minimum(1., (total_steps - step) / cooldown_steps)

    return jnp.asarray(lr, dtype=jnp.float32)

  return step_fn