"""Utils."""
import jax
import jax.numpy as jnp


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
  warmup_steps = steps("warmup",
                       kw,
                       data_size,
                       batch_size,
                       total_steps,
                       default=0)
  cooldown_steps = steps("cooldown",
                         kw,
                         data_size,
                         batch_size,
                         total_steps,
                         default=0)

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
