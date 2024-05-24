# ----------------------------------------------------------------
# Modified from Scenic (https://github.com/google-research/scenic)
# Copyright 2024 The Scenic Authors.
# ----------------------------------------------------------------
"""Base model classes."""
from typing import Any, Callable, Dict, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
import ml_collections
from absl import logging

Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]


class BaseModel:
  """Defines commonalities between all models.
  
  A model is a class with three members: get_metrics_fn, loss_fn and a flax_model.

  get_metrics_fn returns a callable function, metric_fn, that calculates the 
  metrics and returns a dictionary. The metric function computes f(x_i, y_i) on
  a minibatch, it has API:
    ```
    metric_fn(logits, label, weights).
    ```
  
  The trainer will then aggregate and compute the mean across all samples 
  evaluated.

  loss_fn is a function of API:
    ```
    loss = loss_fn(logits, batch, model_params=None)
    ```
  
  This model defines a cross-entropy loss with weight decay, where they weight
  decay factor is determined by config.l2_decay_factor.

  flax_model is returned from the build_flax_model function. A typical usage 
  pattern will be:
    ```
    model_cls = model_lib.models.get_model_cls('fully_connected_classification')
    model = model_cls(config, dataset.meta_data)
    flax_model = model.build_flax_model()
    dummy_input = jnp.zeros(input_shape, model_input_dtype)
    model_state, params = flax_model.init(
      rng, dummy_input, train=False).pop('params')
    ```
  
  And this is how to call the model:
    ```
    variables = {'params': params, **model_state}
    logits, new_model_state = flax_model.apply(variables, inputs, ...)
    ```
  """

  def __init__(self, config: Optional[ml_collections.ConfigDict],
               dataset_meta_data: Dict[str, Any]) -> None:
    if config is None:
      logging.warning('You are creating the model with default config.')
      config = self.default_flax_model_config()
    self.config = config
    self.dataset_meta_data = dataset_meta_data
    self.flax_model = self.build_flax_model()

  def get_metrics_fn(self, split: Optional[str] = None) -> MetricFn:
    """Returns a callable metric function for the model.

    The metrics function is for pmap-based models, where we need to normalise
    by doing p-sums over other devices.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].

    Returns:
      A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    raise NotImplementedError('Subclasses must implement get_metrics_fn.')

  def get_metrics_fn_jit(self, split: Optional[str] = None) -> MetricFn:
    """Returns a callable metric function for the model.

    The metrics function is for jit-based models, where we normalise by doing
    sums over global arrays.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].

    Returns:
      A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    raise NotImplementedError('Subclasses must implement get_metrics_fn_jit.')

  def loss_function(self,
                    logits: jnp.ndarray,
                    batch: Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns the loss.

    Args:
      logits: Output of model in shape [batch, length, num_classes].
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    raise NotImplementedError('Subclasses must implement loss_function.')

  def build_flax_model(self) -> nn.Module:
    raise NotImplementedError('Subclasses must implement build_flax_model().')

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    """Default config for the flax model that is built in `build_flax_model`.

    This function in particular serves the testing functions and supposed to
    provide config that are passed to the flax_model when it's built in
    `build_flax_model` function, e.g., `model_dtype_str`.
    """
    raise NotImplementedError(
        'Subclasses must implement default_flax_model_config().')
