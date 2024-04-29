"""Model utils."""
import functools
from typing import Optional

import jax
import jax.nn as nn
import jax.numpy as jnp


def apply_weights(output: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
  """Applies given weights of the inputs in the minibatch to outputs.
  
  Note that weights can be per-example (i.e. of shape `[batch,]`) or per
  pixel/token (i.e. of shape `[batch, height, width]` or `[batch, len]`)
  so we need to broadcast it to the output shape.

  Args:
    output: Computed output, which can be loss or the correctly classified
      examples, etc.
    weights: Weights of inputs in the batch, which can be None or an array of
      shape [batch, ...].
  
  Returns:
    Weighted output.
  """
  if output.ndim < weights.ndim:
    raise ValueError('Output rank should be >= weights rank')
  desired_weights_shape = weights.shape + (1,) * (output.ndim - weights.ndim)
  weights = jax.lax.broadcast_in_dim(weights,
                                     shape=desired_weights_shape,
                                     broadcast_dimensions=tuple(
                                         range(weights.ndim)))
  return output * weights

def apply_label_smoothing(one_hot_targets: jnp.ndarray,
                          label_smoothing: Optional[float]) -> jnp.ndarray:
  """Apply label smoothing to the one-hot targets.
  
  Applies label smoothing such that the on-values are transformed from 1.0 to
  `1.0 - label_smoothing + label_smoothing / num_classes` and off values are 
  transformed from 0.0 to `label_smoothing / num_classes`.
  https://arxiv.org/abs/1512.00567

  Note that another way of performing label smoothing (which we don't use here)
  is to take `label_smoothing` mass from the on-values and distribute it to the
  off-values; in other words, transform the on-values to `1.0 - label_smoothing`
  and the off-values to `label_smoothing / (num_classes - 1)`.
  http://jmlr.org/papers/v20/18-789.html

  Args:
    one_hot_targets: One-hot targets for an example, a [batch, ..., num_classes]
      float array.
    label_smoothing: A scalar in [0, 1] used to smooth the labels.

  Returns:
    A float array of the same shape as `one_hot_targets` with smoothed label
    values.
  """
  on_value = 1.0 - label_smoothing
  num_classes = one_hot_targets.shape[-1]
  off_value = label_smoothing / num_classes
  one_hot_targets = one_hot_targets * on_value + off_value
  return one_hot_targets


def weighted_unnormalized_softmax_cross_entropy(
    logits: jnp.ndarray,
    one_hot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_smoothing: Optional[float] = None,
    label_weights: Optional[jnp.ndarray] = None,
    logits_normalized: bool = False,
    keep_label_dimension: bool = False,
) -> jnp.ndarray:
  """Computes weighted softmax cross entropy given logits and targets.
  
  This computes sum_(x, y) softmax-ce(x, y) for a single, potentially padded
  minibatch. If the minibatch is padded (that is it contains null examples)
  it is assumed that `weights` is a binary mask where 0 indicates that the 
  example is null.

  Args:
    logits: Output of the model with shape [batch, ..., num_classes].
    one_hot_targets: One hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch x ...] (rank of one_hot_targets - 1)
    label_smoothing: Scalar to use to smooth the one-hot labels.
    label_weights: Weight per label of shape [num_classes].
    logits_normalized: If True, the logits are assumed to already be noramlized.
    keep_label_dimension: If True, the class dimension of the output loss is not
      summed over.
  
  Returns:
    The softmax cross entropy of the examples in the given batch.
  """
  if logits.ndim != one_hot_targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s one_hot_targets' %
        (str(logits.shape), str(one_hot_targets.shape)))

  # Optionally apply label smoothing.
  if label_smoothing is not None:
    one_hot_targets = apply_label_smoothing(one_hot_targets, label_smoothing)

  # Optionally apply label weights.
  if label_weights is not None:
    one_hot_targets *= label_weights

  if not logits_normalized:
    logits = nn.log_softmax(logits)

  loss = -one_hot_targets * logits

  if weights is not None:
    loss = apply_weights(loss)

  if not keep_label_dimension:
    loss = loss.sum(axis=-1)

  return loss


@functools.partial(jax.vmap, in_axes=[0, 0], out_axes=0)
def simple_gather(x: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
  """Gathers elements from `x` using the indices `idx`.
  
  `output[i] = x[i, idx[i]]`. This simple gather operation assumes that the 
  first dimension is the batch dimension. The indices index into the second
  dimension. The rest of the dimensions are copied as-is from `x` into output.
  Note that the implementation below only handles a single element in the batch.
  `jax.vmap` extends this to the batch dimension.

  Args:
    x: Inputs of shape [bs, n, d].
    idx: Array of shape [bs, m] and dtype jnp.int32 or int64 that specifies
      indices to be gathered from `x`.
  
  Returns:
    Gathered output of shape [bs, m, d].
  """
  return x[idx]
