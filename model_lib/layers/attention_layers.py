"""Common attention modules."""
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from model_lib.layers import nn_layers


def _attention_dropout(attn_weights: jnp.ndarray,
                       *,
                       rate: float,
                       broadcast: bool = True,
                       dropout_rng: jnp.ndarray) -> jnp.ndarray:
  """Applies dropout on the attention weights.

  This function *always* applies dropout. There is no deterministic option.
  Args:
      attn_weights: Attention matrix.
      rate: Dropout rate.
      broadcast: Whether to use same dropout mask for batch dimensions.
      dropout_rng: RNG key to generate random mask.

  Returns:
      attention weights with dropout applied.
  """
  keep_prob = 1.0 - rate
  if broadcast:
    # Dropout is broadcast across batch+head+non-attention dimensions.
    dropout_shape = list(attn_weights.shape)
    dropout_shape[0] = 1  # broadcast batch
    dropout_shape[-2] = 1  # broadcast heads
    keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
  else:
    keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
  multiplier = (
      keep.astype(attn_weights.dtype) /
      jnp.asarray(keep_prob, dtype=attn_weights.dtype))
  return attn_weights * multiplier


def dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    *,
    mask: Optional[jnp.ndarray] = None,
    bias: Optional[jnp.ndarray] = None,
    broadcast_dropout: bool = True,
    dropout_rate: float = 0.1,
    dtype: jnp.dtype = jnp.float32,
    precision: Optional[jax.lax.Precision] = None,
    deterministic: bool,
    dropout_rng: Optional[jnp.ndarray] = None,
    capture_attention_weights: bool = True) -> jnp.ndarray:
  """Computes the dot-product attention given query, key, value.

  This is the core function for applying attention based on 
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Note: query, key, value need not have any batch dimensions.

  Args:
      query: Queries for calculating attention with shape of 
        `[batch..., q_length, num_heads, qk_depth_per_head]`.
      key: Keys for calculating attention with shape of 
        `[batch..., kv_length, num_heads, qk_depth_per_head]`.
      value: Values to be used in attention with shape of 
        `[batch..., kv_length, num_heads, v_depth_per_head]`.
      mask: Mask for the attention weights. This should be broadcastable to
        shape `[batch..., num_heads, q_length, kv_length]`. This should be a
        tensor of booleans, with `False` representing values to be masked out.
      bias: Bias for the attention weights. This should be broadcastable to 
        shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
        incorporating causal masks, padding masks, proximity bias etc.
      broadcast_dropout: Use a broadcasted dropout along batch dims.
      dropout_rate: Dropout rate for the attention weights. Defaults to 0.1.
      dtype: The dtype of the computation. Defaults to jnp.float32.
      precision: Numerical precision of the computation. Defaults to None.
      deterministic: Deterministic or not (to apply dropout).
      dropout_rng: Optional JAX PRNGKey to be used for Dropout.
      capture_attention_weights: Whether to add an identity layer to tag the
        attention weights to be used for capturing them using Flax
        capture_intermediate, e.g. for visualization. Note that if this is set
        to True, this function can only be called within a flax module.

  Returns:
      Output of shape `[batch..., length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert key.shape[:-3] == query.shape[:-3] == value.shape[:-3], (
      'q, k, v must have same batch dimensions.')
  assert key.shape[-1] == query.shape[-1], 'q, k depths must match.'
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

  # Calculate attention matrix.
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum(
      '...qhd,...khd->...hqk', query, key, precision=precision)

  # Apply attention bias/mask
  if mask is not None:
    big_neg = jnp.finfo(jnp.float32).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  if bias is not None:
    attn_weights = attn_weights + bias

  # Normalize the attention weights.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  if capture_attention_weights:
    # Tag the intermediate weights for logging/visualization
    attn_weights = nn_layers.IdentityLayer(name='attn_weights')(attn_weights)

  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.:
    if dropout_rng is None:
      raise ValueError(
          'Did not provide `dropout_rng` to dot_product_attention()')
    else:
      attn_weights = _attention_dropout(
          attn_weights,
          rate=dropout_rate,
          broadcast=broadcast_dropout,
          dropout_rng=dropout_rng)

  # Return weighted sum over values for each query position.
  return jnp.einsum(
      '...hqk,...khd->...qhd', attn_weights, value, precision=precision)


def get_fixed_sincos_position_embedding(
    x_shape: jnp.shape,
    temperature: float = 10_000,
    dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
  """Provides a fixed position encoding for 2D and 3D coordinates.
  
  The embedding follows the initialization method used in the paper:
  "Attention is All You Need", https://arxiv.org/abs/1706.03762

  Args:
    x_shape: the shape of the input for which a position embedding is needed.
    temperature: Temperature parameter.
    dtype: dtype of the position embedding.
  Returns:
    Matrix of PE, has shape [1, *x.shape[1:]].
  """
  assert len(x_shape) in (4, 5), f'Unsupported input shape: {x_shape}.'
  num_parts = 4 if len(x_shape) == 4 else 6
  channels = x_shape[-1]
  assert channels % num_parts == 0, f'Channels must be multiple of {num_parts}.'
  omega = jnp.arange(
      channels // num_parts, dtype=jnp.float32) / (
          channels / num_parts)
  omega = 1. / (temperature**omega)

  if len(x_shape) == 4:  # 2D Input.
    _, h, w, _ = x_shape
    y, x = jnp.mgrid[:h, :w]
    y = jnp.einsum('m,d->md', y.flatten(), omega)
    x = jnp.einsum('m,d->md', x.flatten(), omega)
    p = [jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)]
    shape = (1, h, w, channels)
  elif len(x_shape) == 5:  # 3D Input.
    _, t, h, w, _ = x_shape
    z, y, x = jnp.mgrid[:t, :h, :w]
    z = jnp.einsum('m,d->md', z.flatten(), omega)
    y = jnp.einsum('m,d->md', y.flatten(), omega)
    x = jnp.einsum('m,d->md', x.flatten(), omega)
    p = [jnp.sin(z), jnp.cos(z), jnp.sin(x), jnp.cos(y), jnp.sin(y), jnp.cos(y)]
    shape = (1, t, h, w, channels)
  else:  # Should never reach here because of assert at beginning.
    raise ValueError(f'Unsupported inputs shape: {x_shape}')

  assert (shape[0] == 1) and (shape[1:] == x_shape[1:])
  pe = jnp.concatenate(p, axis=-1)
  return jnp.asarray(pe, dtype).reshape(*shape)