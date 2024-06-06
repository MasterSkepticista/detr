# ----------------------------------------------------------------
# Modified from Scenic DETR (https://github.com/google-research/scenic/scenic/baselines/detr)
# Copyright 2024 The Scenic Authors.
# ----------------------------------------------------------------

"""Implementation of the DETR Architecture.

End-to-End Object Detection with Transformers: https://arxiv.org/abs/2005.12872
Implementation is based on: https://github.com/facebookresearch/detr
"""

import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_lib.layers import attention_layers
from models import bit, detr_base_model, resnet

pytorch_kernel_init = functools.partial(nn.initializers.variance_scaling,
                                        1. / 3., 'fan_in', 'uniform')


def uniform_initializer(minval, maxval, dtype=jnp.float32):

  def init(key, shape, dtype=dtype):
    return jax.random.uniform(key, shape, dtype, minval=minval, maxval=maxval)

  return init


class InputPosEmbeddingSine(nn.Module):
  """Creates sinusoidal positional embeddings for the inputs."""
  hidden_dim: int
  dtype: jnp.dtype = jnp.float32
  scale: Optional[float] = None
  temperature: float = 10_000

  @nn.compact
  def __call__(self, padding_mask: jnp.ndarray) -> jnp.ndarray:
    """Creates the positional embeddings for transformer inputs.
    
    Args:
      padding_mask: Binary matrix with 0 at padded image regions. Shape is
        [batch, height, width]
      
    Returns:
      Positional embeddings for the inputs.
    
    Raises:
      ValueError if `hidden_dim` is not an even number.
    """
    if self.hidden_dim % 2:
      raise ValueError('`hidden_dim` must be an even number.')

    mask = padding_mask.astype(jnp.float32)
    y_embed = jnp.cumsum(mask, axis=1)
    x_embed = jnp.cumsum(mask, axis=2)

    # Normalization
    eps = 1e-6
    scale = self.scale if self.scale is not None else 2 * jnp.pi
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    num_pos_feats = self.hidden_dim // 2
    dim_t = jnp.arange(num_pos_feats, dtype=jnp.float32)
    dim_t = self.temperature**(2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, jnp.newaxis] / dim_t
    pos_y = y_embed[:, :, :, jnp.newaxis] / dim_t
    pos_x = jnp.stack(
        [
            jnp.sin(pos_x[:, :, :, 0::2]),
            jnp.cos(pos_x[:, :, :, 1::2]),
        ],
        axis=4,
    ).reshape(padding_mask.shape + (-1,))
    pos_y = jnp.stack(
        [
            jnp.sin(pos_y[:, :, :, 0::2]),
            jnp.cos(pos_y[:, :, :, 1::2]),
        ],
        axis=4,
    ).reshape(padding_mask.shape + (-1,))

    pos = jnp.concatenate([pos_y, pos_x], axis=3)
    b, h, w = padding_mask.shape
    pos = jnp.reshape(pos, [b, h * w, self.hidden_dim])
    return jnp.asarray(pos, self.dtype)


class QueryPosEmbedding(nn.Module):
  """Creates learnt positional embeddings for object queries.
  
  Attributes:
    hidden_dim: Hidden dimension for the pos embeddings.
    num_queries: Number of object queries.
    posemb_init: Positional embeddings initializer.
    dtype: Jax dtype; the dtype of the computation (default: float32)
  """
  hidden_dim: int
  num_queries: int
  posemb_init: Callable[..., Any] = jax.nn.initializers.normal(stddev=1.0)
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self) -> jnp.ndarray:
    """Creates the positional embeddings for queries.
    
    Returns:
      Positional embedding for object queries.
    """
    query_pos = self.param('query_emb', self.posemb_init,
                           (self.num_queries, self.hidden_dim))
    query_pos = jnp.expand_dims(query_pos, 0)
    return jnp.asarray(query_pos, self.dtype)


class MlpBlock(nn.Module):
  """Transformer Mlp/Feedforward block."""
  mlp_dim: int
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable[..., Any] = nn.initializers.xavier_uniform()
  bias_init: Callable[..., Any] = nn.initializers.normal(stddev=1e-6)
  activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  dtype: jnp.ndarray = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               deterministic: bool = True) -> jnp.ndarray:
    out_dim = self.out_dim or inputs.shape[-1]
    x = nn.Dense(
        self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(
            inputs)
    x = self.activation_fn(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    output = nn.Dense(
        out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(
            x)
    output = nn.Dropout(rate=self.dropout_rate)(output, deterministic)
    return output


class MultiHeadDotProductAttention(nn.Module):
  """DETR Customized MHDPA.
  
  Attributes:
    num_heads: Number of transformer heads.
    qkv_features: Size of query, key, value dim.
    out_features: Size of the last projection.
    dropout_rate: Dropout rate for attention weights.
    broadcast_dropout: Whether to broadcast dropout mask across batch and heads.
    kernel_init: Dense kernel initializer.
    bias_init: Dense bias initializer.
    use_bias: Whether to use bias in the Dense projections.
    dtype: The dtype of the computation.
  """
  num_heads: int
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  dropout_rate: float = 0.
  broadcast_dropout: bool = False
  kernel_init: Callable[..., Any] = jax.nn.initializers.xavier_uniform()
  bias_init: Callable[..., Any] = jax.nn.initializers.zeros
  use_bias: bool = True
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs_q: jnp.ndarray,
               inputs_kv: Optional[jnp.ndarray] = None,
               *,
               pos_emb_q: Optional[jnp.ndarray] = None,
               pos_emb_k: Optional[jnp.ndarray] = None,
               pos_emb_v: Optional[jnp.ndarray] = None,
               key_padding_mask: Optional[jnp.ndarray] = None,
               train: bool = False) -> jnp.ndarray:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key and value vectors,
    applies dot-product attention and projects the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` or for self-attention by only specifying `inputs_q` and 
    setting `inputs_kv` to None.

    Args:
        inputs_q: Input queries of shape `[bs, len, features]`.
        inputs_kv: Key/values of same shape as `inputs_q`, or None for 
          self-attention, in which case key/values will be derived from `inputs_q`.
        pos_emb_q: Positional embedding to be added to query. Defaults to None.
        pos_emb_k: Positional embedding to be added to key. Defaults to None.
        pos_emb_v: Positional embedding to be added to value. Defaults to None.
        key_padding_mask: Binary array. Key-value tokens that are padded are 0, and 1 otherwise. Defaults to None.
        train: Train or not (to apply dropout). Defaults to False.

    Returns:
        output of shape `[bs, len, features]`.
    """
    if inputs_kv is None:
      inputs_kv = inputs_q

    assert inputs_kv.ndim == inputs_q.ndim == 3
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]

    assert qkv_features % self.num_heads == 0, (
        'qkv dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    def add_positional_emb(x, pos_emb_x):
      return x if pos_emb_x is None else x + pos_emb_x

    query, key, value = (add_positional_emb(inputs_q, pos_emb_q),
                         add_positional_emb(inputs_kv, pos_emb_k),
                         add_positional_emb(inputs_kv, pos_emb_v))
    dense = functools.partial(
        nn.DenseGeneral,
        axis=-1,
        features=(self.num_heads, head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype)
    # Project inputs to multi-headed q/k/v
    # Dimensions are then [bs, ctx, n_heads, n_features_per_head]
    query, key, value = (dense(name='query')(query), dense(name='key')(key),
                         dense(name='value')(value))

    # Create attention masks
    if key_padding_mask is not None:
      attention_bias = (1 - key_padding_mask) * -1e10
      # add head and query dimension
      attention_bias = jnp.expand_dims(attention_bias, -2)
      attention_bias = jnp.expand_dims(attention_bias, -2)
    else:
      attention_bias = None

    # Apply attention
    x = attention_layers.dot_product_attention(
        query,
        key,
        value,
        bias=attention_bias,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        dropout_rng=self.make_rng('dropout') if train else None,
        deterministic=not train,
        capture_attention_weights=True)

    # back to the original inputs dimensions
    out = nn.DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=True,
        dtype=self.dtype,
        name='out')(
            x)
    return out


class EncoderBlock(nn.Module):
  """DETR Transformer Encoder block.
  
  Attributes:
    num_heads: Number of heads.
    qkv_dim: Dimension of the query/key/value.
    mlp_dim: Dimension of the mlp on top of attention block.
    pre_norm: If use LayerNorm before attention/mlp blocks.
    dropout_rate: Dropout rate at MLPs.
    attention_dropout_rate: Dropout rate for attention weights.
    dtype: Data type of the computation.
  """
  num_heads: int
  qkv_dim: int
  mlp_dim: int
  pre_norm: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               pos_embedding: Optional[jnp.ndarray] = None,
               padding_mask: Optional[jnp.ndarray] = None,
               train: bool = False) -> jnp.ndarray:
    """Applies EncoderBlock module.

    Args:
      inputs: Input data of shape `[bs, len, features]`.
      pos_embedding: Positional embedding to be added to the queries and keys
        in the self-attention operation. Defaults to None.
      padding_mask: Binary mask containing 0 for padded tokens.
      train: Train or not (to apply dropout). Defaults to False.

    Returns:
      Output after encoder block.
    """
    self_attn = MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.qkv_dim,
        dropout_rate=self.attention_dropout_rate,
        broadcast_dropout=False,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros,
        use_bias=True,
        dtype=self.dtype)

    mlp = MlpBlock(
        mlp_dim=self.mlp_dim,
        activation_fn=nn.relu,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate)

    assert inputs.ndim == 3

    if self.pre_norm:
      x = nn.LayerNorm(dtype=self.dtype)(inputs)
      x = self_attn(
          inputs_q=x,
          pos_emb_q=pos_embedding,
          pos_emb_k=pos_embedding,
          pos_emb_v=None,
          key_padding_mask=padding_mask,
          train=train)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = x + inputs
      y = nn.LayerNorm(dtype=self.dtype)(x)
      y = mlp(y, deterministic=not train)
      out = x + y
    else:
      x = self_attn(
          inputs_q=inputs,
          pos_emb_q=pos_embedding,
          pos_emb_k=pos_embedding,
          pos_emb_v=None,
          key_padding_mask=padding_mask,
          train=train)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = x + inputs
      x = nn.LayerNorm(dtype=self.dtype)(x)
      y = mlp(x, deterministic=not train)
      y = x + y
      out = nn.LayerNorm(dtype=self.dtype)(y)

    return out


class DecoderBlock(nn.Module):
  """DETR Transformer decoder block

  Attributes:
    num_heads: Number of heads.
    qkv_dim: Size of the query/key/value.
    mlp_dim: Size of the MLP on top of the attention block.
    pre_norm: If use LayerNorm before attention/mlp blocks.
    dropout_rate: Dropout rate for MLP/post-attention.
    attention_dropout_rate: Dropout rate for attention weights.
    dtype: Data type of the computation.
  """
  num_heads: int
  qkv_dim: int
  mlp_dim: int
  pre_norm: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               obj_queries: jnp.ndarray,
               encoder_output: jnp.ndarray,
               *,
               pos_embedding: Optional[jnp.ndarray] = None,
               query_pos_emb: Optional[jnp.ndarray] = None,
               key_padding_mask: Optional[jnp.ndarray] = None,
               train: bool = False) -> jnp.ndarray:
    """Applies DecoderBlock module.

    Args:
        obj_queries: Input data for decoder.
        encoder_output: Output of encoder, which are encoded inputs.
        pos_embedding: Positional embedding to be added to the keys in x-attn. Defaults to None.
        query_pos_emb: Positional embedding to be added to the queries. Defaults to None.
        key_padding_mask: Binary mask containing 0 for pad tokens in the key. Defaults to None.
        train: Train or not (to apply dropout). Defaults to False.

    Returns:
        Output after transformer decoder block.
    """
    assert query_pos_emb is not None, ('Given that object_queries are zeros '
                                       'and not learnable, we should add '
                                       'learnable query_pos_emb to them.')
    # Seems in DETR the self-attention in the first layer basically does
    # nothing, as the value vector is a zero vector and we add no learnable
    # positional embedding to it!
    self_attn = MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.qkv_dim,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros,
        use_bias=True,
        dtype=self.dtype)

    cross_attn = MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.qkv_dim,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros,
        use_bias=True,
        dtype=self.dtype)

    mlp = MlpBlock(
        mlp_dim=self.mlp_dim,
        activation_fn=nn.relu,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate)

    assert obj_queries.ndim == 3
    if self.pre_norm:
      x = nn.LayerNorm(dtype=self.dtype)(obj_queries)
      x = self_attn(
          inputs_q=x,
          pos_emb_q=query_pos_emb,
          pos_emb_k=query_pos_emb,
          pos_emb_v=None,
          train=train)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = x + obj_queries
      # cross attention block
      y = nn.LayerNorm(dtype=self.dtype)(x)
      y = cross_attn(
          inputs_q=y,
          inputs_kv=encoder_output,
          pos_emb_q=query_pos_emb,
          pos_emb_k=pos_embedding,
          pos_emb_v=None,
          key_padding_mask=key_padding_mask,
          train=train)
      y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
      y = y + x
      # mlp block
      z = nn.LayerNorm(dtype=self.dtype)(y)
      z = mlp(z, deterministic=not train)
      out = z + y
    else:
      x = self_attn(
          inputs_q=obj_queries,
          pos_emb_q=query_pos_emb,
          pos_emb_k=query_pos_emb,
          pos_emb_v=None,
          train=train)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = x + obj_queries
      x = nn.LayerNorm(dtype=self.dtype)(x)
      # cross attn
      y = cross_attn(
          inputs_q=x,
          inputs_kv=encoder_output,
          pos_emb_q=query_pos_emb,
          pos_emb_k=pos_embedding,
          pos_emb_v=None,
          key_padding_mask=key_padding_mask,
          train=train)
      y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
      y = y + x
      y = nn.LayerNorm(dtype=self.dtype)(y)
      # mlp block
      z = mlp(y, deterministic=not train)
      z = y + z
      out = nn.LayerNorm(dtype=self.dtype)(z)

    return out


class Encoder(nn.Module):
  """Applies multiple encoder blocks sequentially."""
  num_heads: int
  num_layers: int
  qkv_dim: int
  mlp_dim: int
  normalize_before: bool = False
  norm: Optional[Callable[..., Any]] = None
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               pos_embedding: Optional[jnp.ndarray] = None,
               padding_mask: Optional[jnp.ndarray] = None,
               train: bool = False) -> jnp.ndarray:
    assert inputs.ndim == 3
    x = inputs
    for lyr in range(self.num_layers):
      x = EncoderBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          pre_norm=self.normalize_before,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          dtype=self.dtype)(
              x,
              pos_embedding=pos_embedding,
              padding_mask=padding_mask,
              train=train)

    if self.norm is not None:
      x = self.norm(x)
    return x


class Decoder(nn.Module):
  """Applies multiple decoder blocks sequentially.
  
  Attributes:
    num_heads: Number of heads.
    num_layers: Number of layers.
    qkv_dim: Dimension of the query/key/value.
    mlp_dim: Dimension of the mlp on top of attention block.
    normalize_before: If use LayerNorm before attention/mlp blocks.
    norm: Flax module to use for normalization.
    return_intermediate: If return the outputs from intermediate layers.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout rate for attention weights.
    dtype: Data type of the computation (default: float32).
  """
  num_heads: int
  num_layers: int
  qkv_dim: int
  mlp_dim: int
  normalize_before: bool = False
  norm: Optional[Callable[..., Any]] = None
  return_intermediate: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               obj_queries: jnp.ndarray,
               encoder_output: jnp.ndarray,
               *,
               key_padding_mask: Optional[jnp.ndarray] = None,
               pos_embedding: Optional[jnp.ndarray] = None,
               query_pos_emb: Optional[jnp.ndarray] = None,
               train: bool = False) -> jnp.ndarray:
    assert encoder_output.ndim == 3
    assert obj_queries.ndim == 3
    y = obj_queries
    outputs = []
    for lyr in range(self.num_layers):
      y = DecoderBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          pre_norm=self.normalize_before,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          dtype=self.dtype,
          name=f'decoderblock_{lyr}')(
              y,
              encoder_output,
              pos_embedding=pos_embedding,
              query_pos_emb=query_pos_emb,
              key_padding_mask=key_padding_mask,
              train=train)
      if self.return_intermediate:
        outputs.append(y)

    if self.return_intermediate:
      y = jnp.stack(outputs, axis=0)

    if self.norm is not None:
      y = self.norm(y)

    return y


class DETRTransformer(nn.Module):
  """Encoder/Decoder blocks for DETR.
  
  Attributes:
    num_queries: Number of object queries.
    query_emb_size: Size of the embedding learnt for object queries.
    num_heads: Number of transformer heads.
    num_encoder_layers: Number of encoder layers.
    num_decoder_layers: Number of decoder layers.
    qkv_dim: Dimension of the query/key/value.
    mlp_dim: Dimension of the mlp on top of the attention block.
    return_intermediate_dec: Whether to return the outputs from intermediate layers of decoder.
    normalize_before: If use LayerNorm before attention/mlp blocks.
    dropout_rate: Dropout rate of MLP/post-attention.
    attention_dropout_rate: Dropout rate for attention weights.
    dtype: Datatype of the computation (defaults to float32).
  """
  num_queries: int
  query_emb_size: Optional[int] = None
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  return_intermediate_dec: bool = False
  normalize_before: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               padding_mask: Optional[jnp.ndarray] = None,
               pos_embedding: Optional[jnp.ndarray] = None,
               query_pos_emb: Optional[jnp.ndarray] = None,
               train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    encoder_norm = nn.LayerNorm(
        dtype=self.dtype) if self.normalize_before else None
    encoder_output = Encoder(
        num_heads=self.num_heads,
        num_layers=self.num_encoder_layers,
        qkv_dim=self.qkv_dim,
        mlp_dim=self.mlp_dim,
        normalize_before=self.normalize_before,
        norm=encoder_norm,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype,
        name='encoder')(
            inputs,
            padding_mask=padding_mask,
            pos_embedding=pos_embedding,
            train=train)

    query_dim = self.query_emb_size or inputs.shape[-1]
    obj_query_shape = tuple([inputs.shape[0], self.num_queries, query_dim])
    # Note that we always learn query_pos_emb, so we simply use constant zero
    # vectors for obj_queries, and later when applying attention, we have:
    # query = query_pos_emb + obj_queries
    obj_queries = jnp.zeros(obj_query_shape)

    decoder_norm = nn.LayerNorm(dtype=self.dtype)
    decoder_output = Decoder(
        num_heads=self.num_heads,
        num_layers=self.num_decoder_layers,
        qkv_dim=self.qkv_dim,
        mlp_dim=self.mlp_dim,
        normalize_before=self.normalize_before,
        return_intermediate=self.return_intermediate_dec,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        norm=decoder_norm,
        dtype=self.dtype,
        name='decoder')(
            obj_queries,
            encoder_output,
            key_padding_mask=padding_mask,
            pos_embedding=pos_embedding,
            query_pos_emb=query_pos_emb,
            train=train)
    return decoder_output, encoder_output


class BBoxCoordPredictor(nn.Module):
  """FFN block for predicting bounding box coordinates."""
  mlp_dim: int
  num_layers: int = 3
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    for _ in range(self.num_layers - 1):
      # This is like pytorch initializes biases in linear layers.
      bias_range = 1 / np.sqrt(x.shape[-1])
      x = nn.Dense(
          self.mlp_dim,
          kernel_init=pytorch_kernel_init(dtype=self.dtype),
          bias_init=uniform_initializer(
              -bias_range, bias_range, dtype=self.dtype),
          dtype=self.dtype)(
              x)
      x = nn.relu(x)

    bias_range = 1 / np.sqrt(x.shape[-1])
    x = nn.Dense(
        4,
        kernel_init=pytorch_kernel_init(dtype=self.dtype),
        bias_init=uniform_initializer(
            -bias_range, bias_range, dtype=self.dtype),
        dtype=self.dtype)(
            x)
    output = nn.sigmoid(x)
    return output


class ObjectClassPredictor(nn.Module):
  """Linear Projection block for prediction classes."""
  num_classes: int
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies Linear Projection to inputs.
    
    Args:
      inputs: Input data.

    Returns:
      Output of Linear Projection block.
    """
    bias_range = 1. / np.sqrt(inputs.shape[-1])
    return nn.Dense(
        self.num_classes,
        kernel_init=pytorch_kernel_init(dtype=self.dtype),
        bias_init=uniform_initializer(-bias_range, bias_range, self.dtype),
        dtype=self.dtype)(
            inputs)


class DETR(nn.Module):
  """Detection Transformer Model.
  
  Attributes:
    num_classes: Number of object classes.
    hidden_dim: Hidden dimension of the inputs to the model.
    num_queries: Number of object queries, ie detection slot. This is the 
      maximal number of objects this model can detect in a single frame.
    query_emb_size: Size of the embedding learned for object queries.
    transformer_num_heads: Number of transformer heads.
    transformer_num_encoder_layers: Number of transformer encoder layers.
    transformer_num_decoder_layers: Number of transformer decoder layers.
    transformer_qkv_dim: Dimension of the transformer query/key/value.
    transformer_mlp_dim: Dimension of the MLP above attention block.
    transformer_normalize_before: If use LayerNorm before attention/mlp blocks.
    backbone_module: ResNet flavor to use. 'bit' uses the ResNet variant from
      `BigTransfer`. 'resnet' uses the original ResNet module.
    backbone_width: Backbone ResNet width (Defaults to 1).
    backbone_depth: Backbone ResNet depth as integer or sequence of block sizes (Defaults to 50).
    aux_loss: If train with auxiliary loss.
    dropout_rate: Dropout rate of MLP/post-attention.
    attention_dropout_rate: Attention dropout rate.
    dtype: Data type of the computation (default: float32).
  """
  num_classes: int
  hidden_dim: int = 512
  num_queries: int = 100
  query_emb_size: Optional[int] = None
  transformer_num_heads: int = 8
  transformer_num_encoder_layers: int = 6
  transformer_num_decoder_layers: int = 6
  transformer_qkv_dim: int = 512
  transformer_mlp_dim: int = 2048
  transformer_normalize_before: bool = False
  backbone_module: str = 'bit'
  backbone_width: int = 1
  backbone_depth: Union[int, Sequence[int]] = 50
  aux_loss: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               train: bool,
               *,
               padding_mask: Optional[jnp.ndarray] = None,
               update_batch_stats: bool = False) -> Dict[str, Any]:
    """Applies DETR on the inputs.
    
    Args:
      inputs: Input data.
      train: Whether it is training.
      padding_mask: Binary matrix with 0 at padded image regions.
      update_batch_stats: Whether to update batch statistics for the batchnorms
        in backbone (if any). If None, value of `train` flag will be used.
    
    Returns:
      Output dict that has `pred_logits` and `pred_boxes` and potentially
      `aux_outputs`.
    """
    assert self.transformer_qkv_dim == self.hidden_dim

    if update_batch_stats is None:
      update_batch_stats = train

    assert self.backbone_module in [
        'bit', 'resnet'
    ], (f'Unsupported backbone module `{self.backbone_module}`')

    representation = {'bit': 'pre_logits_2d', 'resnet': 'stage_4'}

    # TODO: Remove hardcode of bit.ResNet
    _, backbone_features = bit.ResNet(
        width=self.backbone_width,
        depth=self.backbone_depth,
        dtype=self.dtype,
        name='backbone')(
            inputs, train=update_batch_stats)
    x = backbone_features[representation[self.backbone_module]]
    bs, h, w, _ = x.shape

    if padding_mask is None:
      padding_mask_downsampled = jnp.ones((bs, h, w), dtype=jnp.bool_)
    else:
      padding_mask_downsampled = jax.image.resize(
          padding_mask.astype(jnp.float32), shape=[bs, h, w],
          method='nearest').astype(jnp.bool_)

    pos_emb = InputPosEmbeddingSine(hidden_dim=self.hidden_dim)(
        padding_mask_downsampled)

    query_pos_emb = QueryPosEmbedding(
        hidden_dim=self.hidden_dim, num_queries=self.num_queries)()

    # Project and reshape features to 3 dimensions.
    x = nn.Conv(
        features=self.hidden_dim,
        kernel_size=(1, 1),
        strides=(1, 1),
        dtype=self.dtype)(
            x)
    x = x.reshape(bs, h * w, self.hidden_dim)

    transformer_input = x

    transformer = DETRTransformer(
        num_queries=self.num_queries,
        query_emb_size=self.query_emb_size,
        num_heads=self.transformer_num_heads,
        num_encoder_layers=self.transformer_num_encoder_layers,
        num_decoder_layers=self.transformer_num_decoder_layers,
        qkv_dim=self.transformer_qkv_dim,
        mlp_dim=self.transformer_mlp_dim,
        return_intermediate_dec=self.aux_loss,
        normalize_before=self.transformer_normalize_before,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype,
        name='transformer')
    decoder_output, encoder_output = transformer(
        transformer_input,
        padding_mask=jnp.reshape(padding_mask_downsampled, [bs, h * w]),
        pos_embedding=pos_emb,
        query_pos_emb=query_pos_emb,
        train=train)

    def output_projection(model_output):
      # classification head
      pred_logits = ObjectClassPredictor(
          num_classes=self.num_classes, dtype=self.dtype)(
              model_output)
      pred_boxes = BBoxCoordPredictor(
          mlp_dim=self.hidden_dim, dtype=self.dtype)(
              model_output)
      return pred_logits, pred_boxes

    if not self.aux_loss:
      pred_logits, pred_boxes = output_projection(decoder_output)
      return {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}

    pred_logits, pred_boxes = jax.vmap(output_projection)(decoder_output)
    output = {
        'pred_logits': pred_logits[-1],
        'pred_boxes': pred_boxes[-1],
        'transformer_input': transformer_input,
        'backbone_features': backbone_features,
        'encoder_output': encoder_output,
        'decoder_output': decoder_output[-1],
        'padding_mask': padding_mask_downsampled,
    }
    if self.aux_loss:
      output['aux_outputs'] = []
      for lgts, bxs in zip(pred_logits[:-1], pred_boxes[:-1]):
        output['aux_outputs'].append({'pred_logits': lgts, 'pred_boxes': bxs})

    return output


class DETRModel(detr_base_model.ObjectDetectionWithMatchingModel):

  def build_flax_model(self):
    return DETR(
        num_classes=self.dataset_meta_data['num_classes'],
        hidden_dim=self.config.get('hidden_dim', 512),
        num_queries=self.config.get('num_queries', 100),
        query_emb_size=self.config.get('query_emb_size', None),
        transformer_num_heads=self.config.get('transformer_num_heads', 8),
        transformer_num_encoder_layers=self.config.get(
            'transformer_num_encoder_layers', 6),
        transformer_num_decoder_layers=self.config.get(
            'transformer_num_decoder_layers', 6),
        transformer_qkv_dim=self.config.get('transformer_qkv_dim', 512),
        transformer_mlp_dim=self.config.get('transformer_mlp_dim', 2048),
        transformer_normalize_before=self.config.get(
            'transformer_normalize_before', False),
        backbone_width=self.config.get('backbone_width', 1),
        backbone_depth=self.config.get('backbone_depth', 50),
        aux_loss=self.config.get('aux_loss', False),
        dropout_rate=self.config.get('dropout_rate', 0.0),
        attention_dropout_rate=self.config.get('attention_dropout_rate', 0.0),
        dtype=self.config.get('model_dtype_str', jnp.float32))
