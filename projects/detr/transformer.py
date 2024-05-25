"""DETR Transformer Modules."""
import functools
from typing import Any, Callable, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp


def mask_for_shape(shape: jnp.shape,
                   padding_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Resizes `padding_mask` to `shape`."""
  bs, h, w, _ = shape
  if padding_mask is None:
    resized_mask = jnp.ones((bs, h, w), dtype=jnp.bool_)
  else:
    resized_mask = jax.image.resize(
        padding_mask.astype(jnp.float32), (bs, h, w),
        method='nearest').astype(jnp.bool_)
  return resized_mask


class InputProj(nn.Module):
  """Simple input projection layer.
  
  Attributes:
    embed_dim: Size of the output embedding dimension.
    num_groups: Number of channel groups for group norm.
    kernel_size: Convolution kernel size.
    stride: Stride in all dimensions.
    padding: Same as nn.Conv padding type.
    dtype: DType of the computation (default: float32).
  """
  embed_dim: int
  num_groups: int = 32
  kernel_size: int = 1
  stride: int = 1
  padding: Union[str, int] = 'SAME'
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray):
    """Use conv kernel to project into embed_dim."""
    if isinstance(self.padding, str):
      padding = self.padding
    else:
      padding = [self.padding] * 2
    x = nn.Conv(
        features=self.embed_dim,
        kernel_size=[self.kernel_size] * 2,
        strides=[self.stride] * 2,
        padding=padding,
        kernel_init=nn.initializers.glorot_uniform(),
        bias_init=nn.initializers.zeros,
    )(x)  # yapf: disable
    x = nn.GroupNorm(num_groups=self.num_groups)(x)


class InputPosEmbeddingSine(nn.Module):
  """Creates sinusoidal positional embeddings for the inputs."""
  hidden_dim: int
  dtype: jnp.dtype = jnp.float32
  scale: Optional[float] = None
  temperature: float = 10_000

  @nn.compact
  def __call__(self, padding_mask: jnp.ndarray) -> jnp.ndarray:
    """Creates positional embeddings for transformer inputs.
    
    Args:
      padding_mask: Binary matrix with 0 at padded image regions. Shape is
        [batch, height, width].
    
    Returns:
      Positional embeddings for the inputs.
    
    Raises:
      ValueError if `hidden_dim` is not an even number.
    """
    if self.hidden_dim % 2 != 0:
      raise ValueError('`hidden_dim` must be an even number.')

    mask = padding_mask.astype(jnp.float32)
    y_embed = jnp.cumsum(mask, axis=1)
    x_embed = jnp.cumsum(mask, axis=2)

    # Normalization
    eps = 1e-6
    scale = self.scale or 2 * jnp.pi
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    num_pos_feats = self.hidden_dim // 2
    dim_t = jnp.arange(num_pos_feats, dtype=jnp.float32)
    dim_t = self.temperature**(2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, jnp.newaxis] / dim_t
    pos_y = y_embed[:, :, :, jnp.newaxis] / dim_t

    pos_x = jnp.stack([jnp.sin(pos_x[..., 0::2]),
                       jnp.cos(pos_x[..., 1::2])],
                      axis=4).reshape(padding_mask.shape + (-1,))
    pos_y = jnp.stack([jnp.sin(pos_y[..., 0::2]),
                       jnp.cos(pos_y[..., 1::2])],
                      axis=4).reshape(padding_mask.shape + (-1,))

    pos = jnp.concatenate([pos_y, pos_x], axis=3)
    b, h, w = padding_mask.shape
    pos = jnp.reshape(pos, [b, h * w, self.hidden_dim])
    return jnp.asarray(pos, self.dtype)


class QueryPosEmbedding(nn.Module):
  """Creates learnt positional embeddings for object queries.
  
  Attributes:
    hidden_dim: Hidden dimension for the embeddings.
    num_queries: Number of object queries.
    posemb_init: Positional embeddings initializer.
    dtype: DType of the computation (default: float32).
  """
  hidden_dim: int
  num_queries: int
  posemb_init: Callable[..., Any] = jax.nn.initializers.normal(stddev=1.0)
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self) -> jnp.ndarray:
    query_pos = self.param('query_emb', self.posemb_init,
                           (self.num_queries, self.hidden_dim))
    query_pos = jnp.expand_dims(query_pos, 0)
    return jnp.asarray(query_pos, dtype=self.dtype)


class MlpBlock(nn.Module):
  """MLP/Feedforward block.
  
  Args:
    mlp_dim: Size of the input MLP dim.
    out_dim: Size of the output dim (infers from input by default.)
    activation_fn: Nonlinearity between layers.
    dropout_rate: Dropout rate.
    dtype: DType of the computation (default float32).
  """
  mlp_dim: Optional[int] = 2048
  out_dim: Optional[int] = None
  activation_fn: Callable[..., Any] = nn.gelu
  dropout_rate: float = 0.
  kernel_init: Callable[..., Any] = nn.initializers.xavier_uniform()
  bias_init: Callable[..., Any] = nn.initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               deterministic: bool = False) -> jnp.ndarray:
    out_dim = self.out_dim or inputs.shape[-1]
    dense = functools.partial(
        nn.Dense,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dtype=self.dtype)

    x = dense(self.mlp_dim)(inputs)
    x = self.activation_fn(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = dense(out_dim)(x)
    output = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    return output


class EncoderBlock(nn.Module):
  """DETR Transformer Encoder Block.
  
  Attributes:
    num_heads: Number of attention heads.
    qkv_dim: Dimension of query/key/value (also called hidden_dim).
    mlp_dim: Dimension of MLP after attention block.
    normalize_before: Whether to apply LayerNorm before attention/MLP.
    dropout_rate: Dropout rate of MLP/post-attention.
    attention_dropout_rate: Dropout rate for attention weights.
    dtype: DType of the computation (default: float32).
  """
  num_heads: int
  qkv_dim: int
  mlp_dim: int
  normalize_before: bool = False
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               pos_embedding: Optional[jnp.ndarray] = None,
               padding_mask: Optional[jnp.ndarray] = None,
               train: bool = False) -> jnp.ndarray:
    """
    Args:
      inputs: Input data of shape [bs, len, features].
      pos_embedding: Positional embedding to be added to the queries and keys
        in the self-attention operation.
      padding_mask: Binary mask containing 0 for padded tokens.
      train: If the model is training (for dropout).
    
    Returns:
      Output after transformer encoder block.
    """
    assert inputs.ndim == 3
    self_attn = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.qkv_dim,
        dropout_rate=self.attention_dropout_rate,
        broadcast_dropout=False,
        dtype=self.dtype)

    mlp = MlpBlock(
        mlp_dim=self.mlp_dim,
        activation_fn=nn.relu,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype)

    def add_positional_embedding(x, pos_emb_x):
      return x if pos_emb_x is None else x + pos_emb_x

    if self.normalize_before:
      x = nn.LayerNorm(dtype=self.dtype)(inputs)
      x = self_attn(
          inputs_q=add_positional_embedding(x, pos_embedding),
          inputs_k=add_positional_embedding(x, pos_embedding),
          inputs_v=x,
          mask=padding_mask,
          deterministic=not train)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = x + inputs
      y = nn.LayerNorm(dtype=self.dtype)(x)
      y = mlp(y, deterministic=not train)
      out = x + y
    else:
      x = self_attn(
          inputs_q=add_positional_embedding(inputs, pos_embedding),
          inputs_k=add_positional_embedding(inputs, pos_embedding),
          inputs_v=inputs,
          mask=padding_mask,
          deterministic=not train)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = x + inputs
      x = nn.LayerNorm(dtype=self.dtype)(x)
      y = mlp(x)
      y = nn.LayerNorm(dtype=self.dtype)(y)
      out = x + y

    return out


class DecoderBlock(nn.Module):
  """DETR Transformer Decoder block.
  
  Attributes:
    num_heads: Number of attention heads.
    qkv_dim: Dimension of query/key/value (also called hidden_dim).
    mlp_dim: Dimension of MLP after attention block.
    normalize_before: Whether to apply LayerNorm before attention/MLP.
    dropout_rate: Dropout rate of MLP/post-attention.
    attention_dropout_rate: Dropout rate for attention weights.
    dtype: DType of the computation (default: float32).
  """
  num_heads: int
  qkv_dim: int
  mlp_dim: int
  normalize_before: bool = False
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.
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
      encoder_output: Output of DETR encoder.
      pos_embedding: Positional encoding used by encoder for cross-attention.
      query_pos_emb: Learned object positional embeddings to be added to the queries.
      key_padding_mask: Token mask of encoder output, applied to keys.
      train: If the model is training (to apply dropout).
    
    Returns:
      Output after decoder block.
    """
    assert encoder_output.ndim == 3
    assert query_pos_emb is not None, ('Given that object queries are zeros '
                                       'and not learnable, we should add '
                                       'learnable query_pos_emb to them.')
    # First decoder self-attention is practically useless (zero inputs)
    self_attn = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.qkv_dim,
        dropout_rate=self.attention_dropout_rate,
        broadcast_dropout=False,
        dtype=self.dtype)

    cross_attn = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.qkv_dim,
        dropout_rate=self.attention_dropout_rate,
        broadcast_dropout=False,
        dtype=self.dtype)

    mlp = MlpBlock(
        mlp_dim=self.mlp_dim,
        activation_fn=nn.relu,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype)

    def add_positional_embedding(x, pos_emb_x):
      return x if pos_emb_x is None else x + pos_emb_x

    if self.normalize_before:
      # Self attention block.
      x = nn.LayerNorm(dtype=self.dtype)(obj_queries)
      x = self_attn(
          inputs_q=add_positional_embedding(x, query_pos_emb),
          inputs_k=add_positional_embedding(x, query_pos_emb),
          inputs_v=x,
          deterministic=not train)
      x = nn.Dropout(rate=self.dropout_rate)(deterministic=not train)
      x = x + obj_queries

      # Cross attention block.
      y = nn.LayerNorm(dtype=self.dtype)(x)
      y = cross_attn(
          inputs_q=add_positional_embedding(y, query_pos_emb),
          inputs_k=add_positional_embedding(encoder_output, pos_embedding),
          inputs_v=encoder_output,
          mask=key_padding_mask,
          deterministic=not train)
      y = nn.Dropout(rate=self.dropout_rate)(deterministic=not train)
      y = y + x

      # MLP Block.
      z = nn.LayerNorm(dtype=self.dtype)(y)
      z = mlp(z, deterministic=not train)
      out = z + y

    else:
      # Self attention block.
      x = self_attn(
          inputs_q=add_positional_embedding(obj_queries, query_pos_emb),
          inputs_k=add_positional_embedding(obj_queries, query_pos_emb),
          inputs_v=obj_queries,
          deterministic=not train)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = x + obj_queries
      x = nn.LayerNorm(dtype=self.dtype)(x)

      # Cross attention block.
      y = cross_attn(
          inputs_q=add_positional_embedding(x, query_pos_emb),
          inputs_k=add_positional_embedding(encoder_output, pos_embedding),
          inputs_v=encoder_output,
          mask=key_padding_mask,
          deterministic=not train)
      y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
      y = y + x
      y = nn.LayerNorm(dtype=self.dtype)(y)

      # MLP Block.
      z = mlp(y, deterministic=not train)
      z = z + y
      out = nn.LayerNorm(dtype=self.dtype)(z)

    return out


class Encoder(nn.Module):
  """Sequence of `EncoderBlock`s for DETR.
  
  Attributes:
    num_heads: Number of attention heads.
    num_layers: Number of encoder blocks applied sequentially.
    qkv_dim: Dimension of query/key/value (also called hidden_dim).
    mlp_dim: Dimension of MLP after attention block.
    normalize_before: Whether to apply LayerNorm before attention/MLP.
    dropout_rate: Dropout rate of MLP/post-attention.
    attention_dropout_rate: Dropout rate for attention weights.
    dtype: DType of the computation (default: float32).
  """
  num_heads: int
  num_layers: int
  qkv_dim: int
  mlp_dim: int
  normalize_before: bool = False
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               padding_mask: jnp.ndarray,
               pos_embedding: jnp.ndarray,
               train: bool = False) -> jnp.ndarray:
    assert inputs.ndim == 3
    x = inputs
    for lyr in range(self.num_layers):
      x = EncoderBlock(
          num_heads=self.num_heads,
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          normalize_before=self.normalize_before,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          dtype=self.dtype)(
              x,
              pos_embedding=pos_embedding,
              padding_mask=padding_mask,
              train=train)

    if self.normalize_before:
      x = nn.LayerNorm(dtype=self.dtype)(x)

    return x


class Decoder(nn.Module):
  """Sequence of `DecoderBlock`s for DETR.
  
  Attributes:
    num_heads: Number of attention heads.
    num_layers: Number of encoder blocks applied sequentially.
    qkv_dim: Dimension of query/key/value (also called hidden_dim).
    mlp_dim: Dimension of MLP after attention block.
    return_intermediate: Whether to return outputs of each decoder layer.
    normalize_before: Whether to apply LayerNorm before attention/MLP.
    dropout_rate: Dropout rate of MLP/post-attention.
    attention_dropout_rate: Dropout rate for attention weights.
    dtype: DType of the computation (default: float32).
  """
  num_heads: int
  num_layers: int
  qkv_dim: int
  mlp_dim: int
  return_intermediate: bool = True
  normalize_before: bool = False
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.
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
    assert obj_queries.ndim == 3
    assert encoder_output.ndim == 3

    y = obj_queries
    outputs = []
    for lyr in range(self.num_layers):
      y = DecoderBlock(
          num_heads=self.num_heads,
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          normalize_before=self.normalize_before,
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

    # Decoder norm is always applied (and is common) as per authors' impl
    # https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/transformer.py#L33
    y = nn.LayerNorm(dtype=self.dtype)(y)
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
    return_intermediate: Whether to return outputs of intermediate decoder 
      blocks.
    normalize_before: If use LayerNorm before attention/mlp blocks.
    dropout_rate: Dropout rate of MLP/post-attention.
    attention_dropout_rate: Dropout rate for attention weights.
    dtype: DType of the computation (defaults to float32).
  """
  num_queries: int
  query_emb_size: Optional[int] = None
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  qkv_dim: int = 256
  mlp_dim: int = 2048
  return_intermediate: bool = True
  normalize_before: bool = False
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               padding_mask: Optional[jnp.ndarray] = None,
               train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Applies DETR on the inputs.
    
    Args:
      inputs: Tokenized image patches of shape [bs, num_tokens, hidden_dim].
      padding_mask: Boolean mask with same shape as inputs.
      train: If the model is training.
    
    Returns:
      Tuple of Encoder and Decoder outputs.
    """
    pos_embedding = InputPosEmbeddingSine(self.qkv_dim)
    encoder_output = Encoder(
        num_heads=self.num_heads,
        num_layers=self.num_encoder_layers,
        qkv_dim=self.qkv_dim,
        mlp_dim=self.mlp_dim,
        normalize_before=self.normalize_before,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype,
        name='encoder')(
            inputs, padding_mask=padding_mask, train=train)

    # Note that we always learn a query positional embedding, so we simply use
    # constant zero vectors for object queries, and later when applying attention
    # we have query = query_pos_emb + obj_queries
    query_dim = self.query_emb_size or inputs.shape[-1]
    obj_query_shape = tuple([inputs.shape[0], self.num_queries, query_dim])
    obj_queries = jnp.zeros(obj_query_shape)
    query_pos_emb = QueryPosEmbedding(query_dim, self.num_queries)()

    decoder_output = Decoder(
        num_heads=self.num_heads,
        num_layers=self.num_decoder_layers,
        qkv_dim=self.qkv_dim,
        mlp_dim=self.mlp_dim,
        return_intermediate=self.return_intermediate,
        normalize_before=self.normalize_before,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        name='decoder',
        dtype=self.dtype)(
            obj_queries,
            encoder_output,
            key_padding_mask=padding_mask,
            pos_embedding=pos_embedding,
            query_pos_emb=query_pos_emb,
            train=train)

    return encoder_output, decoder_output
