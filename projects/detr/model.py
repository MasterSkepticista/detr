"""Implementation of DETR."""
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from models import resnet
from projects.detr.transformer import DETRTransformer, InputProj, mask_for_shape

ArrayDict = Dict[str, jnp.ndarray]
MetricsDict = Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]


class ObjectClassPredictor(nn.Module):
  """DETR Classification Head.
  
  Attributes:
    num_classes: Number of output classes.
    dtype: DType of the computation (default: float32).
  """
  num_classes: int
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies Linear projection to inputs."""
    # TODO: Use custom init?
    return nn.Dense(self.num_classes, dtype=self.dtype)(inputs)


class BBoxCoordPredictor(nn.Module):
  """DETR Bounding Box Regression Head."""
  mlp_dim: int
  num_layers: int = 3
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    # TODO: Custom init?
    x = inputs
    for _ in range(self.num_layers - 1):
      x = nn.Dense(self.mlp_dim, dtype=self.dtype)(x)
      x = nn.relu(x)

    x = nn.Dense(4, dtype=self.dtype)(x)
    out = nn.sigmoid(x)
    return out


class DETR(nn.Module):
  """DETR.
  
  Attributes:
    num_classes: Number of classes to predict.
    hidden_dim: Size of the hidden embedding dimension (same for qkv_dim).
    num_queries: Number of object queries. This is the maximal number of objects
      this model can detect in a single frame.
    query_emb_size: Size of the embedding learned for object queries.
    transformer_num_heads: Number of heads.
    transformer_num_encoder_layers: Number of transformer encoder layers.
    transformer_num_decoder_layers: Number of transformer decoder layers.
    transformer_mlp_dim: Dimension of the MLP above attention block.
    transformer_normalize_before: If use LayerNorm before attention/mlp blocks.
    backbone_width: Backbone ResNet width (Defaults to 1).
    backbone_depth: Backbone ResNet depth (Defaults to 50).
    aux_loss: If train with auxiliary loss.
    dropout_rate: Dropout rate of the MLP/post-attention.
    attention_dropout_rate: Attention dropout rate.
    dtype: Data type of the computation (default: float32).
  """
  num_classes: int
  hidden_dim: int = 256
  num_queries: int = 100
  query_emb_size: Optional[int] = None
  transformer_num_heads: int = 8
  transformer_num_encoder_layers: int = 6
  transformer_num_decoder_layers: int = 6
  transformer_mlp_dim: int = 2048
  transformer_normalize_before: bool = False
  backbone_width: int = 1
  backbone_depth: Union[int, Sequence[int]] = 50
  aux_loss: bool = True
  dropout_rate: float = 0.1
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
      inputs: Input image batch.
      train: Whether it is training.
      padding_mask: Binary matrix with 0 at padded image regions.
      update_batch_stats: Whether to update batch statistics for the batchnorms
        in the backbone (if any). If None, value of `train` flag will be used.
    
    Returns:
      Output dict that has `pred_logits` and `pred_boxes` and potentially 
      `aux_outputs`.
    """
    update_batch_stats = update_batch_stats or train

    # Extract image features from backbone
    _, features = resnet.ResNet(
        width=self.backbone_width,
        depth=self.backbone_depth,
        dtype=self.dtype,
        name='backbone')(
            inputs, train=update_batch_stats)
    x = features['stage_4']
    bs, h, w, _ = x.shape

    # Resize padding mask to image features shape
    padding_mask = mask_for_shape(x.shape, padding_mask=padding_mask)

    # Tokenize image features
    transformer_input = InputProj(
        embed_dim=self.hidden_dim,
        kernel_size=1,
        stride=1,
        dtype=self.dtype,
        name='input_proj')(x).reshape(bs, h * w, self.hidden_dim)

    # Pass features through transformer.
    encoder_output, decoder_output = DETRTransformer(
        num_queries=self.num_queries,
        query_emb_size=self.query_emb_size or self.hidden_dim,
        num_heads=self.transformer_num_heads,
        num_encoder_layers=self.transformer_num_encoder_layers,
        num_decoder_layers=self.transformer_num_decoder_layers,
        qkv_dim=self.hidden_dim,
        mlp_dim=self.transformer_mlp_dim,
        return_intermediate=self.aux_loss,
        normalize_before=self.transformer_normalize_before,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype,
        name='transformer')(
            inputs=transformer_input,
            padding_mask=padding_mask.reshape(bs, h * w),
            train=train)

    def output_projection(model_output):
      pred_logits = ObjectClassPredictor(
          self.num_classes, dtype=self.dtype)(
              model_output)
      pred_boxes = BBoxCoordPredictor(
          mlp_dim=self.hidden_dim, dtype=self.dtype)(
              model_output)
      return pred_logits, pred_boxes

    pred_logits, pred_boxes = jax.vmap(output_projection)(decoder_output)
    output = {
        'pred_logits': pred_logits[-1],
        'pred_boxes': pred_boxes[-1],
        'transformer_input': transformer_input,
        'backbone_features': features,
        'encoder_output': encoder_output,
        'decoder_output': decoder_output[-1],
        'padding_mask': padding_mask,
    }
    if self.aux_loss:
      output['aux_outputs'] = []
      for lgts, bxs in zip(pred_logits[:-1], pred_boxes[:-1]):
        output['aux_outputs'].append({'pred_logits': lgts, 'pred_boxes': bxs})

    return output
