"""DETR Loss, Metrics and Model container."""
from typing import Any, Dict, Tuple, Sequence, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
import ml_collections

from model_lib.base_models import base_model, box_utils
from model_lib import matchers
from projects.detr.model import DETR

ArrayDict = Dict[str, jnp.ndarray]
MetricsDict = Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]


def compute_cost(*,
                 tgt_labels: jnp.ndarray,
                 out_prob: jnp.ndarray,
                 tgt_bbox: Optional[jnp.ndarray] = None,
                 out_bbox: Optional[jnp.ndarray] = None,
                 class_loss_coef: float,
                 bbox_loss_coef: Optional[float] = None,
                 giou_loss_coef: Optional[float] = None,
                 target_is_onehot: bool = False) -> jnp.ndarray:
  """Computes cost matrices for a batch of predictions.

  Relevant code:
  https://github.com/facebookresearch/detr/blob/647917626d5017e63c1217b99537deb2dcb370d6/models/matcher.py#L35 
  
  Args:
    tgt_labels: Class labels of shape [B, M]. If `target_is_onehot` then [B, M, C].
      Note that the labels corresponding to empty bounding boxes are not yet
      supposed to be filtered out.
    out_prob: Classification probabilities of shape [B, N, C].
    tgt_bbox: Target box coordinates of shape [B, M, 4]. Note that the empty
      bounding boxes are not yet supposed to be filtered out.
    out_bbox: Predicted bbox coordinates of shape [B, N, 4].
    class_loss_coef: Relative weight of classification loss.
    bbox_loss_coef: Relative weight of the bbox loss.
    giou_loss_coef: Relative weight of the giou loss.
    target_is_onehot: Whether targets are one-hot encoded.
  
  Returns:
    A cost matrix [B, N, M].
  """
  if (tgt_bbox is None) != (out_bbox is None):
    raise ValueError('Both `tgt_bbox` and `out_bbox` must be set.')
  if (tgt_bbox is not None) and ((bbox_loss_coef is None) or
                                 (giou_loss_coef is None)):
    raise ValueError('For detection, both `bbox_loss_coef` and `giou_loss_coef`'
                     ' must be set.')

  batch_size, max_num_boxes = tgt_labels.shape[:2]
  num_queries = out_prob.shape[1]
  if target_is_onehot:
    mask = tgt_labels[..., 0] == 0  # All instances NOT padding.
  else:
    mask = tgt_labels != 0

  # [B, N, M]
  cost_class = 1 - out_prob
  max_cost_class = 0.0
  if target_is_onehot:
    cost_class = jnp.einsum('bnl,bml->bnm', cost_class, tgt_labels)
  else:
    cost_class = jax.vmap(jnp.take, (0, 0, None))(cost_class, tgt_labels, 1)

  cost = class_loss_coef * cost_class
  cost_upper_bound = class_loss_coef * max_cost_class

  if out_bbox is not None:
    # Compute L1 cost between boxes.
    diff = jnp.abs(out_bbox[:, :, None, :] - tgt_bbox[:, None, :, :])
    cost_bbox = jnp.sum(diff, axis=-1)
    cost = cost + bbox_loss_coef * cost_bbox

    # Update upper bounds.
    cost_upper_bound = cost_upper_bound + bbox_loss_coef * 4.0

    # Compute Generalized IoU loss between boxes.
    cost_giou = -box_utils.generalized_box_iou(
        box_utils.box_cxcywh_to_xyxy(out_bbox),
        box_utils.box_cxcywh_to_xyxy(tgt_bbox),
        all_pairs=True)
    cost = cost + giou_loss_coef * cost_giou

    # Update upper bounds.
    cost_upper_bound = cost_upper_bound + giou_loss_coef * 1.0

  mask = mask[:, None, :]
  cost = cost * mask + (1.0 - mask) * cost_upper_bound

  # Guard against NaNs and Infs.
  cost = jnp.nan_to_num(
      cost,
      nan=cost_upper_bound,
      neginf=cost_upper_bound,
      posinf=cost_upper_bound)
  assert cost.shape == (batch_size, num_queries, max_num_boxes)

  # Compute the number of unpadded columns for each batch element.
  # It is assumed that all padding is trailing padding.
  n_cols = jnp.sum(mask, axis=-1).squeeze()
  return cost, n_cols


def _targets_from_batch(
    batch: ArrayDict,
    target_is_onehot: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Get target labels and boxes with additional non-object appended."""
  # Append the no-object class label so we are always guaranteed one.
  tgt_labels = batch['label']['labels']
  tgt_boxes = batch['label']['boxes']

  # Append a class label.
  if target_is_onehot:
    # Shape is [batch_size, num_instances, num_classes]
    num_classes = tgt_labels.shape[-1]
    instance = jax.nn.one_hot(0, num_classes)
    reshape_shape = (1,) * (len(tgt_labels.shape) - 1) + (num_classes,)
    instance = instance.reshape(reshape_shape)
    broadcast_shape = tgt_labels.shape[:-2] + (1, num_classes)
    instance = jnp.broadcast_to(instance, broadcast_shape)
  else:
    # Shape is [batch_size, num_instances]
    instance = jnp.zeros_like(tgt_labels[..., :1])
  tgt_labels = jnp.concatenate([tgt_labels, instance], axis=1)

  # Same for boxes.
  instance = jnp.zeros_like(tgt_boxes[..., :1, :])
  tgt_boxes = jnp.concatenate([tgt_boxes, instance], axis=1)
  return tgt_labels, tgt_boxes


def logits_to_probs(logits: jnp.ndarray,
                    log_p: bool = False,
                    is_sigmoid: bool = False) -> jnp.ndarray:
  # TODO: No override for sigmoid logit normalization.
  if not is_sigmoid:
    return jax.nn.log_softmax(logits) if log_p else jax.nn.softmax(logits)
  else:
    return jax.nn.log_sigmoid(logits) if log_p else jax.nn.sigmoid(logits)


class DETRModel(base_model.BaseModel):
  """DETR model for object detection."""

  def __init__(self, config: ml_collections.ConfigDict,
               dataset_meta_data: Dict[str, Any]):
    """Initialize DETR model class.
    
    Args:
      config: Model config.
      dataset_meta_data: Dataset meta data specifies `target_is_onehot`, which
        is False by default. The padded objects have label 0. The first 
        legitimate object has label 1, and so on.
    """
    if config is not None:
      self.loss_terms_weights = {
          'loss_class': config.class_loss_coef,
          'loss_bbox': config.bbox_loss_coef,
          'loss_giou': config.giou_loss_coef
      }
    super().__init__(config, dataset_meta_data)

  def build_flax_model(self) -> nn.Module:
    return DETR(
        num_classes=self.config.num_classes,
        hidden_dim=self.config.hidden_dim,
        num_queries=self.config.num_queries,
        query_emb_size=self.config.query_emb_size,
        transformer_num_heads=self.config.transformer_num_heads,
        transformer_num_encoder_layers=self.config.transformer_num_encoder_layers,
        transformer_num_decoder_layers=self.config.transformer_num_decoder_layers,
        transformer_mlp_dim=self.config.transformer_mlp_dim,
        transformer_normalize_before=self.config.transformer_normalize_before,
        backbone_width=self.config.backbone_width,
        backbone_depth=self.config.backbone_depth,
        aux_loss=self.config.aux_loss,
        dropout_rate=self.config.dropout_rate,
        attention_dropout_rate=self.config.attention_dropout_rate,
        dtype=self.config.model_dtype_str)  # yapf: disable

  def compute_loss_for_layer(
      self,
      tgt_labels: jnp.ndarray,
      pred_logits: jnp.ndarray,
      tgt_boxes: jnp.ndarray,
      pred_boxes: jnp.ndarray,
      indices: Optional[jnp.ndarray] = None) -> ArrayDict:
    """Loss and metrics function for single prediction layer."""
    target_is_onehot = self.dataset_meta_data.get('target_is_onehot', False)
    if not indices:
      pred_prob = logits_to_probs(
          pred_logits, is_sigmoid=self.config.get('sigmoid_loss', False))
      cost, n_cols = compute_cost(
          tgt_labels=tgt_labels,
          out_prob=pred_prob,
          tgt_bbox=tgt_boxes,
          out_bbox=pred_boxes,
          class_loss_coef=self.config.class_loss_coef,
          bbox_loss_coef=self.config.bbox_loss_coef,
          giou_loss_coef=self.config.giou_loss_coef,
          target_is_onehot=target_is_onehot)
      indices = matchers.hungarian_matcher(cost, n_cols=n_cols)
    
    losses = {}
    # Class loss.
    losses.update(
      loss_labels()
    )

    # Boxes loss.
    losses.update(
      loss_boxes()
    )
    return losses

  def loss_and_metrics_function(
      self,
      outputs: ArrayDict,
      batch: ArrayDict,
      matches: Optional[Sequence[jnp.ndarray]] = None
  ) -> Tuple[jnp.ndarray, MetricsDict]:
    """Loss and metrics function for DETR.
    
    Args:
      outputs: Model predictions. Exact fields expected depends on the losses
        used.
      batch: Dict that has `inputs`, `batch_mask` and `label` (ground truth).
        `batch['label']` is a dict where the keys and values depends on the 
        losses used.
      matches: Possibly pass in matches if already done.
    
    Returns:
      total_loss: Total loss weighed appropriately.
      metrics_dict: Individual loss terms for logging purposes.
    """
    tgt_labels, tgt_boxes = _targets_from_batch(
        batch, self.config.get('target_is_onehot', False))

    indices, aux_indices = None, None
    if matches:
      indices, *aux_indices = matches
    losses = self.compute_loss_for_layer(
        pred_logits=outputs['pred_logits'],
        tgt_labels=tgt_labels,
        pred_boxes=outputs['pred_boxes'],
        tgt_boxes=tgt_boxes,
        indices=indices)
