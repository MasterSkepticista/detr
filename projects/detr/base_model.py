"""DETR Loss, Metrics and Model container."""
import functools
from typing import Any, Dict, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from absl import logging
from model_lib import matchers
from model_lib.base_models import base_model, box_utils, model_utils
from projects.detr.model import DETR

ArrayDict = Dict[str, jnp.ndarray]
MetricsDict = Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]


def compute_cost(
    *,
    tgt_labels: jnp.ndarray,
    out_prob: jnp.ndarray,
    tgt_bbox: Optional[jnp.ndarray] = None,
    out_bbox: Optional[jnp.ndarray] = None,
    class_loss_coef: float,
    bbox_loss_coef: Optional[float] = None,
    giou_loss_coef: Optional[float] = None,
    target_is_onehot: bool = False,
) -> jnp.ndarray:
  """Computes cost matrices for a batch of predictions.
  
  Relevant code:
  https://github.com/facebookresearch/detr/blob/647917626d5017e63c1217b99537deb2dcb370d6/models/matcher.py#L35 

  Args:
    tgt_labels: Class labels of shape [B, M]. If target_is_onehot then [B, M, C].
      Note that the labels corresponding to empty bounding boxes are not yet
      supposed to be filtered out.
    out_prob: Classification probabilites of shape [B, N, C].
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
  # [B, N, M]
  cost_class = -out_prob  # Actually, (1-out_prob), but constants are discarded.
  if target_is_onehot:
    cost_class = jnp.einsum('bnl,bml->bnm', cost_class, tgt_labels)
  else:
    cost_class = jax.vmap(jnp.take, (0, 0, None))(cost_class, tgt_labels, 1)

  # Compute L1 cost between boxes.
  diff = jnp.abs(out_bbox[:, :, None] - tgt_bbox[:, None, :])
  cost_bbox = jnp.sum(diff, axis=-1)

  # Compute GIoU cost between boxes.
  cost_giou = -box_utils.generalized_box_iou(
      box_utils.box_cxcywh_to_xyxy(out_bbox),
      box_utils.box_cxcywh_to_xyxy(tgt_bbox),
      all_pairs=True)

  total_cost = (
      class_loss_coef * cost_class + bbox_loss_coef * cost_bbox +
      giou_loss_coef * cost_giou)

  # Compute the number of valid boxes for each batch element.
  # It is assumed that all padding is trailing padding.
  if target_is_onehot:
    tgt_not_padding = tgt_labels[..., 0] == 0  # All instances not padding.
  else:
    tgt_not_padding = tgt_labels != 0
  
  total_cost *= tgt_not_padding[:, None, :]
  n_cols = jnp.sum(tgt_not_padding, axis=-1)
  return total_cost, n_cols


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
  tgt_labels = jnp.concatenate([tgt_labels, instance], axis=-1)

  # Same for boxes.
  instance = jnp.zeros_like(tgt_boxes[..., :1, :])
  tgt_boxes = jnp.concatenate([tgt_boxes, instance], axis=-2)
  return tgt_labels, tgt_boxes


def loss_labels(*,
                pred_logits: jnp.ndarray,
                tgt_labels: jnp.ndarray,
                indices=jnp.ndarray,
                class_loss_coef: float = 1.0,
                eos_coef: float = 0.1,
                batch_mask: Optional[jnp.ndarray] = None,
                target_is_onehot: bool = False) -> ArrayDict:
  """Calculate DETR classification loss.
  
  Args:
    pred_logits: [bs, num_queries, num_classes].
    tgt_labels: [bs, max_objects].
    indices: [bs, 2, min(num_queries, max_objects)].
    class_loss_coef: Classification loss coefficient.
    eos_coef: Label weight for padded/no object.
    batch_mask: Optional [bs, max_objects] boolean mask of valid objects 
      and examples.
    target_is_onehot: If tgt is [bs, max_objects, num_classes].

  Returns:
    `loss_class`: Classification loss with coefficient applied.
  """
  # Apply the permutation communicated by the indices.
  pred_logits = model_utils.simple_gather(pred_logits, indices[:, 0, :])
  tgt_labels = model_utils.simple_gather(tgt_labels, indices[:, 1, :])

  num_classes = pred_logits.shape[-1]
  if target_is_onehot:
    tgt_labels_onehot = tgt_labels
  else:
    tgt_labels_onehot = jax.nn.one_hot(tgt_labels, num_classes)

  pred_log_p = jax.nn.log_softmax(pred_logits)

  # We weight padded objects differently, so class 0 gets eos_coef, while valid
  # object classes get 1.0
  label_weights = jnp.concatenate([
      jnp.array([eos_coef], dtype=jnp.float32),
      jnp.ones(num_classes - 1),
  ])
  loss = model_utils.weighted_unnormalized_softmax_cross_entropy(
      pred_log_p,
      tgt_labels_onehot,
      weights=batch_mask,
      label_weights=label_weights,
      logits_normalized=True)
  loss = loss.sum()
  loss = class_loss_coef * loss

  return {'loss_class': loss}


def loss_boxes(*,
               src_boxes: jnp.ndarray,
               tgt_boxes: jnp.ndarray,
               tgt_labels: jnp.ndarray,
               indices: jnp.ndarray,
               bbox_loss_coef: float = 5.0,
               giou_loss_coef: float = 1.0,
               target_is_onehot: bool = False) -> ArrayDict:
  """Calculate DETR L1 box loss and GIoU loss.
  
  Args:
    src_boxes: [bs, num_queries, 4].
    tgt_boxes: [bs, max_objects, 4].
    tgt_labels: [bs, max_objects].
    indices: [bs, 2, min(num_queries, max_objects)].
    bbox_loss_coef: L1 loss coefficient.
    giou_loss_coef: Generalized IoU (GIoU) loss coefficient.
    target_is_onehot: If `tgt_labels` is [bs, max_objects, num_classes].
  
  Returns:
    Dict of `loss_bbox` and `loss_giou`, unnormalized, with coefficients
    applied.
  """
  src_indices = indices[..., 0, :]
  tgt_indices = indices[..., 1, :]

  src_boxes = model_utils.simple_gather(src_boxes, src_indices)
  tgt_boxes = model_utils.simple_gather(tgt_boxes, tgt_indices)

  # Some of the boxes are padding. We want to discount them from the loss.
  if target_is_onehot:
    tgt_not_padding = 1 - tgt_labels[..., 0]
  else:
    tgt_not_padding = tgt_labels != 0

  # Align this with the permuted target indices.
  tgt_not_padding = model_utils.simple_gather(tgt_not_padding, tgt_indices)
  tgt_not_padding = jnp.asarray(tgt_not_padding, dtype=jnp.float32)

  src_boxes_xyxy = box_utils.box_cxcywh_to_xyxy(src_boxes)
  tgt_boxes_xyxy = box_utils.box_cxcywh_to_xyxy(tgt_boxes)

  # GIoU loss.
  loss_giou = 1 - box_utils.generalized_box_iou(
      src_boxes_xyxy, tgt_boxes_xyxy, all_pairs=False)
  loss_giou *= tgt_not_padding
  loss_giou = giou_loss_coef * loss_giou.sum()

  # L1 loss.
  loss_bbox = model_utils.weighted_box_l1_loss(src_boxes_xyxy, tgt_boxes_xyxy)
  loss_bbox *= tgt_not_padding[..., None]
  loss_bbox = bbox_loss_coef * loss_bbox.sum()

  losses = {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}
  return losses


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
        num_classes=self.dataset_meta_data['num_classes'],
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

  @staticmethod
  def logits_to_probs(logits: jnp.ndarray,
                      log_p: bool = False,
                      is_sigmoid: bool = False) -> jnp.ndarray:
    # TODO: No override for sigmoid logit normalization.
    if not is_sigmoid:
      return jax.nn.log_softmax(logits) if log_p else jax.nn.softmax(logits)
    else:
      return jax.nn.log_sigmoid(logits) if log_p else jax.nn.sigmoid(logits)

  def matcher(self, cost: jnp.ndarray, n_cols: jnp.ndarray) -> jnp.ndarray:
    """Implements a matching function.
    
    Matching function matches output detections against ground truth detections
    and returns indices.

    Args: 
      cost: Matching cost matrix.
      n_cols: Number of non-padded columns in each cost matrix.
    
    Returns:
      Matched indices in the form of a list of tuples (src, dest) where `src`
      and `dst` are indices of corresponding src and target detections.
    """
    matcher_name, matcher_fn = self.config.get('matcher'), None
    if matcher_name == 'lazy':
      matcher_fn = matchers.lazy_matcher
    elif matcher_name == 'sinkhorn':
      matcher_fn = functools.partial(
          matchers.sinkhorn_matcher,
          epsilon=self.config.get('sinkhorn_epsilon', 0.001),
          init=self.config.get('sinkhorn_init', 50),
          decay=self.config.get('sinkhorn_decay', 0.9),
          num_iters=self.config.get('sinkhorn_num_iters', 1000),
          threshold=self.config.get('sinkhorn_threshold', 1e-2),
          chg_momentum_from=self.config.get('sinkhorn_chg_momentum_from', 100),
          num_permutations=self.config.get('sinkhorn_num_permutations', 100))
    elif matcher_name == 'greedy':
      matcher_fn = matchers.greedy_matcher
    elif matcher_name == 'hungarian':
      matcher_fn = functools.partial(matchers.hungarian_matcher, n_cols=n_cols)
    elif matcher_name == 'hungarian_tpu':
      matcher_fn = matchers.hungarian_tpu_matcher
    elif matcher_name == 'hungarian_scan_tpu':
      matcher_fn = matchers.hungarian_scan_tpu_matcher
    elif matcher_name == 'hungarian_cover_tpu':
      matcher_fn = matchers.hungarian_cover_tpu_matcher
    else:
      raise ValueError('Unknown matcher (%s).' % matcher_name)
    return jax.lax.stop_gradient(matcher_fn(cost))

  def compute_loss_for_layer(
      self,
      tgt_labels: jnp.ndarray,
      pred_logits: jnp.ndarray,
      tgt_boxes: jnp.ndarray,
      pred_boxes: jnp.ndarray,
      indices: Optional[jnp.ndarray] = None,
      batch_mask: Optional[jnp.ndarray] = None) -> ArrayDict:
    """Loss and metrics function for single prediction layer."""
    target_is_onehot = self.dataset_meta_data.get('target_is_onehot', False)
    if not indices:
      pred_prob = self.logits_to_probs(pred_logits)
      cost, n_cols = compute_cost(
          tgt_labels=tgt_labels,
          out_prob=pred_prob,
          tgt_bbox=tgt_boxes,
          out_bbox=pred_boxes,
          class_loss_coef=self.config.class_loss_coef,
          bbox_loss_coef=self.config.bbox_loss_coef,
          giou_loss_coef=self.config.giou_loss_coef,
          target_is_onehot=target_is_onehot)
      indices = self.matcher(cost, n_cols=n_cols)

      n = pred_logits.shape[-2]

      def pad_matches(match):
        b, m = match.shape[0], match.shape[-1]
        if n > m:

          def get_unmatched_indices(row, ind):
            return jax.lax.top_k(
                jnp.logical_not(row.at[ind].set(True)), k=n - m)

          get_unmatched_indices = jax.vmap(get_unmatched_indices)
          indices = jnp.zeros((b, n), dtype=jnp.bool_)
          _, indices = get_unmatched_indices(indices, match[:, 0, :])
          indices = jnp.expand_dims(indices, axis=1)  # [b, 1, n-m]

          # Unmatched indices point to the last object
          padding = jnp.concatenate([
              indices,
              jnp.full(indices.shape, fill_value=m - 1),
          ], axis=1)  # [b, 2, n-m]  yapf: disable
          return jnp.concatenate([match, padding], axis=-1)  # [b, 2, n]
        return match

      indices = pad_matches(indices)

    losses = {}
    # Class loss.
    losses.update(
        loss_labels(
            pred_logits=pred_logits,
            tgt_labels=tgt_labels,
            batch_mask=batch_mask,  # TODO: is this always same as tgt_not_padding?
            indices=indices,
            class_loss_coef=self.config.class_loss_coef,
            eos_coef=self.config.eos_coef,
            target_is_onehot=target_is_onehot,
        ))

    # Boxes loss.
    losses.update(
        loss_boxes(
            src_boxes=pred_boxes,
            tgt_boxes=tgt_boxes,
            tgt_labels=tgt_labels,
            indices=indices,
            bbox_loss_coef=self.config.bbox_loss_coef,
            giou_loss_coef=self.config.giou_loss_coef,
            target_is_onehot=target_is_onehot))
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
        indices=indices,
        batch_mask=batch.get('batch_mask'))

    if 'aux_outputs' in outputs:
      for i, aux_outputs in enumerate(outputs['aux_outputs']):
        aux_losses = self.compute_loss_for_layer(
            tgt_labels=tgt_labels,
            pred_logits=aux_outputs['pred_logits'],
            pred_boxes=aux_outputs['pred_boxes'],
            tgt_boxes=tgt_boxes,
            indices=aux_indices[i] if aux_indices is not None else None,
            batch_mask=batch.get('batch_mask'))
        aux_losses = {f'{k}_aux_{i}': v for k, v in aux_losses.items()}
        losses.update(aux_losses)

    # Compute normalization.
    num_targets = jnp.sum(tgt_labels > 0, axis=1)
    norm_type = self.config.get('normalization', 'detr')
    logging.info('Normalization type: %s', norm_type)
    if norm_type == 'detr':
      num_targets = jnp.maximum(num_targets.sum(), 1.)
    elif norm_type == 'global':
      num_targets = jax.lax.pmean(num_targets.sum(), axis_name='batch')
    else:
      raise ValueError(f'Unknown normalization {norm_type}.')

    # Normalize losses by num_boxes.
    losses = jax.tree.map(lambda x: x / num_targets, losses)
    losses['total_loss'] = jax.tree.reduce(jnp.add, losses, 0)

    # Store metrics for logging.
    metrics = {k: (v, 1.) for k, v in losses.items()}
    for k, v in metrics.items():
      metrics[k] = model_utils.psum_metric_normalizer(v)

    return losses['total_loss'], metrics
