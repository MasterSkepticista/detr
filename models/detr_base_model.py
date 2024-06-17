# ----------------------------------------------------------------
# Modified from Scenic DETR (https://github.com/google-research/scenic/scenic/baselines/detr)
# Copyright 2024 The Scenic Authors.
# ----------------------------------------------------------------
"""Base class for DETR object detection with matching."""
import abc
import functools
from typing import Any, Callable, Dict, Optional, Tuple

from flax.linen.module import Module as Module
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from model_lib import matchers
from model_lib.base_models import base_model, box_utils, model_utils

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

  batch_size, max_num_boxes = tgt_labels.shape[:2]
  num_queries = out_prob.shape[1]
  if target_is_onehot:
    mask = tgt_labels[..., 0] == 0  # All instances not padding.
  else:
    mask = tgt_labels != 0

  # [B, N, M]
  cost_class = -out_prob  # Actually, (1-out_prob), but constants are discarded.
  max_cost_class = 0.0
  if target_is_onehot:
    cost_class = jnp.einsum('bnl,bml->bnm', cost_class, tgt_labels)
  else:
    cost_class = jax.vmap(jnp.take, (0, 0, None))(cost_class, tgt_labels, 1)

  cost = class_loss_coef * cost_class
  cost_upper_bound = class_loss_coef * max_cost_class

  if out_bbox is not None:
    # Compute L1 cost between boxes
    diff = jnp.abs(out_bbox[:, :, None] - tgt_bbox[:, None, :])
    cost_bbox = jnp.sum(diff, axis=-1)  # [B, N, M]
    cost = cost + bbox_loss_coef * cost_bbox

    # Update upper bounds
    cost_upper_bound = cost_upper_bound + bbox_loss_coef * 4.0

    # [B, N, M]
    cost_giou = -box_utils.generalized_box_iou(
        box_utils.box_cxcywh_to_xyxy(out_bbox),
        box_utils.box_cxcywh_to_xyxy(tgt_bbox),
        all_pairs=True)
    cost = cost + giou_loss_coef * cost_giou

    # cost_giou < 0, but can be a bit higher in the beginning of training
    cost_upper_bound = cost_upper_bound + giou_loss_coef * 1.0

  mask = mask[:, :, None]
  cost = cost * mask + (1.0 - mask) * cost_upper_bound

  # Guard against NaNs and Infs
  cost = jnp.nan_to_num(
      cost,
      nan=cost_upper_bound,
      posinf=cost_upper_bound,
      neginf=cost_upper_bound)
  assert cost.shape == (batch_size, num_queries, max_num_boxes)

  # Compute the number of unpadded columns for each batch element.
  # It is assumed that all padding is trailing padding.
  n_cols = jnp.where(
      jnp.max(mask, axis=1),  # apply mask per-sample
      jnp.expand_dims(
          jnp.arange(1, max_num_boxes + 1),
          axis=0),  # id-tensor to be broadcasted to all elements in the batch
      0,  # trailing padding boxes to be replaced by id 0
  )
  n_cols = jnp.max(n_cols, axis=1)  # number of valid boxes per sample in batch
  return cost, n_cols


class BaseModelWithMatching(base_model.BaseModel):
  """Base model for object detection or instance segmentation with matching.
  
  This model implements the classification and matching parts which are shared
  between the object detection and instance segmentation models.
  """

  def __init__(self, config: Optional[ml_collections.ConfigDict],
               dataset_meta_data: Dict[str, Any]):
    """Initialize Detection model.
    
    Args:
      config: Configurations of the model.
      dataset_meta_data: Dataset meta data specifies `target_is_onehot`, which
        is False by default. The padded objects have label 0. The first 
        legitimate object has label 1, and so on.
    """
    self.losses_and_metrics = ['labels']
    if config is not None:
      self.loss_terms_weights = {'loss_class': config.class_loss_coef}
    super().__init__(config, dataset_meta_data)

  @property
  @abc.abstractmethod
  def loss_and_metrics_map(
      self) -> Dict[str, Callable[..., Tuple[ArrayDict, MetricsDict]]]:
    """Returns a dict that lists all losses for this model."""
    return {'labels': self.labels_losses_and_metrics}

  def compute_cost_matrix(self, predictions: ArrayDict,
                          targets: ArrayDict) -> jnp.ndarray:
    """Implements the matching cost matrix computations.
    
    Args:
      predictions: Dict of outputs from a model. Must contain `pred_boxes`
        and `pred_probs` keys with shapes [B, N, 4] and [B, N, L] respectively.
      targets: Dict of GT targets. Must contain `boxes` and `labels` keys of
        shapes [B, M, 4] and [B, M, L] respectively.
    
    Returns:
      The matching cost matrix of shape [B, N, M]
    """
    raise NotImplementedError(
        'Subclasses must implement `compute_cost_matrix`.')

  def matcher(self,
              cost: jnp.ndarray,
              n_cols: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Implements a matching function.
    
    Matching function matches output detections against ground truth detections
    and returns indices.

    Args:
      cost: Matching cost matrix.
      n_cols: Number of non-padded columns in each cost matrix.
    
    Returns:
      Matched indices in the form of a list of tuples (src, dst) where `src`
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

  def logits_to_probs(self,
                      logits: jnp.ndarray,
                      log_p: bool = False) -> jnp.ndarray:
    is_sigmoid = self.config.get('sigmoid_loss', False)
    if not is_sigmoid:
      return jax.nn.log_softmax(logits) if log_p else jax.nn.softmax(logits)
    else:
      return jax.nn.log_sigmoid(logits) if log_p else jax.nn.sigmoid(logits)

  def labels_losses_and_metrics(
      self,
      outputs: ArrayDict,
      batch: ArrayDict,
      indices: jnp.ndarray,
      log: bool = True) -> Tuple[ArrayDict, MetricsDict]:
    """Classification softmax cross entropy loss and (optionally) top-1 correct.
    
    Args:
      outputs: Model predictions. For the purpose of this loss, outputs must 
        have key `pred_logits`. `outputs['pred_logits']` is a nd-array of the 
        predicted pre-softmax logits of shape [B, N, C].
      batch: Dict that has `inputs`, `batch_mask` and `label` (ground truth).
        `batch['label']` is a dict. For the purpose of this loss, label dict
        must have key `labels` where the value is an int nd-array of labels. It
        may be one-hot if `dataset_meta_data.target_is_onehot` was set to True.
        If `batch['batch_mask']`  is provided, it is used to weigh the loss for
        different images in the current batch of examples.
      indices: Matcher output of shape [B, 2, min(N, M)] which conveys source
        to target pairing of objects.
      log: If true, return class_error as well.
    
    Returns:
      loss: Dict with keys `loss_class`.
      metrics: Dict with keys `loss_class` and `class_error`.
    """
    assert 'pred_logits' in outputs
    assert 'label' in batch

    batch_weights = batch.get('batch_mask')
    losses, metrics = {}, {}
    targets = batch['label']
    if isinstance(targets, dict):
      targets = targets['labels']

    src_logits = outputs['pred_logits']
    num_classes = src_logits.shape[-1]

    # Apply the permutation communicated by indices.
    src_logits = model_utils.simple_gather(src_logits, indices[:, 0])
    tgt_labels = model_utils.simple_gather(targets, indices[:, 1])
    if self.dataset_meta_data.get('target_is_onehot', False):
      tgt_labels_onehot = tgt_labels
    else:
      tgt_labels_onehot = jax.nn.one_hot(tgt_labels, num_classes)

    # Normalization before masking is important so that masked classes can be
    # assigned when using softmax normalization.
    src_log_p = self.logits_to_probs(src_logits, log_p=True)
    unnormalized_loss_class, denom = self._compute_per_example_class_loss(
        tgt_labels_onehot=tgt_labels_onehot,
        src_log_p=src_log_p,
        batch_weights=batch_weights,
        neg_category_ids=None,
        loose_box=None,
    )

    metrics['loss_class'] = (unnormalized_loss_class.sum(), denom.sum())

    norm_type = self.config.get('normalization', 'detr')
    if norm_type == 'detr':
      denom = jnp.maximum(denom.sum(), 1.)
    elif norm_type == 'global':
      denom = jax.lax.pmean(denom.sum(), axis_name='batch')
      denom = jnp.maximum(denom, 1.)
    else:
      raise ValueError(f'Unknown loss normalization `{norm_type}`.')
    normalized_loss_class = unnormalized_loss_class.sum() / denom

    losses['loss_class'] = normalized_loss_class

    if log:
      # For normalization, we need to have  number of inputs that we do
      # prediction for , which is number of examples in the batch times
      # number of boxes (including padded boxes).
      # Note that tgt_labels_onehot.shape = (bs, num_boxes, num_classes)
      if batch_weights is not None:
        batch_num_inputs = batch_weights.sum() * tgt_labels_onehot.shape[-2]
      else:
        batch_num_inputs = np.prod(tgt_labels_onehot.shape[:-1])

      # Class accuracy for non-padded (label != 0) one hot labels
      not_padded = tgt_labels_onehot[:, :, 0] == 0
      if batch_weights is not None:
        # Extends `not_padded` by also-masked elements.
        not_padded = not_padded * jnp.expand_dims(batch_weights, axis=1)
      num_correct_no_pad = model_utils.weighted_correctly_classified(
          src_log_p[..., 1:], tgt_labels_onehot[..., 1:], weights=not_padded)
      metrics['class_accuracy_not_pad'] = (num_correct_no_pad, not_padded.sum())

      if not self.config.get('sigmoid_loss', False):
        num_correct = model_utils.weighted_correctly_classified(
            src_log_p, tgt_labels_onehot, weights=batch_weights)
        metrics['class_accuracy'] = (num_correct, batch_num_inputs)

    # Sum metrics and normalizers over all replicas.
    for k, v in metrics.items():
      metrics[k] = model_utils.psum_metric_normalizer(v)
    return losses, metrics

  def _compute_per_example_class_loss(
      self,
      *,
      tgt_labels_onehot: jnp.ndarray,
      src_log_p: jnp.ndarray,
      batch_weights: Optional[jnp.ndarray],
      neg_category_ids: Optional[jnp.ndarray] = None,
      loose_box: Optional[jnp.ndarray] = None,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the unnormalized per-example classification loss and denom."""
    num_classes = src_log_p.shape[-1]

    # We want to weigh the padded objects differently. So class 0 gets eos_coef
    # and others get 1.0
    label_weights = jnp.concatenate([
        jnp.array([self.config.get('eos_coef', 1.0)], dtype=jnp.float32),
        jnp.ones(num_classes - 1)
    ])
    unnormalized_loss_class = model_utils.weighted_unnormalized_softmax_cross_entropy(
        src_log_p,
        tgt_labels_onehot,
        weights=batch_weights,
        label_weights=label_weights,
        logits_normalized=True)

    if label_weights is not None:
      denom = (tgt_labels_onehot * label_weights).sum(axis=[1, 2])
    else:  # Normalize by number of boxes after removing padding label.
      denom = tgt_labels_onehot[..., 1:].sum(axis=[1, 2])

    if batch_weights is not None:
      denom *= batch_weights

    return unnormalized_loss_class, denom

  def get_losses_and_metrics(self, loss: str, outputs: ArrayDict,
                             batch: ArrayDict, indices: jnp.ndarray,
                             **kwargs) -> Tuple[ArrayDict, MetricsDict]:
    """A convenience wrapper to all the loss_* functions in this class."""
    assert loss in self.loss_and_metrics_map, f'Unknown loss {loss}.'
    return self.loss_and_metrics_map[loss](outputs, batch, indices, **kwargs)

  def loss_function(
      self,
      outputs: ArrayDict,
      batch: ArrayDict,
      matches: Optional[jnp.ndarray] = None,
      model_params: Optional[jnp.ndarray] = None,
  ) -> Tuple[jnp.ndarray, MetricsDict]:
    """Loss and metrics function for ObjectDetectionWithMatchingModel.
    
    Args:
      output: Model prediction. The exact fields depend on the losses used.
      batch: Dict that has `inputs`, `batch_mask` and `label` (gt).
        batch['label'] is a dict where keys and values depend on the losses used
      matches: Output of hungarian matcher if available.
      model_params: PyTree (optional); parameters of the model.
    
    Returns:
      total_loss: Total loss weighted appropriately using self.loss_terms_weights
        and combined across all auxiliary outputs.
      metrics_dict: Individual loss terms with and without weighting for 
        logging purposes.
    """
    # Append a padding instance to the inputs. Those are not guaranteed by the
    # input pipeline to always be present.
    batch = batch.copy()  # These are shallow copies
    batch['label'] = batch['label'].copy()

    # Append a class label.
    if self.dataset_meta_data['target_is_onehot']:
      # Shape is [batch, num_instances, num_classes]
      label_shape = batch['label']['labels'].shape
      num_classes = label_shape[-1]
      instance = jax.nn.one_hot(0, num_classes)
      reshape_shape = (1,) * (len(label_shape) - 1) + (num_classes,)
      broadcast_shape = label_shape[:-2] + (1, num_classes)
      instance = jnp.broadcast_to(
          jnp.reshape(instance, reshape_shape), broadcast_shape)
    else:
      instance = jnp.zeros_like(batch['label']['labels'][..., :1])
    batch['label']['labels'] = jnp.concatenate(
        [batch['label']['labels'], instance], axis=-1)

    # Same for boxes.
    instance = jnp.zeros_like(batch['label']['boxes'][..., :1, :])
    batch['label']['boxes'] = jnp.concatenate(
        [batch['label']['boxes'], instance], axis=-2)

    if matches is None:
      if 'cost' not in outputs:
        cost, n_cols = self.compute_cost_matrix(outputs, batch['label'])
      else:
        cost, n_cols = outputs['cost'], outputs.get('cost_n_cols')
      matches = self.matcher(cost, n_cols)
      if 'aux_outputs' in outputs:
        matches = [matches]
        for aux_pred in outputs['aux_outputs']:
          if 'cost' not in outputs:
            cost, n_cols = self.compute_cost_matrix(aux_pred, batch['label'])
          else:
            cost, n_cols = aux_pred['cost'], outputs.get('cost_n_cols')
          matches.append(self.matcher(cost, n_cols))

    if not isinstance(matches, (list, tuple)):
      # Ensure matches come as a sequence.
      matches = [matches]

    # If the matching is not complete (i.e. the number of queries is larger than
    # the number of labels) we will pad the matches.
    n = outputs['pred_logits'].shape[-2]

    def pad_matches(match):
      b, m = match.shape[0], match.shape[-1]
      if n > m:

        def get_unmatched_indices(row, ind):
          return jax.lax.top_k(jnp.logical_not(row.at[ind].set(True)), k=n - m)

        get_unmatched_indices = jax.vmap(get_unmatched_indices)

        indices = jnp.zeros((b, n), dtype=jnp.bool_)
        _, indices = get_unmatched_indices(indices, match[:, 0, :])
        indices = jnp.expand_dims(indices, axis=1)  # [b, 1, n-m]

        padding = jnp.concatenate(
            [
                indices,
                jnp.full(indices.shape, fill_value=m - 1)
                # unmatched indices point to last object
            ],
            axis=1)  # [b, 2, n-m]
        return jnp.concatenate([match, padding], axis=-1)  # [b, 2, n]
      return match

    matches = [pad_matches(match) for match in matches]

    indices, *aux_indices = matches  # Matches from the terminal decoder.

    # Compute all requested losses.
    loss_dict = {}
    metrics_dict = {}
    for loss in self.losses_and_metrics:
      loss, metrics = self.get_losses_and_metrics(loss, outputs, batch, indices)
      loss_dict.update(loss)
      metrics_dict.update(metrics)

    # The outputs might have auxiliary predictions from more layers. We process
    # them below.
    if aux_indices is not None:
      for i, aux_outputs in enumerate(outputs['aux_outputs']):
        # Computes all the losses for this auxiliary output except class_error
        for loss in self.losses_and_metrics:
          # Disable class error for loss on labels.
          kwargs = {'log': False} if loss == 'labels' else {}
          l_dict, m_dict = self.get_losses_and_metrics(loss, aux_outputs, batch,
                                                       aux_indices[i], **kwargs)
          l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
          loss_dict.update(l_dict)
          # Add metrics for aux outputs.
          metrics_dict.update({k + f'_aux_{i}': v for k, v in m_dict.items()})

    # Calculate total loss by combining loss_dict with loss_terms_weights
    total_loss = []
    for k, v in loss_dict.items():
      k = k.split('_aux_')[0]  # Remove aux suffix for term weights
      if k in self.loss_terms_weights:
        total_loss.append(self.loss_terms_weights[k] * v)
    total_loss = sum(total_loss)

    if self.config.get('l2_decay_factor') is not None:
      raise NotImplementedError('Loss-coupled weight decay is not supported. '
                                'Use decoupled weight decay instead.')

    # Process metrics dictionary to generate final unnormalized metrics.
    metrics = self.get_metrics(metrics_dict)
    metrics['total_loss'] = (total_loss, 1)
    return total_loss, metrics

  def get_metrics(self, metrics_dict: MetricsDict) -> MetricsDict:
    """Arrange loss dictionary into a metrics dictionary."""
    metrics = {}
    # Some metrics don't get scaled, so no need to keep their unscaled version,
    # i.e. those that are not in self.loss_terms_weights.keys()
    for k, v in metrics_dict.items():
      coef = self.loss_terms_weights.get(k.split('_aux_')[0])
      if coef is not None:
        metrics[f'{k}_unscaled'] = v
        metrics[k] = (coef * v[0], v[1])
      else:
        metrics[k] = v

    return metrics

  def build_flax_model(self) -> Module:
    raise NotImplementedError('Subclasses must implement `build_flax_model()`.')

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    """Default config for the flax model that is built in `build_flax_model`.
    
    This function in particular serves the testing functions and is supposed to
    provide config that are passed to the flax_model.
    """
    raise NotImplementedError(
        'Subclasses must implement `default_flax_model_config()`.')


class ObjectDetectionWithMatchingModel(BaseModelWithMatching):
  """Base model for object detection with matching."""

  def __init__(self, config: Optional[ml_collections.ConfigDict],
               dataset_meta_data: Dict[str, Any]):
    """Initialize Detection model.

    Args:
      config: Hyper-parameter dictionary.
      dataset_meta_data: Dataset meta data specifies `target_is_onehot`, which
        is False by default, and a required `num_classes`, which is the number
        of object classes including bg/unlabeled/padding. The padded objects
        have label 0. The first legitimate object has label 1, and so on.
    """
    super().__init__(config, dataset_meta_data)
    self.losses_and_metrics.append('boxes')
    if config is not None:
      self.loss_terms_weights['loss_bbox'] = config.bbox_loss_coef
      self.loss_terms_weights['loss_giou'] = config.giou_loss_coef

  @property
  def loss_and_metrics_map(
      self) -> Dict[str, Callable[..., Tuple[ArrayDict, MetricsDict]]]:
    """Returns a dict that lists all losses for this model."""
    return {
        **super().loss_and_metrics_map,
        'boxes': self.boxes_losses_and_metrics,
    }

  def compute_cost_matrix(self, predictions: Dict[str, jax.Array],
                          targets: Dict[str, jax.Array]) -> jax.Array:
    """Implements the matching cost matrix computations.
    
    Args:
      predictions: Dict of outputs from a model. Must contain `pred_boxes`
        and `pred_probs` keys with shapes [B, N, 4] and [B, N, L] respectively.
      targets: Dict of GT targets. Must contain `boxes` and `labels` keys of
        shapes [B, M, 4] and [B, M, L] respectively.
    
    Returns:
      The matching cost matrix of shape [B, N, M]
    """
    return compute_cost(
        tgt_labels=targets['labels'],
        out_prob=self.logits_to_probs(predictions['pred_logits']),
        tgt_bbox=targets['boxes'],
        out_bbox=predictions['pred_boxes'],
        class_loss_coef=self.config.class_loss_coef,
        bbox_loss_coef=self.config.bbox_loss_coef,
        giou_loss_coef=self.config.giou_loss_coef,
        target_is_onehot=self.dataset_meta_data['target_is_onehot'])

  def boxes_losses_and_metrics(
      self, outputs: ArrayDict, batch: ArrayDict,
      indices: jnp.ndarray) -> Tuple[ArrayDict, MetricsDict]:
    """Bounding box losses: L1 regression loss and GIoU loss.
    
    Args:
      outputs: dict; Model predictions. For the purpose of this loss, outputs
        must have a key `pred_boxes`. outputs['pred_boxes'] an ndarray of 
        predicted box coordinates in (cx, cy, w, h) format. This ndarray has
        shape [bs, num_boxes, 4].
      batch: dict; that has `inputs`, `batch_mask` and `label` (ground truth).
        batch['label'] is a dict. For the purpose of this loss, batch['label']
        must have key 'boxes', which has a value in the same format as 
        output['pred_boxes']. Additionally in batch['label'], key 'labels' is
        required that should match the specs defined in the member function
        `labels_losses_and_metrics`. This is to decide which boxes are invalid 
        and need to be ignored. Invalid boxes have class label 0. If 
        batch['batch_mask'] is provided, it is used to weigh the loss for 
        different images in the current batch of examples.
      indices: list[tuple[ndarray, ndarray]]; Matcher output which conveys 
        source to target pairing of objects.

    Returns:
      loss: dict with keys `loss_bbox`. These are losses averaged over the batch.
        Therefore they have shape [].
      metrics: dict with keys `loss_bbox` and `loss_giou`. These are metrics
        psumed over the batch. Therefore they have shape [].
    """
    assert 'pred_boxes' in outputs
    assert 'label' in batch

    targets = batch['label']
    assert 'boxes' in targets
    assert 'labels' in targets
    losses, metrics = {}, {}
    batch_weights = batch.get('batch_mask')

    src_boxes = model_utils.simple_gather(outputs['pred_boxes'], indices[:, 0])
    tgt_boxes = model_utils.simple_gather(targets['boxes'], indices[:, 1])

    # Some of the boxes are padding. We want to discount them from the loss.
    target_is_onehot = self.dataset_meta_data.get('target_is_onehot', False)
    if target_is_onehot:
      tgt_not_padding = targets['labels'][..., 0] == 0
    else:
      tgt_not_padding = targets['labels'] != 0

    # `tgt_not_padding` has shape [bs, num_boxes].
    # Align this with the model predictions using simple gather.
    tgt_not_padding = model_utils.simple_gather(tgt_not_padding, indices[:, 1])
    num_targets = tgt_not_padding.sum(axis=1)

    # GIoU Loss.
    unnormalized_loss_giou = 1 - box_utils.generalized_box_iou(
        box_utils.box_cxcywh_to_xyxy(src_boxes),
        box_utils.box_cxcywh_to_xyxy(tgt_boxes),
        all_pairs=False)
    if batch_weights is not None:
      num_targets *= batch_weights
      unnormalized_loss_giou = model_utils.apply_weights(
          unnormalized_loss_giou, batch_weights)

    # This implementation assumes tight bboxes only. L1 loss is computed on
    # boxes in cxcywh format, as in the original DETR.
    unnormalized_loss_bbox = model_utils.weighted_box_l1_loss(
        src_boxes, tgt_boxes, weights=batch_weights).sum(axis=2)

    # Ignore padded boxes.
    unnormalized_loss_bbox *= tgt_not_padding
    unnormalized_loss_giou *= tgt_not_padding

    # Normalize by the number of boxes in batch.
    num_targets = jnp.maximum(
        jax.lax.pmean(num_targets.sum(), axis_name='batch'), 1)
    normalized_loss_bbox = unnormalized_loss_bbox.sum() / num_targets
    normalized_loss_giou = unnormalized_loss_giou.sum() / num_targets

    losses['loss_bbox'] = normalized_loss_bbox
    metrics['loss_bbox'] = (normalized_loss_bbox, 1.)
    losses['loss_giou'] = normalized_loss_giou
    metrics['loss_giou'] = (normalized_loss_giou, 1.)

    # Sum metrics and normalizers over all replicas.
    for k, v in metrics.items():
      metrics[k] = model_utils.psum_metric_normalizer(v)
    return losses, metrics
