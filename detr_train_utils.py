# ----------------------------------------------------------------
# Modified from Scenic DETR (https://github.com/google-research/scenic/scenic/baselines/detr)
# Copyright 2024 The Scenic Authors.
# ----------------------------------------------------------------

"""Train utils specific to DETR."""
import copy
import json
import os
from typing import Any, Dict, Optional, Set

import PIL
import jax
import numpy as np
import scipy.special
import tensorflow as tf
from absl import logging

from dataset_lib.coco_dataset import coco_eval
from model_lib.base_models import box_utils
from train_lib import train_utils


class DetrGlobalEvaluator:
  """An interface between DETR implementation and COCO evaluators."""

  def __init__(self, dataset_name: str, **kwargs):
    del dataset_name  # unused.

    self.coco_evaluator = coco_eval.DetectionEvaluator(threshold=0.0, **kwargs)
    self._included_image_ids = set()
    self._num_examples_added = 0

  def add_example(self, prediction: Dict[str, np.ndarray],
                  target: Dict[str, np.ndarray]):
    """Add a single example to the evaluator.
    
    Args:
      prediction: Model prediction dictionary with keys `pred_img_ids`, 
        `pred_probs` in shape of `[num_objects, num_classes]` and `pred_boxes`
        in shape of `[num_objects, 4]`. Box coordinates should be in raw DETR 
        format, i.e. [cx, cy, w, h] in range [0, 1].
      target: Target dictionary with keys `orig_size`, `size` and `image/id`.
        Must also contain `padding_mask` if the input image was padded.
    """
    if 'pred_boxes' not in prediction:
      # Add dummy to make eval work.
      prediction = copy.deepcopy(prediction)
      prediction['pred_boxes'] = np.zeros(
          (prediction['pred_logits'].shape[0], 4)) + 0.5

    # Convert from DETR [cx, cy, w, h] to COCO [x, y, w, h] bounding box format.
    boxes = box_utils.box_cxcywh_to_xyxy(prediction['pred_boxes'])
    boxes = np.array(boxes)
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]

    # Scale from relative to absolute size:
    # Note that padding is implemented such that the model's predictions
    # are [0, 1] normalized to the non-padded image, so scaling by `orig_size`
    # will convert correctly to the original image coordinates.
    h, w = np.asarray(target['orig_size'])
    scale_factor = np.array([w, h, w, h])
    boxes = boxes * scale_factor[np.newaxis, :]
    boxes_np = np.asarray(boxes)

    # Get scores, excluding the background class.
    if 'pred_probs' in prediction:
      scores = prediction['pred_probs'][:, 1:]
    else:
      scores = scipy.special.softmax(prediction['pred_logits'], axis=-1)[:, 1:]

    # Add example to evaluator.
    self.coco_evaluator.add_annotation(
        bboxes=boxes_np,
        scores=np.asarray(scores),
        img_id=int(target['image/id']))
    self._num_examples_added += 1

  def compute_metrics(self,
                      included_image_ids: Optional[Set[int]] = None,
                      clear_annotations: bool = True) -> Dict[str, Any]:
    """Computes the metrics for all added predictions."""
    if included_image_ids is not None:
      self.coco_evaluator.coco.reload_ground_truth(included_image_ids)
    return self.coco_evaluator.compute_coco_metrics(
        clear_annotations=clear_annotations)

  def clear(self):
    self.coco_evaluator.clear_annotations()
    self._num_examples_added = 0

  def __len__(self):
    return self._num_examples_added

  def write_pred_annotations_to_file(self,
                                     path: str,
                                     fname_app: Optional[str] = None,
                                     clear_annotations: Optional[bool] = True):
    """Writes predictions to file in a JSON format.
    
    Args:
      path: Path to write the prediction annotation JSON file.
      fname_app: Optional string to append to the file name.
      clear_annotations: Clear annotations after they are stored in a file.
    """
    if not tf.io.gfile.exists(path):
      tf.io.gfile.makedirs(path)
    json_file_name = f"predictions{fname_app if fname_app else ''}.json"
    json_file_path = os.path.join(path, json_file_name)

    def _convert_to_serializable(obj):
      if isinstance(obj, np.ndarray):
        return obj.tolist()
      elif isinstance(obj, np.float32):
        return float(obj)
      else:
        raise TypeError(f'Unserializable object {obj} of type {type(obj)}')

    with tf.io.gfile.GFile(json_file_path, 'w') as f:
      f.write(
          json.dumps(
              self.coco_evaluator.annotations,
              default=_convert_to_serializable))
    logging.info('Predicted annotations are stored in %s.', json_file_path)
    if clear_annotations:
      self.coco_evaluator.clear_annotations()


def normalize_metrics_summary(metrics_summary, split,
                              object_detection_loss_keys):
  """Normalizes the metrics in the given metrics summary.
  
  Note that currently we only support metrics of the form 1/N sum f(x_i).

  Args:
    metrics_summary: dict; Each value is a sum of the calculated metric over all
      examples.
    split: str; Split for which we normalize the metrics. Used for logging.
    object_detection_loss_keys: list; A loss key used for computing the object
      detection loss.

  Returns:
    Normalized metrics summary.
  
  Raises:
    TrainingDivergedError: Due to observing a NaN in the metrics.
  """
  for key, val in metrics_summary.items():
    metrics_summary[key] = val[0] / val[1]
    if np.isnan(metrics_summary[key]):
      raise train_utils.TrainingDivergedError(
          'NaN detected in {}'.format(f'{split}_{key}'))

  # Compute and add object_detection_loss using globally normalized terms.
  object_detection_loss = 0
  for loss_term_key in object_detection_loss_keys:
    object_detection_loss += metrics_summary[loss_term_key]
  metrics_summary['object_detection_loss'] = object_detection_loss

  return metrics_summary


def draw_boxes_side_by_side(pred: Dict[str, Any], batch: Dict[str, Any],
                            label_map: Dict[int, str]) -> np.ndarray:
  """Side-by-side visualization of detection predictions and ground truth."""
  viz = []

  # Unnormalize images [-1, 1] -> [0, 255].
  imgs = tf.image.convert_image_dtype(batch['inputs'] / 2. + 0.5, tf.uint8)

  font = PIL.ImageFont.load_default()

  # Iterates over images in the batch and makes visualizations.
  for idx in range(imgs.shape[0]):
    h, w = batch['label']['size'][idx]
    gt_im = PIL.Image.fromarray(imgs[idx])
    gt_draw = PIL.ImageDraw.Draw(gt_im)
    for bb, cls, is_crowd in zip(batch['label']['boxes'][idx],
                                 batch['label']['labels'][idx],
                                 batch['label']['is_crowd'][idx]):
      if cls == 0:
        continue  # Dummy object

      bcx, bcy, bw, bh = bb * [w, h, w, h]  # unnormalize.
      bb = [bcx - bw / 2, bcy - bh / 2, bcx + bw / 2, bcy + bh / 2]
      edgecolor = (255, 0, 0) if is_crowd else (255, 255, 0)
      gt_draw.rectangle(bb, fill=None, outline=edgecolor, width=3)
      gt_draw.text([bb[0], max(bb[1] - 10, 0)],
                   label_map[cls],
                   font=font,
                   fill=(0, 0, 255))

    # For model predictions.
    pred_im = PIL.Image.fromarray(imgs[idx])
    pred_draw = PIL.ImageDraw.Draw(pred_im)
    pred_lbls = np.argmax(pred['pred_logits'], axis=-1)
    for bb, cls in zip(pred['pred_boxes'][idx], pred_lbls[idx]):
      h, w = batch['label']['size'][idx]
      bcx, bcy, bw, bh = bb * [w, h, w, h]  # unnormalize.
      bb = [bcx - bw / 2, bcy - bh / 2, bcx + bw / 2, bcy + bh / 2]
      edgecolor = (0, 255, 0) if cls else (0, 150, 0)
      pred_draw.rectangle(
          bb, fill=None, outline=edgecolor, width=3 if cls else 1)

    # Separate loop for text to prevent occlusion of text by boxes.
    for bb, cls in zip(pred['pred_boxes'][idx], pred_lbls[idx]):
      if not cls:
        continue
      h, w = batch['label']['size'][idx]
      bcx, bcy, bw, bh = bb * [w, h, w, h]
      bb = [bcx - bw / 2, bcy - bh / 2, bcx + bw / 2, bcy + bh / 2]
      pred_draw.text([bb[0], max(bb[1] - 10, 0)],
                     label_map[cls],
                     font=font,
                     fill=(0, 0, 255))
    gt_im_np = np.asarray(gt_im)
    pred_im_np = np.asarray(pred_im)
    composite = np.concatenate([pred_im_np, gt_im_np], axis=1)
    viz.append(composite)
  return np.stack(viz, axis=0)

def process_and_fetch_to_host(pred_or_tgt, batch_mask):
  """Used to collect predictions and targets of the whole valid/test set.
  
  Args:
    pred_or_tgt: PyTree; a pytree of jnp.array where leaves are of shape
      `[num_devices, bs, X, ..., Y]`.
    batch_mask: An nd-array of shape `[num_devices, bs]`, where zero values
      indicate padded examples.
  
  Returns:
    A list of length num_devices * bs of items where each item is a tree with
    the same structure as `pred_or_tgt` and each leaf contains a single example.
  """
  # Fetch to host in a single call.
  pred_or_tgt, batch_mask = jax.device_get((pred_or_tgt, batch_mask))
  batch_mask = np.array(batch_mask).astype(bool)

  def _split_mini_batches(x):
    # Filter out padded examples.
    x = x[batch_mask]
    # Split minibatch of examples into a list of examples.
    x_list = np.split(x, x.shape[0], axis=0)
    # Squeeze out the dummy dimension.
    return jax.tree_util.tree_map(lambda x: np.squeeze(x, axis=0), x_list)

  leaves, treedef = jax.tree_util.tree_flatten(pred_or_tgt)

  batch_shape = batch_mask.shape
  assert all([leaf.shape[:2] == batch_shape for leaf in leaves
             ]), ('Inconsistent batch shapes')

  # Split batched leaves into lists of examples.
  leaves = list(map(_split_mini_batches, leaves))

  # Go from leaf-lists to list of trees.
  out = []
  if leaves:
    num_examples = np.sum(batch_mask, dtype=np.int32)
    for example_ind in range(num_examples):
      out.append(treedef.unflatten([leaf[example_ind] for leaf in leaves]))
  return out
