"""Utilities for boxes.

Implementation borrowed from google-research/scenic

Axis-aligned utils implemented based on:
https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
"""
from typing import Any, Tuple, Union

import jax.numpy as jnp
import numpy as np

Array = Union[jnp.ndarray, np.ndarray]
PyModule = Any


def box_cxcywh_to_xyxy(boxes: Array, np_backend: PyModule = jnp) -> Array:
  """Converts boxes from [cx, cy, w, h] format to [x, y, x', y'] format."""
  x_c, y_c, w, h = np_backend.split(boxes, 4, axis=-1)
  b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
  return np_backend.concatenate(b, axis=-1)


def box_iou(boxes1: Array,
            boxes2: Array,
            np_backend: PyModule = jnp,
            all_pairs: bool = True,
            eps: float = 1e-6) -> Tuple[Array, Array]:
  """Computes IoU between two sets of boxes.
  
  Boxes are in [x, y, x', y'] format where [x, y] is top-left, [x', y'] is 
  bottom-right.

  Args:
    boxes1: Predicted bounding boxes in shape [B, N, 4].
    boxes2: Target bounding boxes in shape [B, M, 4]. Can have a different 
      number of boxes if all_pairs is True.
    np_backend: numpy module. Either use regular numpy package or jax.numpy.
    all_pairs: Whether to compute IoU between all pairs of boxes or not.
    eps: Epsilon for numerical stability.
  
  Returns:
    If `all_pairs==True`, returns the pairwise IoU cost matrix of shape [B, N, M]
    If `all_pairs==False`, returns the IoU between corresponding boxes. The 
    shape of the return value is then [B, N].
    Also returns `union` array for other downstream calculations.
  """
  # First compute box areas. These will be used later for computing the union.
  wh1 = boxes1[..., 2:] - boxes1[..., :2]
  area1 = wh1[..., 0] * wh1[..., 1]  # [B, N]

  wh2 = boxes2[..., 2:] - boxes2[..., :2]
  area2 = wh2[..., 0] * wh2[..., 1]  # [B, M]

  if all_pairs:
    # Compute pairwise top-left and bottom-right corners of the intersection
    # of the boxes.
    lt = np_backend.maximum(boxes1[..., :, None, :2],
                            boxes2[..., None, :, :2])  # (B, N, M, 2)
    rb = np_backend.minimum(boxes1[..., :, None, 2:],
                            boxes2[..., None, :, 2:])  # (B, N, M, 2)

    # intersection = area of the box defined by (lt, rb)
    wh = (rb - lt).clip(0.0)  # [B, N, M, 2]
    intersection = wh[..., 0] * wh[..., 1]  # [B, N, M]

    # union = sum of areas - intersection
    union = area1[..., :, None] + area2[..., None, :] - intersection
    iou = intersection / (union + eps)

  else:
    # Compute top-left and bottom-right corners of the intersection between
    # corresponding boxes.
    assert boxes1.shape[1] == boxes2.shape[1], (
        'Different number of boxes provided with `all_pairs=False`.')
    lt = np_backend.maximum(boxes1[..., :2], boxes2[..., :2])  # (B, N, 2)
    rb = np_backend.minimum(boxes1[..., 2:], boxes2[..., 2:])  # (B, N, 2)

    # intersection = area of boxes bound by (lt, rb)
    wh = (rb - lt).clip(0.0)  # [B, N, 2]
    intersection = wh[..., 0] * wh[..., 1]  # [B, N]

    # union = sum of areas - intersection
    union = area1 + area2 - intersection
    iou = intersection / (union + eps)

  return iou, union


def generalized_box_iou(boxes1: Array,
                        boxes2: Array,
                        np_backend: PyModule = jnp,
                        all_pairs: bool = True,
                        eps: float = 1e-6) -> Array:
  """Generalized IoU from https://giou.stanford.edu/.
  
  The boxes should be in [x, y, x', y'] format specifying top-left and 
  bottom-right corners.

  Args:
    boxes1: Predicted bounding-boxes in shape [..., N, 4].
    boxes2: Target bounding-boxes in shape [..., M, 4].
    np_backend: Either to use numpy package or jax.numpy.
    all_pairs: Whether to compute generalized IoU scores between all-pairs of 
      boxes or not. Note that if all_pairs == False, we must have M == N.
    eps: Epsilon for numerical stability.
  
  Returns:
    If all_pairs == True, returns [B, N, M] pairwise matrix, of generalized ious
    If all_pairs == False, returns [B, N] matrix of generalized ious.
  """
  # This code assumes degenerate boxes are pre-checked. We cannot do asserts
  # on inputs with jitting.
  iou, union = box_iou(
      boxes1, boxes2, np_backend=np_backend, all_pairs=all_pairs, eps=eps)
  # Generalized IoU has an extra term which takes into account the area of the
  # box containing both of these boxes. The following code is very similar to
  # that for computing intersection, but the min/max are flipped.
  if all_pairs:
    lt = np_backend.minimum(boxes1[..., :, None, :2],
                            boxes2[..., None, :, :2])  # (B, N, M, 2)
    rb = np_backend.maximum(boxes1[..., :, None, 2:],
                            boxes2[..., None, :, 2:])  # (B, N, M, 2)
  else:
    lt = np_backend.minimum(boxes1[..., :2], boxes2[..., :2])  # (B, N, M, 2)
    rb = np_backend.maximum(boxes1[..., 2:], boxes2[..., 2:])  # (B, N, M, 2)

  # Now compute the covering box's area
  wh = (rb - lt).clip(0.0)  # Either [B, N, 2] or [B, N, M, 2]
  area = wh[..., 0] * wh[..., 1]  # Either [B, N] or [B, N, M]

  # Finally compute generalized IoU from IoU, union and area.
  return iou - (area - union) / (area + eps)
