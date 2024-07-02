"""COCO evaluation metrics based on pycocotools.

Implementation is based on a cut-down version of
https://github.com/google/flax/blob/ac5e46ed448f4c6801c35d15eb15f4638167d8a1/examples/retinanet/coco_eval.py

"""
import contextlib
import os
from typing import Optional

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class DetectionEvaluator:
  """Main Evaluator class."""

  def __init__(self,
               annotations_loc: Optional[str] = None,
               threshold: float = 0.05,
               disable_output: bool = True):
    """Initializes a DetectionEvaluator object.
    
    Args:
      annotations_loc: A path towards the JSON file storing the COCO-style 
        ground truths for object detection.
      threshold: A scalar which indicates the lower threshold (inclusive) for 
        the scores. Anything below this value will be removed.
      disable_output: if True disables the output produced by the COCO API.
    """
    self.annotations = []
    self.annotated_img_ids = []
    self.threshold = threshold
    self.disable_output = disable_output

    if self.disable_output:
      with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
          self.coco = COCO(annotations_loc)
    else:
      self.coco = COCO(annotations_loc)

    # Dict to translate model labels to COCO category IDs:
    self.label_to_coco_id = {
        i: cat['id'] for i, cat in enumerate(self.coco.dataset['categories'])
    }

  @staticmethod
  def construct_result_dict(coco_metrics):
    """Packs the COCOEval results into a dictionary.
    
    Args:
      coco_metrics: An array of length 12, as returned by `COCOeval.summarize()`
    Returns:
      A dictionary which contains all the COCO metrics.
    """
    return {
        'AP': coco_metrics[0],
        'AP_50': coco_metrics[1],
        'AP_75': coco_metrics[2],
        'AP_small': coco_metrics[3],
        'AP_medium': coco_metrics[4],
        'AP_large': coco_metrics[5],
        'AR_max_1': coco_metrics[6],
        'AR_max_10': coco_metrics[7],
        'AR_max_100': coco_metrics[8],
        'AR_small': coco_metrics[9],
        'AR_medium': coco_metrics[10],
        'AR_large': coco_metrics[11]
    }

  def clear_annotations(self):
    """Clears the annotations collected in this object.
    
    It is important to call this method either at the end of at the beginning
    of a new evaluation round (or both). Otherwise, previous model inferences 
    will skew the results due to residual annotations.
    """
    self.annotations.clear()
    self.annotated_img_ids.clear()

  def extract_classifications(self, bboxes, scores):
    """Extracts the label for each bbox, and sorts them by score.
    
    More specifically, after extracting each bbox's label, the bboxes and
    scores are sorted in descending order based on score. The scores which fall
    below `threshold` are removed.

    Args:
      bboxes: a matrix of shape (|B|, 4) where |B| is the number of bboxes;
        each row contains `[x, y, w, h]` of the bbox.
      scores: a matrix of shape (|B|, K) where `K` is the number of classes in 
        the object detection task.
    
    Returns:
      A tuple consisting of the bboxes, a vector of length |B| containing
      the label of each of the anchors, and a vector of length |B| containing
      the label score. All elements are sorted in descending order relative to
      the score.
    """
    # Extract the labels and max score for each anchor.
    labels = np.argmax(scores, axis=-1)

    # Get the score associated to each anchor's label
    scores = scores[np.arange(labels.shape[0]), labels]

    # Apply the threshold.
    kept_idx = np.where(scores >= self.threshold)[0]
    scores = scores[kept_idx]
    labels = labels[kept_idx]
    bboxes = bboxes[kept_idx]

    # Sort everything in descending order and return.
    sorted_idx = np.flip(np.argsort(scores, axis=0))
    scores = scores[sorted_idx]
    labels = labels[sorted_idx]
    bboxes = bboxes[sorted_idx]

    return bboxes, labels, scores

  def add_annotation(self, bboxes, scores, img_id):
    """Add a single inference example as a COCO annotation for later evaluation.
    
    Labels should not include background/padding class, but only valid object 
    classes.

    Note that this method raises an exception if the `threshold` is too high 
    and thus eliminates all detections.

    Args:
      bboxes: [num_objects, 4] array of bboxes in COCO format [x, y, w, h] in
        absolute image coordinates.
      scores: [num_objects, num_classes] array of scores (softmax outputs).
      img_id: scalar COCO image ID.
    """
    # Get the sorted bboxes, labels and scores (threshold is applied here).
    i_bboxes, i_labels, i_scores = self.extract_classifications(bboxes, scores)

    if not i_bboxes.size:
      raise ValueError('All objects were thresholded out.')

    # Iterate through the thresholded predictions and pack them in COCO format.
    for bbox, label, score in zip(i_bboxes, i_labels, i_scores):
      single_classification = {
          'image_id': img_id,
          'category_id': self.label_to_coco_id[label],
          'bbox': bbox.tolist(),
          'score': score
      }
      self.annotations.append(single_classification)
      self.annotated_img_ids.append(img_id)

  def get_annotations_and_ids(self):
    """Returns copies of `self.annotations` and `self.annotated_img_ids`."""
    return self.annotations.copy(), self.annotated_img_ids.copy()

  def set_annotations_and_ids(self, annotations, ids):
    """Sets the `self.annotations` and `self.annotated_img_ids`.
    
    This method should only be used when trying to compute the metrics
    across hosts, where one host captures the data from everyone in an effor to
    to produce the entire dataset metrics.

    Args:
      annotations: the new `annotations`
      ids: the new `annotated_img_ids`
    """
    self.annotations = annotations
    self.annotated_img_ids = ids

  def compute_coco_metrics(self, clear_annotations=False):
    """Compute the COCO metrics for the collected annotations.
    
    Args:
      clear_annotations: If True, clears the `self.annotations` parameter
        after obtaining the COCO metrics.
      
    Returns:
      The COCO metrics as a dictionary defining the following entries:
      ```
      Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
      Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]
      Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]
      Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
      Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
      Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
      Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
      Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
      ```
    """

    def _run_eval():
      # Create prediction object for producing mAP metric values.
      pred_object = self.coco.loadRes(self.annotations)

      # Compute mAP
      coco_eval = COCOeval(self.coco, pred_object, 'bbox')
      coco_eval.params.imgIds = self.annotated_img_ids
      coco_eval.evaluate()
      coco_eval.accumulate()
      coco_eval.summarize()
      return coco_eval

    if self.disable_output:
      with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
          coco_eval = _run_eval()
    else:
      coco_eval = _run_eval()

    # Clear annotations if requested.
    if clear_annotations:
      self.clear_annotations()

    # Pack the results
    return self.construct_result_dict(coco_eval.stats)
