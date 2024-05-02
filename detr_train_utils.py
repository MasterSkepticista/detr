"""Train utils specific to DETR."""
import numpy as np

from train_lib import train_utils


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
