"""Training config for DETR on MS-Coco dataset using exact Hungarian Matching.

This config replicates facebookresearch/detr ResNet50 (non-DC5) variant,
achieves ~40.5mAP with a 300ep schedule.
"""
from configs.common import get_coco_config


def get_config():
  config = get_coco_config()
  config.matcher = 'hungarian_cover_tpu'
  config.dataset_configs.max_boxes = config.num_queries - 1  # A padding instance is added before matching.

  return config