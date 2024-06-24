"""Training config for DETR on MS-Coco dataset using exact Hungarian Matching.

This config replicates facebookresearch/detr ResNet50 (non-DC5) variant,
achieves ~40.5mAP with a 300ep schedule.
"""
import ml_collections
from configs.common import get_coco_config


def get_config():
  config = get_coco_config()
  config.matcher = 'hungarian_cover_tpu'
  config.dataset_configs.max_boxes = 99  # A padding instance is added before matching.

  # Pretrained checkpoints
  config.load_pretrained_backbone = True
  config.freeze_backbone_batch_stats = True
  config.pretrained_backbone_configs = ml_collections.ConfigDict()
  config.pretrained_backbone_configs.checkpoint_path = 'artifacts/r50x1_i1k_checkpoint'

  # Annotations
  config.annotations_loc = './instances_val2017.json'
  return config