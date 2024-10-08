"""Training config for DETR using Sinkhorn matcher.

Sinkhorn matcher is GPU-friendly, and computes regularized (approximate) 
bipartite matching. Depending on whether the bottleneck is training or matching,
this config can be 50-100% faster overall than exact Hungarian matching while 
achieving comparable mAP scores.
"""
from configs.common import get_coco_config


def get_config():
  config = get_coco_config()

  config.matcher = 'sinkhorn'
  # Should be `config.num_queries - 1` because (i) Sinkhorn currently requires
  # square cost matrices; and (ii) an additional empty box is appended inside
  # the model.
  config.dataset_configs.max_boxes = config.num_queries - 1

  config.sinkhorn_epsilon = 1e-3
  # Speeds up convergence using epsilon decay. Start with a value 50 times
  # higher than the target and decay by a factor 0.9 between iterations.
  config.sinkhorn_init = 50
  config.sinkhorn_decay = 0.9
  config.sinkhorn_num_iters = 1000  # Sinkhorn number of iterations.
  config.sinkhorn_threshold = 1e-2  # Reconstruction threshold.
  # Starts using momentum after after 100 Sinkhorn iterations.
  config.sinkhorn_chg_momentum_from = 100
  config.sinkhorn_num_permutations = 100

  return config
