"""Training config for DETR on MS-Coco dataset, using Sinkhorn matcher."""
import ml_collections

COCO_TRAIN_SIZE = 118_287


def get_config():
  config = ml_collections.ConfigDict()

  config.rng_seed = 42

  # Dataset config
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.name = 'coco/2017'
  config.dataset_configs.max_size = 1333
  # Should be `config.num_queries - 1` because (i) Sinkhorn currently requires
  # square cost matrices; and (ii) an additional empty box is appended inside
  # the model.
  config.dataset_configs.max_boxes = 99
  config.dataset_configs.input_range = (-1., 1.)

  # Model config
  config.model_dtype_str = 'float32'
  config.matcher = 'sinkhorn'
  config.hidden_dim = 256
  config.num_queries = 100
  config.query_emb_size = None  # Same as hidden size.
  config.transformer_num_heads = 8
  config.transformer_num_encoder_layers = 6
  config.transformer_num_decoder_layers = 6
  config.transformer_qkv_dim = 256
  config.transformer_mlp_dim = 2048
  config.transformer_normalize_before = False
  config.backbone_module = 'bit'  # Choose from ['bit', 'resnet']
  config.backbone_width = 1
  config.backbone_depth = 50
  config.dropout_rate = 0.
  config.attention_dropout_rate = 0.1

  # Sinkhorn config
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

  # Loss
  config.aux_loss = True
  config.class_loss_coef = 1.0
  config.bbox_loss_coef = 5.0
  config.giou_loss_coef = 2.0
  config.eos_coef = 0.1

  # Training
  config.batch_size = 64
  config.total_epochs = 300

  # Optimizer (AdamW)
  steps_per_epoch = COCO_TRAIN_SIZE // config.batch_size
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.grad_clip_norm = 0.1
  config.optimizer_configs.base_lr = 1e-4
  config.optimizer_configs.backbone_lr_reduction = 0.1
  config.optimizer_configs.schedule = dict(
    decay_type='stair', steps=[200 * steps_per_epoch], mults=[0.1])
  config.optimizer_configs.optax_kw = dict(b1=0.9, b2=0.999, weight_decay=1e-4)

  # Pretrained checkpoints
  config.load_pretrained_backbone = True
  config.freeze_backbone_batch_stats = True
  config.pretrained_backbone_configs = ml_collections.ConfigDict()
  config.pretrained_backbone_configs.checkpoint_path = 'artifacts/bit_r50x1_i1k_checkpoint'

  # Annotations
  config.annotations_loc = './instances_val2017.json'

  # Logging/checkpointing
  config.checkpoint = True
  config.xprof = False
  config.log_summary_steps = 400
  config.log_eval_steps = steps_per_epoch
  return config
