"""Common config for training DETR on MS-COCO dataset."""
import ml_collections

COCO_TRAIN_SIZE = 118_287


def get_coco_config():
  config = ml_collections.ConfigDict()

  config.rng_seed = 42

  # Dataset config
  config.dataset_name = 'coco_detr'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.max_size = 1333
  config.dataset_configs.max_boxes = 99
  config.dataset_configs.shuffle_buffer_size = 10_000
  config.dataset_configs.input_range = (-1., 1.)

  # Model config
  config.model_dtype_str = 'float32'
  config.hidden_dim = 256
  config.num_queries = 100
  config.query_emb_size = None  # Same as hidden size.
  config.transformer_num_heads = 8
  config.transformer_num_encoder_layers = 6
  config.transformer_num_decoder_layers = 6
  config.transformer_qkv_dim = 256
  config.transformer_mlp_dim = 2048
  config.transformer_normalize_before = False
  config.backbone_width = 1
  config.backbone_depth = 50
  config.dropout_rate = 0.1
  config.attention_dropout_rate = 0.1

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

  # Logging/checkpointing
  config.checkpoint = True
  config.xprof = False
  config.debug = False
  config.log_large_summary_steps = steps_per_epoch
  config.log_summary_steps = 400
  config.log_eval_steps = steps_per_epoch
  return config
