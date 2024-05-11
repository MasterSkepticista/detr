"""Training config for DETR on MS-Coco dataset."""
import ml_collections

COCO_TRAIN_SIZE = 118_287


def get_config():
  config = ml_collections.ConfigDict()

  config.rng_seed = 0

  # Dataset config
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.name = 'coco/2017'
  config.dataset_configs.max_size = 640
  config.dataset_configs.max_boxes = 100
  config.dataset_configs.input_range = (-1., 1.)

  # Model config
  config.model_dtype_str = 'bfloat16'
  config.matcher = 'hungarian_cover_tpu'
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
  config.dropout_rate = 0.
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

  # Optimizer
  config.grad_clip_norm = 0.1
  config.optax_name = 'scale_by_adam'
  config.optax = dict(b1=0.9, b2=0.999, mu_dtype='bfloat16')
  config.lr = 1e-4
  config.wd = 1e-4  # Weight decay is decoupled and applied before LR scaling.
  config.schedule = [
      ('^(?!.*bn).*', dict(decay_type='cosine')),
      ('.*bn.*', None),
  ]
  config.lr_mults = [
      ('backbone.*', 0.1),  # Backbone lr
      ('(?!.*backbone).*', 1.0),  # Everything other than backbone
  ]

  # Pretrained checkpoints
  config.load_pretrained_backbone = True
  config.freeze_backbone_batch_stats = True
  config.pretrained_backbone_configs = ml_collections.ConfigDict()
  config.pretrained_backbone_configs.checkpoint_path = '/home/karan/workspace/orion-jax/artifacts/r50_i1k'

  # Annotations
  config.annotations_loc = './instances_val2017.json'

  # Logging/checkpointing
  steps_per_epoch = COCO_TRAIN_SIZE // config.batch_size
  config.checkpoint = True
  config.xprof = False
  config.log_summary_steps = 200
  config.log_eval_steps = steps_per_epoch
  return config
