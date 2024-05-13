"""Data generators for COCO-style object detection datasets."""
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import jax_utils

from dataset_lib import dataset_utils
import transforms

# Values from ImageNet (as used by backbone)
_MEAN_RGB = [0.48, 0.456, 0.406]
_STD_RGB = [0.229, 0.224, 0.225]


def make_coco_transforms(split_name: str, max_size: int = 1333):
  """Returns augmentation/preprocessing for images and labels."""
  scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
  ratio = max_size / 1333.

  scales = [int(ratio * s) for s in scales]

  # These scales are as per DETR torch implementation for RandomResize -> Crop
  scales2 = [int(ratio * s) for s in [400, 500, 600]]

  normalize_boxes = transforms.NormalizeBoxes()
  init_padding_mask = transforms.InitPaddingMask()

  if split_name == 'train':
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomSelect(
            transforms.RandomResize(scales, max_size=max_size),
            transforms.Compose([
                transforms.RandomResize(scales2),
                transforms.RandomSizeCrop(int(ratio * 384), int(ratio * 600)),
                transforms.RandomResize(scales, max_size=max_size),
            ]),
        ),
        normalize_boxes,
        init_padding_mask,
    ])
  elif split_name == 'validation':
    return transforms.Compose([
        transforms.Resize(max(scales), max_size=max_size),
        normalize_boxes,
        init_padding_mask,
    ])
  else:
    raise ValueError(f'Transforms not defined for split `{split_name}`')


def decode_boxes(boxes, size):
  """Convert yxyx[0-1] boxes from TF-Example to xyxy unnormalized."""
  h = tf.cast(size[0], tf.float32)
  w = tf.cast(size[1], tf.float32)

  y0, x0, y1, x1 = tf.split(boxes, 4, axis=-1)

  x0 = tf.clip_by_value(x0 * w, 0.0, w)
  y0 = tf.clip_by_value(y0 * h, 0.0, h)
  x1 = tf.clip_by_value(x1 * w, 0.0, w)
  y1 = tf.clip_by_value(y1 * h, 0.0, h)
  return tf.concat([x0, y0, x1, y1], axis=-1)


def decode_coco_detection_example(example, input_range=None):
  """Creates an <input, label> pair from a serialized TF Example."""

  image = example['image']
  decoded = image.dtype != tf.string

  if not decoded:
    image = tf.io.decode_image(image, channels=3, expand_animations=False)

  image = tf.image.convert_image_dtype(image, tf.float32)

  # Normalize
  if input_range:
    image = image * (input_range[1] - input_range[0]) + input_range[0]
  else:
    mean_rgb = tf.constant(_MEAN_RGB, shape=[1, 1, 3], dtype=tf.float32)
    std_rgb = tf.constant(_STD_RGB, shape=[1, 1, 3], dtype=tf.float32)
    image = (image - mean_rgb) / std_rgb

  boxes = decode_boxes(example['objects']['bbox'], tf.shape(image)[0:2])

  target = {
      'area': example['objects']['area'],
      'boxes': boxes,
      'objects/id': example['objects']['id'],
      'is_crowd': example['objects']['is_crowd'],
      'labels': example['objects']['label'] + 1,  # 0'th class will be bg
  }

  # Filter degenerate objects
  keep = tf.where(
      tf.logical_and(boxes[:, 2] > boxes[:, 0], boxes[:, 3] > boxes[:, 1]))[:,
                                                                            0]
  target_kept = {k: tf.gather(v, keep) for k, v in target.items()}
  target_kept['orig_size'] = tf.shape(image)[0:2]
  target_kept['image/id'] = example['image/id']

  return {
      'inputs': image,
      'label': target_kept,
  }


def load_split_from_tfds(
    builder: tfds.core.DatasetBuilder,
    *,
    train: bool,
    batch_size: int,
    decode_fn: Callable,
    preprocess_fn: Callable,
    max_size: int,
    max_boxes: int,
) -> Tuple[tf.data.Dataset, tfds.core.DatasetInfo]:

  split = 'train' if train else 'validation'
  data_range = tfds.even_splits(split, jax.process_count())[jax.process_index()]
  ds = builder.as_dataset(
      data_range,
      shuffle_files=False,
      decoders={'image': tfds.decode.SkipDecoding()})
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)
  ds = ds.cache()

  # Padding structure for each tensor of the example.
  padded_shapes = {
      'inputs': [max_size, max_size, 3],
      'padding_mask': [max_size, max_size],
      'label': {
          'boxes': [max_boxes, 4],
          'area': [max_boxes,],
          'objects/id': [max_boxes,],
          'is_crowd': [max_boxes,],
          'labels': [max_boxes,],
          'image/id': [],
          'size': [2,],
          'orig_size': [2,],
      },
  }

  if train:
    ds = ds.shuffle(64 * batch_size,)
    ds = ds.repeat()
    ds = ds.map(decode_fn, tf.data.AUTOTUNE)
    ds = ds.map(preprocess_fn, tf.data.AUTOTUNE)
    ds = ds.padded_batch(batch_size, padded_shapes, drop_remainder=True)
  else:
    ds = ds.map(decode_fn, tf.data.AUTOTUNE)
    ds = ds.map(preprocess_fn, tf.data.AUTOTUNE)
    ds = ds.padded_batch(batch_size, padded_shapes, drop_remainder=False)
    ds = ds.cache()  # WARNING! Only if you have enough memory.
    ds = ds.repeat()

  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds, builder.info


def build_pipeline(*, rng, batch_size: int, eval_batch_size: int,
                   num_shards: int, dataset_configs: ml_collections.ConfigDict):
  del rng

  builder = tfds.builder(dataset_configs.name)

  max_size = dataset_configs.get('max_size', 1333)
  max_boxes = dataset_configs.get('max_boxes', 100)

  train_preprocess_fn = make_coco_transforms('train', max_size)
  eval_preprocess_fn = make_coco_transforms('validation', max_size)

  decode_fn = partial(
      decode_coco_detection_example,
      input_range=dataset_configs.get('input_range'))
  train_ds, ds_info = load_split_from_tfds(
      builder,
      train=True,
      batch_size=batch_size,
      decode_fn=decode_fn,
      preprocess_fn=train_preprocess_fn,
      max_size=max_size,
      max_boxes=max_boxes)

  eval_ds, _ = load_split_from_tfds(
      builder,
      train=False,
      batch_size=eval_batch_size,
      decode_fn=decode_fn,
      preprocess_fn=eval_preprocess_fn,
      max_size=max_size,
      max_boxes=max_boxes)

  # 0 is the background class, dataset classes run from 1..N
  num_classes = ds_info.features['objects']['label'].num_classes + 1

  maybe_pad_batches_train = partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size)
  maybe_pad_batches_eval = partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size)
  shard_batches = partial(dataset_utils.shard, num_shards=num_shards)

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(shard_batches, train_iter)

  eval_iter = iter(eval_ds)
  eval_iter = map(dataset_utils.tf_to_numpy, eval_iter)
  eval_iter = map(maybe_pad_batches_eval, eval_iter)
  eval_iter = map(shard_batches, eval_iter)

  if dataset_configs.get('prefetch_to_device'):
    train_iter = jax_utils.prefetch_to_device(
        train_iter, dataset_configs.get('prefetch_to_device', 2))
    eval_iter = jax_utils.prefetch_to_device(
        eval_iter, dataset_configs.get('prefetch_to_device', 2))

  meta_data = {
      'num_classes': num_classes,
      'input_shape': [-1, max_size, max_size, 3],
      'num_train_examples': builder.info.splits['train'].num_examples,
      'num_eval_examples': builder.info.splits['validation'].num_examples,
      'input_dtype': jnp.float32,
      'target_is_onehot': False
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)
