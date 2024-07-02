"""Transforms for object detection.

Modifications from Scenic:
* Bug fix for RandomResize crop region.
"""
from typing import Any, Dict

import tensorflow as tf


def tf_int32(t):
  return tf.cast(t, tf.int32)


def tf_float(t):
  return tf.cast(t, tf.float32)


def identity(features: Dict[str, Any]) -> Dict[str, Any]:
  out = {}
  for k, v in features.items():
    if isinstance(v, tf.Tensor):
      out[k] = tf.identity(v)
    elif isinstance(v, dict):
      out[k] = identity(v)
    else:
      raise TypeError(f'Unknown type `{v}` for identity conversion.')
  return out


def get_hw(features, dtype=tf.float32):
  """Returns (h, w) of image as float32 tensors."""
  if isinstance(features, dict):
    image = features['inputs']
    shape = tf.shape(image)
  elif isinstance(features, tf.Tensor):
    shape = tf.shape(features)
  else:
    raise ValueError(f'Unknown type `{features}`')

  h = tf.cast(shape[0], dtype)
  w = tf.cast(shape[1], dtype)
  return (h, w)


class Compose:
  """Chain transforms together.
  
  Attributes:
    transforms: List of `transforms` to apply sequentially.
  """

  def __init__(self, transforms: list):
    self.transforms = transforms

  def __call__(self, features):
    for t in self.transforms:
      features = t(features)
    return features

  def __repr__(self):
    format_string = self.__class__.__name__ + '('
    for t in self.transforms:
      format_string += '\n'
      format_string += f'  {t}'
    format_string += '\n)'
    return format_string


class NormalizeBoxes:
  """Convert xyxy unnormalized boxes to cxcywh[0-1] normalized."""

  def __call__(self, features):
    h, w = get_hw(features['inputs'])

    if 'boxes' in features['label']:
      boxes = features['label']['boxes']

      x0, y0, x1, y1 = tf.split(boxes, 4, axis=-1)
      boxes = tf.concat([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)],
                        axis=-1)
      boxes = boxes / tf.reshape(tf.stack([w, h, w, h]), shape=(1, 4))
      features['label']['boxes'] = boxes

    return features


class InitPaddingMask:
  """Create a padding mask of `ones` to match the current unpadded image."""

  def __call__(self, features):
    h, w = get_hw(features['inputs'])
    features['padding_mask'] = tf.ones([h, w])
    return features


class RandomSelect:
  """Randomly choses between two sets of transforms with probability [p, 1-p]"""

  def __init__(self, transforms1, transforms2, p: float = 0.5):
    self.transforms1 = transforms1
    self.transforms2 = transforms2
    self.p = p

  def __call__(self, features):
    rnd = tf.random.uniform([], 0., 1., tf.float32)
    if rnd < self.p:
      return self.transforms1(identity(features))
    else:
      return self.transforms2(identity(features))


class RandomResize:
  """Randomly resize image to one of the given scales."""

  def __init__(self, scales: list, max_size: int = None):
    self.scales = tf.constant(scales)
    self.max_size = max_size

  def __call__(self, features):
    logits = tf.zeros([1, len(self.scales)])
    idx = tf.random.categorical(logits, 1)[0, 0]
    return resize(features, self.scales[idx], self.max_size)


class RandomHorizontalFlip:
  """Horizontally flip image and features with probability `p`"""

  def __init__(self, p: float = 0.5):
    self.p = p

  def __call__(self, features):
    flip = tf.random.uniform([], 0., 1.) > self.p
    if flip:
      features = hflip(identity(features))
    return features


class RandomSizeCrop:
  """Crop a random region from the image."""

  def __init__(self, min_size: int, max_size: int):
    assert min_size <= max_size
    self.min_size = min_size
    self.max_size = max_size

  def __call__(self, features):
    h, w = get_hw(features['inputs'], dtype=tf.int32)

    # Sample a height/width to crop from the image
    w_crop = tf.random.uniform([],
                               self.min_size,
                               tf.minimum(w, self.max_size),
                               dtype=tf.int32)
    h_crop = tf.random.uniform([],
                               self.min_size,
                               tf.minimum(h, self.max_size),
                               dtype=tf.int32)

    # Sample a coordinate
    i = tf.random.uniform([], 0, h - h_crop + 1, dtype=tf.int32)
    j = tf.random.uniform([], 0, w - w_crop + 1, dtype=tf.int32)
    region = (i, j, h_crop, w_crop)
    return crop(features, region)


class Resize:
  """Resize features with smallest side atleast the given size."""

  def __init__(self, size: int, max_size: int):
    assert isinstance(size, int)
    self.size = size
    self.max_size = max_size

  def __call__(self, features):
    return resize(features, self.size, self.max_size)


def hflip(features):
  """Horizontally flips image `inputs` and corresponding target boxes."""
  image = features['inputs']
  target = features['label']

  flipped_image = tf.image.flip_left_right(image)

  if 'boxes' in target:
    boxes = target['boxes']
    _, w = get_hw(image)
    # Remember, these are decoded and unnormalized box coordinates
    x0, y0, x1, y1 = tf.split(boxes, 4, axis=-1)
    target['boxes'] = tf.concat([w - x1, y0, w - x0, y1], axis=-1)

  features['inputs'] = flipped_image
  features['label'] = target
  return features


def get_size_with_aspect_ratio(image_size, size, max_size=None):
  """Find resulting (h, w) that satisfies original aspect ratio and max_size."""
  h, w = image_size[0], image_size[1]

  if max_size is not None:
    minimum_size = tf_float(tf.minimum(h, w))
    maximum_size = tf_float(tf.maximum(h, w))
    scaling_ratio = tf_float(size) / minimum_size
    if (scaling_ratio * maximum_size) > max_size:
      size = tf_int32(tf.floor(minimum_size * max_size / maximum_size))

  if (h <= w and tf.equal(h, size)) or (w <= h and tf.equal(w, size)):
    return (h, w)

  if h < w:
    oh = size
    ow = tf_int32(w * size / h)
  else:
    ow = size
    oh = tf_int32(h * size / w)

  return (oh, ow)


def resize(features, size: int, max_size: int):
  """Resize image, boxes and other attributes such that smallest side is equal
  to `size`, with largest side at most `max_size`.

  Args:
      features: An unbatched dict of features containing `inputs` and `label`.
      size: Target size of smallest side of the image.
      max_size: Size constraint of largest side of the image after resizing.

  Returns:
      Features dict with resized image and features, addl. key `size`.
  """
  image = features['inputs']
  target = features['label']

  original_size = tf.shape(image)[0:2]
  new_size = get_size_with_aspect_ratio(original_size, size, max_size)
  resized_image = tf.image.resize(image, new_size)
  target['size'] = tf.stack(new_size)

  # Compute resize ratios to be applied to bboxes, area etc.
  r_height = tf_float(new_size[0] / original_size[0])
  r_width = tf_float(new_size[1] / original_size[1])

  if 'boxes' in target:
    x0, y0, x1, y1 = tf.split(target['boxes'], 4, axis=-1)
    target['boxes'] = tf.concat(
        [x0 * r_width, y0 * r_height, x1 * r_width, y1 * r_height], axis=-1)

  if 'area' in target:
    target['area'] = tf_float(target['area']) * (r_height * r_width)

  features['inputs'] = resized_image
  features['label'] = target
  return features


def crop(features, region):
  """Crops `region` from image and adjusts bboxes/area accordingly."""
  image = features['inputs']
  target = features['label']

  i, j, h, w = region
  cropped_image = image[i:i + h, j:j + w, :]
  features['inputs'] = cropped_image
  target['size'] = tf.stack([h, w])

  fields = ['labels', 'is_crowd', 'area', 'objects/id', 'boxes']

  # Adjust boxes/area
  if 'boxes' in target:
    boxes = target['boxes']  # Reminder: xyxy unnormalized coordinates.
    # Case 1. Boxes lie entirely within crop window
    cropped_boxes = boxes - tf_float(tf.reshape(tf.stack([j, i, j, i]), [1, 4]))

    # Case 2. Boxes exceed crop window, in which case, clip the boxes
    cropped_boxes = tf.minimum(
        tf.reshape(cropped_boxes, [-1, 2, 2]),
        tf.reshape(tf_float(tf.stack([w, h])), [1, 1, 2]))

    # Case 3. Boxes begin before crop window (leading to -ve coords) -> clip
    cropped_boxes = tf.nn.relu(cropped_boxes)
    target['boxes'] = tf.reshape(cropped_boxes, [-1, 4])

    # Recompute box area
    if 'area' in target:
      target['area'] = tf.reduce_prod(
          cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :], axis=1)

    # Remove degenerate boxes
    cropped_boxes = tf.reshape(target['boxes'], [-1, 2, 2])
    keep = tf.logical_and(cropped_boxes[:, 1, 0] > cropped_boxes[:, 0, 0],
                          cropped_boxes[:, 1, 1] > cropped_boxes[:, 0, 1])
    for field in fields:
      if field in target:
        target[field] = target[field][keep]

  features['label'] = target
  return features
