"""BiT ResNet models as in the paper: https://arxiv.org/pdf/1912.11370.pdf"""
import functools
from typing import Optional, Sequence, Union

import flax.linen as nn
import jax.numpy as jnp


def standardize(x, axis, eps):
  x = x - jnp.mean(x, axis=axis, keepdims=True)
  x = x / jnp.sqrt(jnp.mean(jnp.square(x), axis=axis, keepdims=True) + eps)
  return x


class StdConv(nn.Conv):
  """Implements Weight Standardization for Conv2D."""

  def param(self, name, *a, **kw):
    param = super().param(name, *a, **kw)
    if name == "kernel":
      param = standardize(param, axis=(0, 1, 2), eps=1e-10)
    return param


class RootBlock(nn.Module):
  """Root block of ResNetV2."""
  width: int
  dtype: Optional[jnp.dtype] = jnp.float32

  @nn.compact
  def __call__(self, x):
    x = StdConv(
        self.width,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding=[(3, 3), (3, 3)],
        use_bias=False,
        name='conv_root',
        dtype=self.dtype)(
            x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])
    return x


class BottleneckUnit(nn.Module):
  """Bottleneck residual unit with pre-activation."""
  filters: int
  strides: Sequence[int] = (1, 1)
  dtype: Optional[jnp.dtype] = jnp.float32

  @nn.compact
  def __call__(self, x):
    out_filters = self.filters * 4

    conv = functools.partial(StdConv, use_bias=False, dtype=self.dtype)
    norm = functools.partial(nn.GroupNorm, dtype=self.dtype)

    residual = x
    x = norm(name="gn1")(x)
    x = nn.relu(x)

    if x.shape[-1] != out_filters or self.strides != (1, 1):
      residual = conv(
          out_filters, (1, 1), strides=self.strides, name='conv_proj')(
              x)

    x = conv(self.filters, (1, 1), name='conv1')(x)

    x = norm(name='gn2')(x)
    x = nn.relu(x)
    x = conv(
        self.filters, (3, 3),
        self.strides,
        padding=[(1, 1), (1, 1)],
        name='conv2')(
            x)

    x = norm(name='gn3')(x)
    x = nn.relu(x)
    x = conv(out_filters, (1, 1), name='conv3')(x)
    x = x + residual
    return x


class Stage(nn.Module):
  """Single ResNetV2 Stage (sequence of same-resolution units)."""
  size: int
  filters: int
  first_stride: Sequence[int] = (1, 1)
  dtype: Optional[jnp.dtype] = jnp.float32

  @nn.compact
  def __call__(self, x):
    out = {}
    x = out["unit01"] = BottleneckUnit(
        self.filters, self.first_stride, name='unit01', dtype=self.dtype)(
            x)
    for i in range(1, self.size):
      x = out[f'unit{i+1:02d}'] = BottleneckUnit(
          self.filters, name=f'unit{i+1:02d}', dtype=self.dtype)(
              x)
    return x, out


class ResNet(nn.Module):
  """BiT ResNet model."""
  num_classes: Optional[int] = None
  width: int = 1
  depth: Union[int, Sequence[int]] = 50
  dtype: Optional[jnp.dtype] = jnp.float32

  @nn.compact
  def __call__(self, x, train=False):
    blocks = get_block_desc(self.depth)
    width = int(64 * self.width)
    out = {}

    x = out['stem'] = RootBlock(width, dtype=self.dtype, name='root_block')(x)

    # Stage 1: No downsampling with stride as root block already does once.
    x, out['stage1'] = Stage(
        blocks[0], filters=width, name='block1', dtype=self.dtype)(
            x)
    for i, block_size in enumerate(blocks[1:], 1):
      x, out[f'stage{i+1}'] = Stage(
          block_size,
          filters=width * 2**i,
          first_stride=(2, 2),
          name=f'block{i+1}',
          dtype=self.dtype)(
              x)
    # Pre-head
    x = out['norm_pre_head'] = nn.GroupNorm(
        dtype=self.dtype, name='norm-pre-head')(
            x)
    x = out['pre_logits_2d'] = nn.relu(x)

    # Head
    if self.num_classes:
      x = out['pre_logits'] = jnp.mean(x, axis=(1, 2))
      head = nn.Dense(
          self.num_classes,
          name='head',
          kernel_init=nn.initializers.zeros,
          dtype=self.dtype)
      x = out['logits'] = head(x)
    return x, out


# A dictionary mapping the number of layers in a resnet to the number of blocks
# in each stage of the model.
# NOTE: Does not include 18/34 as they need non-bottleneck block
def get_block_desc(depth):
  if isinstance(depth, list):
    depth = tuple(depth)
  return {
      26: [2, 2, 2, 2],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }.get(depth, depth)
