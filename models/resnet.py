"""Implementation of ResNet.

Modifications from google-research/scenic:
* Initializations for Conv2D and Dense match torchvision reference.
"""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp
from model_lib.layers import nn_layers


class ResidualBlock(nn.Module):
  """Bottleneck ResNet block.
  
  Attributes:
    filters: Number of filters.
    strides: Tuple of ints, strides applied in the 3x3 conv layer.
    bottleneck: Whether to build a bottleneck version of residual.
    zero_scale_init: Whether to initialize scale parameter of last normalization
      of residual block as zeros instead of ones. 
    dtype: DType of the computation.
  """
  filters: int
  strides: Tuple[int, int] = (1, 1)
  bottleneck: bool = True
  zero_scale_init: bool = False
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
    needs_projection = x.shape[-1] != self.filters * 4 or self.strides != (1, 1)
    nout = self.filters * 4 if self.bottleneck else self.filters

    batch_norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype)
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)

    residual = x
    if needs_projection:
      residual = conv(nout, (1, 1), self.strides, name='proj_conv')(residual)
      residual = batch_norm(name='proj_bn')(residual)
    
    if self.bottleneck:
      x = conv(self.filters, (1, 1), name='conv1')(x)
      x = batch_norm(name='bn1')(x)
      x = nn_layers.IdentityLayer(name='relu1')(nn.relu(x))
    
    y = conv(self.filters, (3, 3), self.strides, padding=[(1, 1), (1, 1)], name='conv2')(x)
    y = batch_norm(name='bn2')(y)
    y = nn_layers.IdentityLayer(name='relu2')(nn.relu(y))

    if self.bottleneck:
      y = conv(nout, (1, 1), name='conv3')(y)
    else:
      y = conv(nout, (3, 3), padding=[(1, 1), (1, 1)], name='conv3')(y)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block
    # behaves like an identity. This improves the model by 0.2~0.3%
    # according to https://arxiv.org/abs/1706.02677
    scale_init = nn.initializers.ones
    if self.zero_scale_init:
      scale_init = nn.initializers.zeros

    y = batch_norm(name='bn3', scale_init=scale_init)(y)
    y = nn_layers.IdentityLayer(name='relu3')(nn.relu(residual + y))
    return y


class ResNet(nn.Module):
  """ResNet architecture.
  
  Attributes:
    num_classes: Num output classes. If None, a dict of intermediate features
      is returned.
    width: Multiplier for filter widths. Default is 1.
    depth: Number of layers from standard ResNet configurations. Default is 50.
    dtype: DType of computation (default: float32).
  """
  num_classes: Optional[int] = None
  width: int = 1
  depth: int = 50
  kernel_init: Callable[..., Any] = nn.initializers.lecun_normal()
  bias_init: Callable[..., Any] = nn.initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      train: bool = False) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Applies ResNet model to the inputs.
    
    Args:
      x: Inputs to the model.
      train: Whether it is training or not.
    
    Returns:
      Un-normalized logits and a dict of representations
    """
    num_filters = 64 * self.width
    if self.depth not in BLOCK_SIZE_OPTIONS:
      raise ValueError('Please provide a valid number of layers.')
    block_sizes, bottleneck = BLOCK_SIZE_OPTIONS[self.depth]
    x = nn.Conv(
        num_filters,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding=[(3, 3), (3, 3)],
        use_bias=False,
        dtype=self.dtype,
        name='stem_conv')(
            x)
    x = nn.BatchNorm(
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
        name='init_bn')(
            x)
    x = nn_layers.IdentityLayer(name='init_relu')(nn.relu(x))
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])
    x = nn_layers.IdentityLayer(name='stem_pool')(x)

    residual_block = functools.partial(
        ResidualBlock, dtype=self.dtype, bottleneck=bottleneck)
    representations = {'stem': x}
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        filters = num_filters * 2**i
        x = residual_block(filters=filters, strides=strides)(x, train)
      representations[f'stage_{i+1}'] = x
    
    # Head.
    if self.num_classes:
      x = jnp.mean(x, axis=(1, 2))
      x = nn_layers.IdentityLayer(name='pre_logits')(x)
      x = nn.Dense(
        self.num_classes, 
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dtype=self.dtype, 
        name='output_projection')(x)
    return x, representations


# A dictionary mapping the number of layers in a resnet to the number of 
# blocks in each stage of the model. The second argument indicates whether to
# use bottleneck layers or not.
BLOCK_SIZE_OPTIONS = {
    18: ([2, 2, 2, 2], False),
    26: ([2, 2, 2, 2], True),
    34: ([3, 4, 6, 3], False),
    50: ([3, 4, 6, 3], True),
    101: ([3, 4, 23, 3], True),
    152: ([3, 8, 36, 3], True),
    200: ([3, 24, 36, 3], True)
}
