"""ResNet with PyTorch weights.

Refer to the gist below on steps to convert torchvision weights to a checkpoint
that can be read by the `load(...)` function here.

https://gist.github.com/MasterSkepticista/c854bce837a5cb5ca0489bd33b3a2259
"""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from common_lib.tree_utils import recover_tree
from model_lib.layers import nn_layers

PyTree = Any


class Downsample(nn.Module):
  """Conv + BN downsample layer."""
  num_filters: int
  strides: Tuple[int, int]
  conv: Callable[..., Any]
  norm: Callable[..., Any]

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = self.conv(self.num_filters, (1, 1), strides=self.strides, name="0")(x)
    x = self.norm(name="1")(x)
    return x


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
  bottleneck: bool
  zero_scale_init: bool
  strides: Tuple[int, int] = (1, 1)
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
    needs_projection = x.shape[-1] != self.filters * 4 or self.strides != (1, 1)
    nout = self.filters * 4 if self.bottleneck else self.filters

    norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype)
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)

    residual = x
    if needs_projection:
      residual = Downsample(
          nout,
          strides=self.strides,
          conv=conv,
          norm=norm,
          name="downsample",
      )(residual)  # yapf: disable

    if self.bottleneck:
      x = conv(self.filters, (1, 1), name='conv1')(x)
      x = norm(name='bn1')(x)
      x = nn_layers.IdentityLayer(name='relu1')(nn.relu(x))

    y = conv(
        self.filters, (3, 3),
        self.strides,
        padding=[(1, 1), (1, 1)],
        name='conv2')(
            x)
    y = norm(name='bn2')(y)
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

    y = norm(name='bn3', scale_init=scale_init)(y)
    y = nn_layers.IdentityLayer(name='relu3')(nn.relu(residual + y))
    return y


class Stage(nn.Module):
  """Single stage of sequential residual blocks.
  
  Attributes:
    num_units: Number of residual blocks.
    filters: Filter count exiting this stage.
    zero_scale_init: If True, initialize scale parameter of last residual block
      of the stage with zeros. This makes the block behave like an identity.
    first_stride: Stride applied on the first block of this stage.
    bottleneck: If True, use bottleneck variant of the block.
    dtype: DType of the computation. Defaults to float32.
  """
  num_units: int
  filters: int
  zero_scale_init: bool
  first_stride: Tuple[int, int] = (1, 1)
  bottleneck: bool = True
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:

    residual_block = functools.partial(
        ResidualBlock,
        bottleneck=self.bottleneck,
        zero_scale_init=self.zero_scale_init,
        dtype=self.dtype)

    x = residual_block(
        self.filters, strides=self.first_stride, name='0')(x, train)

    for i in range(1, self.num_units):
      x = residual_block(self.filters, name=str(i))(x, train)

    return x


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
  zero_scale_init: bool = False
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

    # Stem blocks.
    x = nn.Conv(
        num_filters,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding=[(3, 3), (3, 3)],
        use_bias=False,
        dtype=self.dtype,
        name='conv1')(x)  # yapf: disable
    x = nn.BatchNorm(
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
        name='bn1')(x)  # yapf: disable
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])

    stage = functools.partial(
        Stage,
        bottleneck=bottleneck,
        zero_scale_init=self.zero_scale_init,
        dtype=self.dtype)

    # Stage 1: No downsampling with stride as root block already does once.
    representations = {}
    x = representations['layer1'] = stage(
        block_sizes[0], filters=num_filters, name='layer1')(x, train)

    for i, block_size in enumerate(block_sizes[1:], 1):
      x = representations[f'layer{i+1}'] = stage(
          block_size,
          filters=num_filters * 2**i,
          first_stride=(2, 2),
          name=f'layer{i+1}')(x, train)

    # Head.
    if self.num_classes:
      x = jnp.mean(x, axis=(1, 2))
      x = nn_layers.IdentityLayer(name='pre_logits')(x)
      x = nn.Dense(
          self.num_classes,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          dtype=self.dtype,
          name='fc')(
              x)
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


def load(params: PyTree, batch_stats: PyTree,
         path: str) -> Tuple[PyTree, PyTree]:
  """Loads PyTorch weights created using the conversion tool.

  Gist: https://gist.github.com/MasterSkepticista/c854bce837a5cb5ca0489bd33b3a2259
  Note: This function does not support partial loading of parameters.
  
  Args:
    params: A PyTree of params, used to assert tree structure.
    batch_stats: A PyTree of batch stats, used to assert tree structure.
    path: str, path to npz checkpoint file.
  
  Returns:
    Tuple of params and batch_stats pytrees loaded from path.
  
  Raises:
    AssertionError if either of params or batch_stats tree structures do not
    match those of loaded weights.
  """
  # This implementation assumes npz to contain the following dict
  # variables = {
  #   "params/a/b/0/kernel": np.ndarray,
  #   "params/a/b/1/kernel": np.ndarray,
  #   ...,
  #   "batch_stats/x/y/mean": np.ndarray,
  #   "batch_stats/x/y/var": np.ndarray,
  # }
  variables = np.load(path)
  names, values = zip(*list(variables.items()))
  variables = recover_tree(names, values)
  restored_params, restored_batch_stats = (variables["params"],
                                           variables["batch_stats"])
  # Verify both trees have same structure.
  assert jax.tree.structure(params) == jax.tree.structure(restored_params)
  assert jax.tree.structure(batch_stats) == jax.tree.structure(
      restored_batch_stats)

  return restored_params, restored_batch_stats
