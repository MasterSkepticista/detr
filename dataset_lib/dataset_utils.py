# ----------------------------------------------------------------
# Modified from Scenic (https://github.com/google-research/scenic)
# Copyright 2024 The Scenic Authors.
# ----------------------------------------------------------------

"""Utils to handle dataset."""
import dataclasses
from typing import Any, Dict, Iterator

import jax
import jax.numpy as jnp
import numpy as np

PyTree = Any
DatasetIterator = Iterator[Any]


@dataclasses.dataclass
class Dataset:
  train_iter: DatasetIterator = None
  valid_iter: DatasetIterator = None
  test_iter: DatasetIterator = None
  meta_data: Dict[str, Any] = dataclasses.field(default_factory=dict)


def maybe_pad_batch(batch: Dict[str, PyTree],
                    train: bool,
                    batch_size: int,
                    inputs_key: str = 'inputs',
                    batch_dim: int = 0):
  """Zero pad the batch towards the right to the batch_size.
  
  All leave tensors in the batch pytree will be padded. This function expects 
  the root structure of the batch pytree to be a dictionary, and it returns a
  dictionary with the same nested structure. This function will add a key 
  `batch_mask` added to the root dict, with 1.0 indicating indices that is
  true data and 0.0 indicating padded indices. `batch_mask` will be used for 
  calculating weighted cross entropy or weighted accuracy.

  In this function, we assume the last partial batch from the training set, so
  if the batch is from training set (i.e. `train=True`), or when the batch is
  from the test/validation set, but it is a complete batch, we *modify* the 
  the batch dict by adding an array of ones as the `batch_mask` of all examples
  in the batch. Otherwise, we create a new dict that has the padded batch and 
  its corresponding `batch_mask` array.

  Args:
    batch: A batch pytree. If `inputs_key` is not set, first leaf array shape
      is used to compute the batch_size.
    train: if the passed batch comes from a training set. No padding is done.
    batch_size: All arrays in the pytree will be padded to have first dimension
      equal to `batch_size`.
    inputs_key: Indicating the key used for the input that we do batch padding
      based on.
    batch_dim: Batch dimension. The default is 0, but it can be different if a
      sharded batch is given.
  
  Returns:
    A dictionary mapping the same keys to the padded batches. Additionally, we
    add a key representing weights, to indicate which indices were padded.
  """
  assert batch_dim >= 0, f'batch_dim=={batch_dim} is expected to be >= 0'

  sample_tensor = batch[inputs_key]
  assert sample_tensor.shape[batch_dim] <= batch_size, (
      f'The indicated target batch_size is {batch_size}, but the size of the '
      f'current batch is larger than that: {sample_tensor.shape[batch_dim]}')

  pad_length = batch_size - sample_tensor.shape[batch_dim]

  if train and pad_length != 0:
    raise ValueError(
        'This function assumes `train` set to drop the last '
        'partial batch. Please use `drop_remainder=True` for `train` split.')

  assert 'batch_mask' not in batch, (
      'When the labels of the task are not pixel-level, batch_mask should not '
      'be already present in the batch.')
  unpadded_mask_shape = sample_tensor.shape[:batch_dim + 1]

  # Most don't need padding. Return quickly to avoid slowdown.
  if train or pad_length == 0:
    if 'batch_mask' not in batch:
      batch['batch_mask'] = np.ones(unpadded_mask_shape, dtype=np.float32)
    return batch

  # When padding is actually needed
  def zero_pad(array):
    pad_width = ([(0, 0)] * batch_dim + [(0, pad_length)] + [(0, 0)] *
                 (array.ndim - batch_dim - 1))
    return np.pad(array, pad_width, mode='constant')

  padded_batch = jax.tree_map(zero_pad, batch)
  padded_batch_mask = zero_pad(np.ones(unpadded_mask_shape, dtype=np.float32))
  if 'batch_mask' in padded_batch:
    padded_batch['batch_mask'] *= padded_batch_mask
  else:
    padded_batch['batch_mask'] = padded_batch_mask
  return padded_batch


def tf_to_numpy(batch):
  """Converts each tf tensor in `batch` to numpy array without copy."""
  # Zero-copy conversion from tf to numpy
  return jax.tree_map(lambda v: v._numpy(), batch)


def shard(batch, num_shards=None):
  """Reshapes tensors in `batch` with leading dimension split across 
  `num_shards`."""

  if not num_shards:
    num_shards = jax.local_device_count()

  def _shard(x: jnp.ndarray):
    return x.reshape((num_shards, -1) + x.shape[1:])

  batch = jax.tree_map(_shard, batch)
  return batch


def unshard(pytree):
  """Reshapes all arrays in the pytree from [ndev, bs, ...] to [host_bs, ...].
  
  Args:
    pytree: A pytree of sharded arrays.
  
  Returns:
    Unsharded data.
  """

  def _unshard_array(array):
    ndev, bs = array.shape[:2]
    return array.reshape((ndev * bs,) + array.shape[2:])

  return jax.tree_util.tree_map(_unshard_array, pytree)
