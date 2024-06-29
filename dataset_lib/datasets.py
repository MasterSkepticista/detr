# ----------------------------------------------------------------
# Modified from Scenic (https://github.com/google-research/scenic)
# Copyright 2024 The Scenic Authors.
# ----------------------------------------------------------------
import importlib
from typing import Callable, List

from absl import logging
from dataset_lib import dataset_utils


class DatasetRegistry(object):
  """Static class for keeping track of available datasets."""
  _REGISTRY = {}

  @classmethod
  def add(cls, name: str, builder_fn: Callable[..., dataset_utils.Dataset]):
    """Adds a dataset to the registry.
    
    Args:
      name: A unique dataset name.
      builder_fn: Function to be called to construct the datasets.
    
    Raises:
      KeyError: If the provided name is not unique.
    """
    if name in cls._REGISTRY:
      raise KeyError(f'Dataset with name ({name}) already registered.')
    cls._REGISTRY[name] = builder_fn

  @classmethod
  def get(cls, name: str) -> Callable[..., dataset_utils.Dataset]:
    """Get a dataset form the registry by its name.
    
    Args:
      name: Dataset name.

    Returns:
      Dataset builder function that accepts dataset-specific parameters and 
      returns a dataset description.
    
    Raises:
      KeyError: If the dataset is not found.
    """
    if name not in cls._REGISTRY:
      if name in _IMPORT_TABLE:
        module = importlib.import_module(_IMPORT_TABLE[name])
        logging.info('On-demand import of dataset (%s) from module (%s)', name,
                     module)
      else:
        raise KeyError(f'Unknown dataset ({name}). Did you register this yet?')
    return cls._REGISTRY[name]

  @classmethod
  def list(cls) -> List[str]:
    """List registered datasets."""
    return list(cls._REGISTRY.keys())


def add_dataset(name: str):
  """Decorator for shorthand dataset registration."""

  def inner(
      builder_fn: Callable[..., dataset_utils.Dataset]
  ) -> Callable[..., dataset_utils.Dataset]:
    DatasetRegistry.add(name, builder_fn)
    return builder_fn

  return inner
