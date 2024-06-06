"""Utility functions for using pretrained models."""

import re
from typing import Any, List, Mapping, Optional, Union

from absl import logging
import flax

from train_lib import train_utils

PyTree = Union[Mapping[str, Any], Any]


def _replace_dict(model: PyTree,
                  restored: PyTree,
                  ckpt_prefix_path: Optional[List[str]] = None,
                  model_prefix_path: Optional[List[str]] = None,
                  name_mapping: Optional[Mapping[str, str]] = None,
                  skip_regex: Optional[str] = None) -> PyTree:
  """Replaces values in model dictionary with `restored` ones from checkpoint."""
  name_mapping = name_mapping or {}

  model = flax.core.unfreeze(model)
  restored = flax.core.unfreeze(restored)

  # Traverse `restored` down to what needs to be restored. This is like
  # traversing to the source path in the `restored` param pytree.
  if ckpt_prefix_path:
    for p in ckpt_prefix_path:
      restored = restored[p]

  # Build the destination subtree from the restored parameters. For instance,
  # If one wants to restore 'backbone/resnet/block01' from `restored` to
  # 'backbone/block01` under `model`,
  if model_prefix_path:
    for p in reversed(model_prefix_path):
      restored = {p: restored}

  # Flatten subtrees to a dict of str -> tensor. Keys are tuples from the path
  # in the nested dictionary to the specific tensor. For instance,
  # {'a': {'b': 1.0, 'c': 2.0'}, 'd': 3.0}
  # -> {('a', 'b'): 1.0, ('a', 'c'): 2.0, ('d',): 3.0}
  restored_flat = flax.traverse_util.flatten_dict(
      dict(restored), keep_empty_nodes=True)
  model_flat = flax.traverse_util.flatten_dict(
      dict(model), keep_empty_nodes=True)
  for m_key, m_params in restored_flat.items():
    for name, to_replace in name_mapping.items():
      m_key = tuple(to_replace if k == name else k for k in m_key)
    m_key_str = '/'.join(m_key)
    if m_key not in model_flat:
      logging.warning('%s does not exist in model. Skip.', m_key_str)
      continue
    if skip_regex and re.findall(skip_regex, m_key_str):
      logging.info('Skip loading parameter %s (regex).', m_key_str)
      continue
    logging.info('Loading %s from checkpoint', m_key_str)
    model_flat[m_key] = m_params

  return flax.core.freeze(flax.traverse_util.unflatten_dict(model_flat))


def init_from_pretrain_state(
    train_state: train_utils.TrainState,
    pretrain_state: Union[PyTree, train_utils.TrainState],
    ckpt_prefix_path: Optional[List[str]] = None,
    model_prefix_path: Optional[List[str]] = None,
    name_mapping: Optional[Mapping[str, str]] = None,
    skip_regex: Optional[str] = None) -> train_utils.TrainState:
  """Updates `train_state` with data from `pretrain_state`.
  
  Args:
    train_state: A raw TrainState for the model.
    pretrain_state: A TrainState that is loaded with parameters/state of a 
      pretrained model.
    ckpt_prefix_path: Path to subtree of `pretrain_state` that needs to 
      be restored. Default (None), will restore entire subtree, both for 
      `params` and `model_state`, src-like.
    model_prefix_path: Path to subtree of `train_state` where parameters 
      from `pretrain_state` subtree will be restored, both for `params` and 
      `model_state`, destination-like.
    name_mapping: Mapping of parameter names from checkpoint to this model.
    skip_regex: If there is a parameter whose parent keys match the regex,
      the parameter will not be replaced from pretrain_state.

  Returns:
    Updated train_state.
  """
  name_mapping = name_mapping or {}
  restored_params = pretrain_state['params']
  restored_model_state = pretrain_state['model_state']
  model_params = _replace_dict(train_state.params, restored_params,
                               ckpt_prefix_path, model_prefix_path,
                               name_mapping, skip_regex)
  train_state = train_state.replace(params=model_params)

  if (restored_model_state is not None and train_state.model_state is not None):
    if model_prefix_path:
      # Insert model prefix after batch_stats
      model_prefix_path = ['batch_stats'] + model_prefix_path
      if 'batch_stats' in restored_model_state:
        ckpt_prefix_path = ckpt_prefix_path or []
        ckpt_prefix_path = ['batch_stats'] + ckpt_prefix_path
    elif 'batch_stats' not in restored_model_state:
      raise NotImplementedError('Backward compatibility not supported.')
    if ckpt_prefix_path and ckpt_prefix_path[0] != 'batch_stats':
      ckpt_prefix_path = ['batch_stats'] + ckpt_prefix_path
    model_state = _replace_dict(train_state.model_state, restored_model_state,
                                ckpt_prefix_path, model_prefix_path,
                                name_mapping, skip_regex)
    train_state = train_state.replace(model_state=model_state)
  return train_state
