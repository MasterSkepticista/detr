"""Utilities to operate on PyTrees."""
import dataclasses
import re
from typing import Any, Callable, Generator, List, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging

PyTree = Any


def check_and_compile_patterns(patterns):
  """Validates and compiles a list of param patterns.
  
  The validation consists of checking for common mistakes, such that, the 
  pattern does not start with a slash.

  Args:
    patterns: a single (string), pattern (regex) or a list of patterns.
  
  Returns:
    A list of verified and compiled regexes.
  """
  if isinstance(patterns, str):
    patterns = [patterns]

  assert isinstance(patterns,
                    (list, tuple)), f'Must be list/tuple, found {patterns}'

  def check_and_compile(pattern):
    assert not pattern.startswith('/'), (
        f'Parameter names never starts with `/`: {pattern}')
    return re.compile(pattern)

  return list(map(check_and_compile, patterns))


def make_mask_trees(tree, patterns, *, log=None):
  """Returns a boolean mask tree for every pattern (only first match)."""
  compiled_patterns = check_and_compile_patterns(patterns)

  def matchfirst(name, _):
    # Given a set of patterns, this function checks for a `name` match for each
    # pattern. The first match is considered `True`, rest are all considered
    # `False`. In effect, avoid patterns that can cause overlapping matches.
    matches = []
    for pattern in compiled_patterns:
      # Should be a full match *and* must not have been caught before.
      matches.append(not any(matches) and bool(pattern.fullmatch(name)))
    if log is not None and True in matches and jax.process_index() == 0:
      logging.info('%s: %s matched by %s', log, name,
                   patterns[matches.index(True)])
    return np.asarray(matches)

  # This returns a PyTree, with each leaf node holding boolean mask array for
  # each pattern that it is applicable for.
  multimask = tree_map_with_names(matchfirst, tree)

  # Split multimask into separate trees, one for each pattern.
  return [
      jax.tree_map(lambda matches: matches[idx], multimask)
      for idx in range(len(patterns))
  ]


def _traverse_with_names(
    tree: PyTree) -> Generator[Tuple[str, PyTree], None, None]:
  """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
  if dataclasses.is_dataclass(tree):
    tree = flax.serialization.to_state_dict(tree)
  if isinstance(tree, (dict, flax.core.frozen_dict.FrozenDict)):
    keys = sorted(tree.keys())
    for key in keys:
      for path, val in _traverse_with_names(tree[key]):
        yield (key + '/' + path).rstrip('/'), val
  else:
    yield '', tree


def tree_flatten_with_names(
    tree: PyTree) -> Tuple[List[Tuple[str, jnp.ndarray]], PyTree]:
  """Populates tree_flatten with leaf names.
  
  This function populates output of tree_flatten with leaf names, using a 
  custom traversal that produces names. The custom trversal does not necessarily
  have the same order as jax, but this function takes care of automatically
  aligning jax' and custom traversals.

  Args:
    tree: A PyTree.

  Returns:
    A list of values with names [(name, value), ...] and the tree_def.
  """
  vals, tree_def = jax.tree_util.tree_flatten(tree)

  # Fake token tree that is used to track Jax's internal tree traversal and
  # adjust our custom tree traversal to be compatible with it.
  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)

  # It just so happens (at the time of writing), Jax uses the same traversal
  # method as `_traverse_with_names`, and therefore `perm` is perfectly sorted
  # making the `np.argsort` redundant.
  # We could in fact directly call `_traverse_with_names(tree)` for same result.
  # Even if Jax's traversal were to change, it is a safeguard to use a fake
  # token-tree and argsort to determine the order. It doesn't hurt, you know!
  val_names, perm = zip(*_traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traversal should visit the same number of leaves.
  assert len(val_names) == len(vals)
  return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def tree_map_with_names(f: Callable[[jnp.ndarray], jnp.ndarray], tree: PyTree,
                        *rest) -> PyTree:
  """Like `jax.tree_map` but with a filter on the leaf path name.
  
  Args:
    f: A function that takes first paramter `name` (path-like "a/b/c"), remaining
      parameters values of `tree` and `*rest` corresponding to the given `name`
      should return a new value for parameter `name`.
    tree: The tree of paramters `f` should be applied to.
    *rest: more trees of the exact same structure.

  Returns:
    A tree identical in structure to `tree` and `*rest` but with the leaves 
    resulting from calling `f` on corresponding name/leaves in `tree` and `rest`
  """
  names_and_vals, tree_def = tree_flatten_with_names(tree)
  names, vals = zip(*names_and_vals)
  rest_vals = [list(zip(*tree_flatten_with_names(t)[0])[1]) for t in rest]
  vals = [f(*name_and_vals) for name_and_vals in zip(names, vals, *rest_vals)]
  return tree_def.unflatten(vals)
