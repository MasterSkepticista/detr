"""Gradient Transformations and other optax utilies."""
import operator
from typing import Any, Sequence, Tuple

import jax
import ml_collections
import optax

import utils as u
from common_lib import tree_utils

PyTree = Any


def make(config: ml_collections.ConfigDict, params: PyTree, *, sched_kw: dict):
  """Returns gradient transform and learning rate functions.
  
  Args:
    config: Optimizer config.
    params: A PyTree of model parameters.
    sched_kw: Dict containing information needed to calculate lr schedule steps.
  
  Returns:
    A tuple (optax tx, schedule_fns).
  """
  # config.schedule is expected in a format: [(pattern1, schedule1), ...]
  # A format like (pattern, None) implies `pattern` params are frozen.
  # If not a list/tuple, given schedule is applied to all params (.*)

  schedule = config.schedule
  if not isinstance(schedule, (tuple, list)):
    schedule = [('.*', schedule)]

  # Optax deals with pytree masks to apply certain transforms to certain set
  # of params: be it custom learning rates, weight decay or schedules for
  # certain sets of params.

  # Build masks for schedule functions.
  masks, scheds = _make_mask_trees(params, schedule, log_msg='config.schedule')

  # Split masks to frozen/unfrozen
  frozen_mask, masks, scheds = _split_frozen(masks, scheds)
  not_frozen_mask = jax.tree_map(operator.not_, frozen_mask)

  # Build schedule functions for each mask
  schedule_fns = [
      u.create_learning_rate_schedule(**sched_kw, **sched) for sched in scheds
  ]
  schedule_txs = [
      optax.masked(optax.scale_by_schedule(fn), mask)
      for fn, mask in zip(schedule_fns, masks)
  ] + [
      # Removes weight decay updates. Note that `weight_decay` already uses an
      # independent mask. Instead of trying to combine that mask with others,
      # we simply set updates to zero for `frozen_mask`. This ofcourse is based
      # on the fact that weight decay tx is applied prior to schedule tx.
      optax.masked(optax.set_to_zero(), frozen_mask)
  ]

  # Gradient clipping
  grad_clip_norm_tx = (
      optax.masked(
          optax.clip_by_global_norm(config.grad_clip_norm), not_frozen_mask)
      if 'grad_clip_norm' in config else optax.identity())

  # Optimizer updates
  tx_func = operator.attrgetter(config.optax_name)(optax)
  opt_txs = [optax.masked(tx_func(**config.get('optax', {})), not_frozen_mask)]

  # Learning rate multipliers
  lr_mult_txs = [optax.scale(config.lr)]
  if config.get('lr_mults'):
    # Custom lr for different param sets
    masks, mults = _make_mask_trees(
        params, config.lr_mults, log_msg='config.lr_mults')
    assert all(mult > 0 for mult in mults), (
        f'Use `schedule=None` to freeze params, not `lr_mults={mults}`')
    lr_mult_txs += [
        optax.masked(optax.scale(mult), mask)
        for mask, mult in zip(masks, mults)
    ]

  # Weight decay
  # This is not gradient-based, instead uses params-side input. It is hence,
  # independent of all previous gradient txs.
  weight_decay_txs = []
  if config.get('wd'):
    # This could also be different for each param set. Defaults to applying
    # weight decay on all params with rank > 1 (should be named `kernel`)
    wd_mults = config.get('wd_mults', [('.*/kernel$', 1.0)])
    masks, mults = _make_mask_trees(params, wd_mults, log_msg='config.wd_mults')
    weight_decay_txs = [
        optax.add_decayed_weights(config.wd * mult, mask)
        for mask, mult in zip(masks, mults)
    ]

  return optax.chain(
      grad_clip_norm_tx,
      *opt_txs,
      *weight_decay_txs,
      *lr_mult_txs,
      *schedule_txs,
      optax.scale(-1.0),
  ), schedule_fns


def _make_mask_trees(params: PyTree, patterns_and_values: Sequence[Tuple[str,
                                                                         Any]],
                     log_msg: str):
  patterns, values = zip(*patterns_and_values)
  masks = tree_utils.make_mask_trees(params, patterns, log=log_msg)
  return masks, values


def _split_frozen(masks, scheds):
  """Computes `frozen_mask` for params where `sched` is `None`."""
  # Build a pytree denoting leaves that are not covered by *any* schedule.
  # We stack all *masks* and see if there's a leaf which is `False` on all masks
  all_false = jax.tree_map(lambda *bools: not any(bools), *masks)

  # Verify that there exists no leaf which is `all_false`.
  assert not any(jax.tree_util.tree_flatten(all_false)[0]), (
      'All params must be covered by scheds. Use `None` for freezing certain params.'
  )

  # Now select masks for sched=None
  frozen_masks = [mask for mask, sched in zip(masks, scheds) if sched is None]

  # Merge all frozen masks. Corner case: if frozen_mask is empty, `all_false`
  # is the frozen_mask.
  frozen_mask = jax.tree_map(lambda *bools: any(bools), *frozen_masks,
                             all_false)

  # Split the frozen and non-frozen masks and scheds.
  masks, scheds = zip(*(
      (mask, sched) for mask, sched in zip(masks, scheds) if sched is not None))
  return frozen_mask, masks, scheds
