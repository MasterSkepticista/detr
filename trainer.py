import functools
from concurrent import futures
from typing import Callable

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from absl import logging
from clu import metric_writers
from flax.training.checkpoints import \
    restore_checkpoint as flax_restore_checkpoint

import bv_optax
import dataset_utils
import detr_train_utils
import utils as u
from models import detr
from train_lib import pretrain_utils, train_utils


def get_train_step(apply_fn: Callable, loss_and_metrics_fn: Callable,
                   update_batch_stats: bool, tx: optax.GradientTransformation):
  """Returns a function that runs a single step of training.
  
  Given the state of the training and a batch of data, the function computes
  the loss and updates the parameters of the model.

  Buffers of the first (train_state) argument is donated to the computation.

  Args:
    apply_fn: Flax model apply function.
    loss_and_metrics_fn: Function to calculate loss and metrics.
    update_batch_stats: bool; whether to update BN stats during training.
    tx: An `optax.GradientTransformation`
  
  Returns:
    Train step function that takes a `train_state` and `batch` and returns
    `new_train_state`, `metrics` and `predictions`.
  """

  def train_step(train_state, batch):

    def loss_fn(params):
      new_rng, rng = jax.random.split(train_state.rng)
      # Bind the rng to the host/device we are on.
      model_rng = train_utils.bind_rng_to_host_device(rng,
                                                      axis_name='batch',
                                                      bind_to='device')
      variables = {'params': params, **train_state.model_state}
      predictions, new_model_state = apply_fn(
          variables,
          batch['inputs'],
          padding_mask=batch['padding_mask'],
          update_batch_stats=update_batch_stats,
          mutable=train_state.model_state.keys(),
          train=True,
          rngs={'dropout': model_rng})
      loss, metrics = loss_and_metrics_fn(predictions,
                                          batch,
                                          model_params=params)
      return loss, (new_model_state, metrics, predictions, new_rng)

    new_global_step = train_state.global_step + 1
    (_, (new_model_state, metrics, predictions,
         new_rng)), grads = jax.value_and_grad(loss_fn,
                                               has_aux=True)(train_state.params)

    grads = jax.tree_map(lambda g: jnp.asarray(g, jnp.bfloat16), grads)
    grads = jax.lax.pmean(grads, axis_name='batch')

    updates, new_opt_state = tx.update(grads,
                                       train_state.opt_state,
                                       params=train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)
    train_state = train_state.replace(global_step=new_global_step,
                                      params=new_params,
                                      opt_state=new_opt_state,
                                      model_state=new_model_state,
                                      rng=new_rng)
    return train_state, metrics, predictions

  return train_step


def get_eval_step(flax_model,
                  loss_and_metrics_fn,
                  logits_to_probs_fn,
                  metrics_only=False,
                  debug=False):
  """Runs a single step of evaluation.
  
  Note that in this code, the buffer of the second argument (batch) is donated
  to the computation.

  Args:
    flax_model: Flax model (an instance of nn.Module).
    loss_and_metrics_fn: A function that given model predictions, a batch and 
      parameters of the model calculates the loss as well as metrics.
    logits_to_probs_fn: Function that takes logits and converts them to probs.
    metrics_only: bool; Only return metrics.
    debug: bool; Whether the debug mode is enabled during evaluation.
      `debug=True` enables model specific logging/storing some values using
      jax.host_callback.
  
  Returns:
    Eval step function which returns predictions and calculated metrics.
  """

  def metrics_fn(train_state, batch, predictions):
    _, metrics = loss_and_metrics_fn(predictions,
                                     batch,
                                     model_params=train_state.params)
    if metrics_only:
      return None, None, metrics

    pred_probs = logits_to_probs_fn(predictions['pred_logits'])
    # Collect necessary predictions and target information from all hosts.
    predictions_out = {
        'pred_probs': pred_probs,
        'pred_logits': predictions['pred_logits'],
        'pred_boxes': predictions['pred_boxes']
    }
    labels = {
        'image/id': batch['label']['image/id'],
        'size': batch['label']['size'],
        'orig_size': batch['label']['orig_size']
    }
    to_copy = [
        'labels', 'boxes', 'not_exhaustive_category_ids', 'neg_category_ids'
    ]
    for name in to_copy:
      if name in batch['label']:
        labels[name] = batch['label'][name]

    targets = {'label': labels, 'batch_mask': batch['batch_mask']}

    predictions_out = jax.lax.all_gather(predictions_out, 'batch')
    targets = jax.lax.all_gather(targets, 'batch')
    return targets, predictions_out, metrics

  def eval_step(train_state, batch):
    variables = {'params': train_state.params, **train_state.model_state}
    predictions = flax_model.apply(variables,
                                   batch['inputs'],
                                   padding_mask=batch['padding_mask'],
                                   train=False,
                                   mutable=False,
                                   debug=debug)
    return metrics_fn(train_state, batch, predictions)

  return eval_step


def train_and_evaluate(*, rng: jnp.ndarray, dataset: dataset_utils.Dataset,
                       config: ml_collections.ConfigDict, workdir: str,
                       writer: metric_writers.MetricWriter):
  lead_host = jax.process_index() == 0

  def info(s, *a):
    if lead_host:
      logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)

  # This pool is used for async I/O operations like logging metrics
  pool = futures.ThreadPoolExecutor(max_workers=2)

  # Calculate total train steps using available information.
  ntrain_img = dataset.meta_data['num_train_examples']
  total_steps = u.steps('total',
                        config,
                        data_size=ntrain_img,
                        batch_size=config.batch_size)
  info('Running for %d steps (%f epochs)', total_steps,
       total_steps / (ntrain_img / config.batch_size))

  # Initialize model, loss_fn
  model = detr.DETRModel(config, dataset.meta_data)
  rng, init_rng = jax.random.split(rng)
  (params, model_state, *_) = train_utils.initialize_model(
      model=model.flax_model,
      input_spec=[(dataset.meta_data['input_shape'],
                   dataset.meta_data.get('input_dtype', jnp.float32))],
      config=config,
      rngs=init_rng)

  # Create optimizer.
  tx, sched_fns = bv_optax.make(config,
                                params,
                                sched_kw=dict(total_steps=total_steps,
                                              batch_size=config.batch_size,
                                              data_size=ntrain_img))
  opt_state = jax.jit(tx.init, backend='cpu')(params)
  sched_fns_cpu = [jax.jit(sched_fn, backend='cpu') for sched_fn in sched_fns]

  # Build TrainState
  rng, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(global_step=0,
                                       params=params,
                                       opt_state=opt_state,
                                       model_state=model_state,
                                       rng=train_rng)

  # Load checkpoint/pretrained weights
  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state)

  if (start_step == 0  # Which means no checkpoint was restored.
      and config.get('init_from') is not None):
    raise NotImplementedError(
        'Init from pretrained checkpoint is not supported.')
  elif start_step == 0 and config.get('load_pretrained_backbone', False):
    # Only load pretrained backbone if we are at the beginning of training.
    bb_ckpt_path = config.pretrained_backbone_configs.get('checkpoint_path')
    bb_train_state = flax_restore_checkpoint(bb_ckpt_path, target=None)
    train_state = pretrain_utils.init_from_pretrain_state(
        train_state, bb_train_state, model_prefix_path=['backbone'])

  # Calculate total number of training steps.
  steps_per_epoch = ntrain_img // config.batch_size
  update_batch_stats = not config.get('freeze_backbone_batch_stats', False)
  if not update_batch_stats:
    if not config.load_pretrained_backbone:
      raise ValueError(
          'Freezing backbone stats without a pretrained backbone '
          'does not make rational sense. Please check your config.')

  # Replicate.
  train_state = flax.jax_utils.replicate(train_state)
  del params

  train_step = get_train_step(
      apply_fn=model.flax_model.apply,
      loss_and_metrics_fn=model.loss_function,  # TODO
      update_batch_stats=update_batch_stats,
      tx=tx)
  train_step_pmapped = jax.pmap(train_step,
                                axis_name='batch',
                                donate_argnums=(0,))

  # Evaluation code.
  eval_step = get_eval_step(flax_model=model.flax_model,
                            loss_and_metrics_fn=model.loss_function,
                            logits_to_probs_fn=model.logits_to_probs,
                            debug=config.debug_eval)
  eval_step_pmapped = jax.pmap(eval_step,
                               axis_name='batch',
                               donate_argnums=(1,))

  # Ceil rounding such that we include the last incomplete batch.
  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  metrics_normalizer_fn = functools.partial(
      detr_train_utils.normalize_metrics_summary,
      object_detection_loss_keys=model.loss_terms_weights.keys())

  def evaluate(train_state, step):
    """Runs evaluation code."""
    pass

  #####################################################

  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  log_summary_steps = config.get('log_summary_steps', 25)
  log_large_summary_steps = config.get('log_large_summary_steps', 0)
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps

  global_metrics_evaluator = None  # Only run eval on the lead host.
  if lead_host:
    global_metrics_evaluator = detr_train_utils.DetrGlobalEvaluator(
        config.dataset_name, annotations_loc=config.annotations_loc)

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None

  info('Starting training loop at step %d.', start_step + 1)
  report_progress = None
