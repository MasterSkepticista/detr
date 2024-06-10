"""Entrypoint for training DETR in JAX."""
from absl import app, flags, logging
from clu import metric_writers
import flax
import jax
from ml_collections import config_flags
import tensorflow as tf

from train_lib import train_utils
import trainer

logging.set_verbosity('info')

config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)

flags.DEFINE_string('workdir', None, 'Path to store checkpoints and logs.')

flax.config.update('flax_use_orbax_checkpointing', False)

FLAGS = flags.FLAGS


def main(unused_argv):
  cfg = FLAGS.config
  workdir = FLAGS.workdir

  # Hide any GPUs form TensorFlow. Otherwise, TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  rng = jax.random.PRNGKey(cfg.rng_seed)
  logging.info('RNG Seed: %s', rng)
  data_rng, rng = jax.random.split(rng)

  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() > 0)

  dataset = train_utils.get_dataset(cfg, rng=data_rng)
  trainer.train_and_evaluate(
      rng=rng, dataset=dataset, config=cfg, workdir=workdir, writer=writer)


if __name__ == "__main__":
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
