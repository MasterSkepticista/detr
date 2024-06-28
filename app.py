"""Entrypoint.

This script performs device and experiment-related initializations, and then
calls the provided main with a PRNGKey, the config, workdir and a metric writer.

Each project here would have its own `main.py`.

Usage in main.py:
  from app import run

  def main(rng: jnp.ndarray,
           config: ml_collections.ConfigDict,
           workdir: str,
           writer: metric_writers.MetricWriter):
    # Train/Eval
  
  if __name__ == "__main__":
    run(main)
"""
import functools
import os

import flax
import jax
import tensorflow as tf
from absl import app, flags, logging
from clu import metric_writers
from ml_collections import config_flags

logging.set_verbosity("info")
FLAGS = flags.FLAGS

# These are general flags that are used across most of orion projects. These
# flags can be accessed via `flags.FLAGS.<flag_name>` and projects can also
# define their own flags in their `main.py`.
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work Unit Directory.")
flags.mark_flags_as_required(["config", "workdir"])

jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")
flax.config.update('flax_use_orbax_checkpointing', False)

def run(main):
  app.run(functools.partial(_run_main, main=main))

def _run_main(argv, *, main):
  """Runs `main` after some initial setup."""
  del argv  # unused
  # Hide GPUs from TF.
  tf.config.set_visible_devices([], "GPU")

  # Initialize JAX distributed if in an MPI env.
  if int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1)) > 1:
    jax.distributed.initialize()

  logging.info(
      f"\u001b[33mHello from process {jax.process_index()} holding "
      f"{jax.local_device_count()}/{jax.device_count()} devices.\u001b[0m")

  rng = jax.random.PRNGKey(FLAGS.config.rng_seed)
  logging.info("RNG: %s", rng)

  writer = metric_writers.AsyncWriter(
      metric_writers.SummaryWriter(logdir=FLAGS.workdir))
  main(rng=rng, config=FLAGS.config, workdir=FLAGS.workdir, writer=writer)