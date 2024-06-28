"""Entrypoint for training DETR in JAX."""
import jax
import ml_collections
import trainer
from app import run
from clu import metric_writers
from train_lib import train_utils


def main(rng, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(config, rng=data_rng)
  trainer.train_and_evaluate(
      rng=rng, dataset=dataset, config=config, workdir=workdir, writer=writer)


if __name__ == "__main__":
  run(main)
