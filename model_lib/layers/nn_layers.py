"""Common neural network modules."""
import flax.linen as nn
import jax.numpy as jnp


class IdentityLayer(nn.Module):
  """Creates a named placeholder for an array."""

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return x
