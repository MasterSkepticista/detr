"""Utilities for logging, debugging or profiling."""
from collections import abc
import operator
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from absl import logging
from clu import parameter_overview
import jax
import jax.numpy as jnp

PyTree = Any


def log_param_shapes(params: PyTree, description: Optional[str] = None) -> int:
  """Logs parameter shapes and statistics.
  
  Args:
    params: PyTree of parameters.
    description: Optional, to be printed with the summary table.
  
  Returns:
    Total parameter count.
  """
  # Log only on lead host.
  total_params = jax.tree_util.tree_reduce(
      operator.add, jax.tree.map(lambda p: p.size, params))
  if jax.process_index() == 0:
    parameter_overview.log_parameter_overview(params, msg=description)
    logging.info('Total params: %d', total_params)
  return total_params


def input_spec_to_jax_shape_dtype_struct(
    spec: Union[Tuple[Tuple[int, ...], jnp.dtype], Tuple[int, ...], None],
    batch_size: Optional[int] = None) -> jax.ShapeDtypeStruct:
  """Parse an input spec into a jax.ShapeDtypeStruct."""
  spec = tuple(spec)
  if len(spec) == 2 and isinstance(spec[0], abc.Iterable):
    shape = (batch_size,) + tuple(spec[0][1:]) if batch_size else spec[0]
    dtype = spec[1]
  else:
    shape = (batch_size,) + tuple(spec[1:]) if batch_size else spec
    dtype = jnp.float32
  return jax.ShapeDtypeStruct(shape, dtype)


def compute_flops(flax_model_apply_fn: Callable[[jnp.ndarray], Any],
                  input_spec: Sequence[Union[Tuple[Tuple[int, ...], jnp.dtype],
                                             Tuple[int, ...], None]],
                  fuse_multiply_add: bool = True) -> float:
  """Performs static analysis of the graph to compute theoretical FLOPs.
  
  Args:
    flax_model_apply_fn: Apply function of the flax model to be analysed.
    input_spec: An iterable of (shape, dtype) pairs specifying the shape and
      dtype of the inputs. If unspecified, dtype is assumed to be float32.
    fuse_multiply_add: If True, count multiply-add (also known as MAC or MULACC)
      as a single FLOP instead of two. This is commonly used in literature.
  
  Returns:
    FLOP count of `flax_model_apply_fn`.
  """
  dummy_input = []
  for spec in input_spec:
    if spec is not None:
      in_st = input_spec_to_jax_shape_dtype_struct(spec, batch_size=1)
      dummy_input.append(jnp.zeros(in_st.shape, in_st.dtype))
    else:
      dummy_input.append(None)

  analysis = jax.jit(
      flax_model_apply_fn).lower(*dummy_input).compile().cost_analysis()[0]
  flops = analysis['flops']
  if fuse_multiply_add:
    flops = flops / 2.
  logging.info('GFLOPs %0.3f for input spec %s', flops / 10**9, input_spec)
  # TODO: See if we can do FLOPs calculation on CPU. Currently we compile whenever `cudnn` FA
  # SDPA API is used.
  logging.warning('FLOPs are computed based on the target device, and may not be accurate.')
  return flops
