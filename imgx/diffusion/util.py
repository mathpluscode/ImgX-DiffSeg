"""Utility functions for diffusion models."""
from __future__ import annotations

import jax.numpy as jnp


def extract_and_expand(arr: jnp.ndarray, t_index: jnp.ndarray, ndim: int) -> jnp.ndarray:
    """Extract values from a 1D array and expand.

    This function is not jittable.

    Args:
        arr: 1D of shape (num_timesteps, ).
        t_index: storing index values < self.num_timesteps,
                shape (batch, ) or has ndim dimension.
        ndim: number of dimensions for an array of shape (batch, ...).

    Returns:
        Expanded array of shape (batch, ...), expanded axes have dim 1.
    """
    if arr.ndim != 1:
        raise ValueError(f"arr must be 1D, got {arr.ndim}D.")
    x = arr[t_index]
    return expand(x, ndim)


def expand(x: jnp.ndarray, ndim: int) -> jnp.ndarray:
    """Expand.

    This function is not jittable.

    Args:
        x: a 1D or nD array.
        ndim: number of dimensions as output.

    Returns:
        Expanded array, expanded axes have dim 1.
    """
    if x.ndim == 1:
        return jnp.expand_dims(x, axis=tuple(range(1, ndim)))
    if x.ndim == ndim:
        return x
    raise ValueError(f"t_index must be 1D or {ndim}D, got {x.ndim}D.")
