"""Functions for slicing images."""
from __future__ import annotations

from jax import numpy as jnp


def merge_spatial_dim_into_batch(
    x: jnp.ndarray, num_spatial_dims: int
) -> jnp.ndarray:
    """Merge spatial dimensions into batch dimension.

    Args:
        x: array with original shape (batch, ..., in_channels).
        num_spatial_dims: target number of spatial dimensions.

    Returns:
        array with ndim=num_spatial_dims+2,
        shape = (extended_batch, ..., in_channels).
    """
    # e.g. if x.shape = (batch, h, w, d, in_channels)
    # then x.ndim == 5, num_spatial_dims = 2
    # axes = (0, 3, 1, 2, 4)
    axes = (
        0,
        *range(num_spatial_dims + 1, x.ndim - 1),
        *range(1, num_spatial_dims + 1),
        x.ndim - 1,
    )
    # move extra dims to front
    # e.g. (batch, h, w, d, in_channels) -> (batch, d, h, w, in_channels)
    x = jnp.transpose(x, axes)
    # e.g. (batch, d, h, w, in_channels) -> (batch*d, h, w, in_channels)
    return jnp.reshape(x, (-1, *x.shape[x.ndim - num_spatial_dims - 1 :]))


def split_spatial_dim_from_batch(
    x: jnp.ndarray,
    num_spatial_dims: int,
    batch_size: int,
    spatial_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Remove spatial dimensions from batch axis.

    Args:
        x: array with merged shape (batch, ..., in_channels),
            x.ndim=num_spatial_dims+2.
        num_spatial_dims: current number of spatial dimensions.
        batch_size: batch size.
        spatial_shape: original spatial shape.

    Returns:
        array with original shape (batch, ..., in_channels).
    """
    # e.g. (batch*d, h, w, out_channels) -> (batch, d, h, w, out_channels)
    x = jnp.reshape(
        x, (batch_size, *spatial_shape[num_spatial_dims:], *x.shape[1:])
    )

    # e.g. if x.shape = (batch, d, h, w, out_channels)
    # then x.ndim == 5, num_spatial_dims = 2
    # axes = (0, 3, 1, 2, 4)
    axes = (
        0,
        *range(x.ndim - 1 - num_spatial_dims, x.ndim - 1),
        *range(1, x.ndim - 1 - num_spatial_dims),
        x.ndim - 1,
    )
    # e.g. (batch, d, h, w, out_channels) -> (batch, h, w, d, out_channels)
    return jnp.transpose(x, axes)
