"""Deformation metrics for ddf."""
from __future__ import annotations

from jax import numpy as jnp


def gradient_along_axis(x: jnp.ndarray, axis: int, spacing: float | jnp.ndarray) -> jnp.ndarray:
    """Calculate gradients on one axis of using central finite difference.

    https://en.wikipedia.org/wiki/Finite_difference
    dx[i] = (x[i+1] - x[i-1]) / 2
    The edge values are padded.

    Args:
        x: shape = (d1, d2, ..., dn).
        axis: axis to calculate gradient.
        spacing: spacing between each pixel/voxel.

    Returns:
         shape = (d1, d2, ..., dn).
    """
    # repeat edge values
    # (d1, ..., di+2, ..., dn)
    x = jnp.pad(
        x,
        pad_width=[(0, 0)] * axis + [(1, 1)] + [(0, 0)] * (x.ndim - axis - 1),
        mode="edge",
    )
    # x[i-1]
    # (d1, ..., di, ..., dn)
    indices = jnp.arange(0, x.shape[axis] - 2)
    x_prev = jnp.take(x, indices=indices, axis=axis, indices_are_sorted=True)
    # x[i+1]
    # (d1, ..., di, ..., dn)
    indices = jnp.arange(2, x.shape[axis])
    x_next = jnp.take(x, indices=indices, axis=axis, indices_are_sorted=True)
    return (x_next - x_prev) / 2 / spacing


def gradient(x: jnp.ndarray, spacing: jnp.ndarray) -> jnp.ndarray:
    """Calculate gradients per axis of using central finite difference.

    Args:
        x: shape = (batch, d1, d2, ..., dn, channel).
        spacing: spacing between each pixel/voxel, shape = (n,).

    Returns:
         shape = (batch, d1, d2, ..., dn, channel, n).
    """
    return jnp.stack(
        [gradient_along_axis(x, axis, spacing[axis - 1]) for axis in range(1, x.ndim - 1)], axis=-1
    )


def jacobian_det(x: jnp.ndarray, spacing: jnp.ndarray) -> jnp.ndarray:
    """Calculate Jacobian matrix of ddf.

    https://arxiv.org/abs/1907.00068

    Args:
        x: shape = (batch, d1, d2, ..., dn, n).
        spacing: spacing between each pixel/voxel, shape = (n,).

    Returns:
         shape = (batch, d1, d2, ..., dn).
    """
    n = x.shape[-1]
    # (batch, d1, d2, ..., dn, n, n)
    grad = gradient(x, spacing)
    # (1, 1, ..., 1, n, n)
    eye = jnp.expand_dims(jnp.eye(n), axis=range(n + 1))
    # (batch, d1, d2, ..., dn)
    return jnp.linalg.det(grad + eye)
