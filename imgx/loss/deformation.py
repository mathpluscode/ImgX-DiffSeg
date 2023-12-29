"""Deformation losses for ddf."""
from functools import partial

import jax
import jax.numpy as jnp

from imgx.metric.deformation import gradient, jacobian_det


def gradient_norm_loss(x: jnp.ndarray, norm_ord: int, spacing: jnp.ndarray) -> jnp.ndarray:
    """Calculate noram of gradients of x using central finite difference.

    Args:
        x: shape = (batch, d1, d2, ..., dn, channel).
        norm_ord: 1 for L1 or 2 for L2.
        spacing: spacing between each pixel/voxel, shape = (n,).

    Returns:
         shape = (batch,).
    """
    # (batch, d1, d2, ..., dn, channel, n)
    grad = gradient(x, spacing)
    if norm_ord == 1:
        return jnp.mean(jnp.abs(grad), axis=tuple(range(1, grad.ndim)))
    if norm_ord == 2:
        return jnp.mean(grad**2, axis=tuple(range(1, grad.ndim)))
    raise ValueError(f"norm_ord = {norm_ord} is not supported.")


def bending_energy_loss(x: jnp.ndarray, spacing: jnp.ndarray) -> jnp.ndarray:
    """Calculate bending energey (L2 norm of second order gradient) using central finite difference.

    Args:
        x: shape = (batch, d1, d2, ..., dn, channel).
        spacing: spacing between each pixel/voxel, shape = (n,).

    Returns:
         shape = (batch,).
    """
    # (batch, d1, d2, ..., dn, channel, n)
    grad_1d = gradient(x, spacing)
    # (batch, d1, d2, ..., dn, channel, n, n)
    grad_2d = jax.vmap(partial(gradient, spacing=spacing), in_axes=-1, out_axes=-1)(grad_1d)
    # (batch,)
    return jnp.mean(grad_2d**2, axis=tuple(range(1, grad_2d.ndim)))


def jacobian_loss(x: jnp.ndarray, spacing: jnp.ndarray) -> jnp.ndarray:
    """Calculate Jacobian loss.

    https://arxiv.org/abs/1907.00068

    If the Jacobian determinant is <0, the transformation is folding, thus not volume preserving.
    Do not penalize the Jacobian determinant if it is >0.

    Args:
        x: shape = (batch, d1, d2, ..., dn, n).
        spacing: spacing between each pixel/voxel, shape = (n,).

    Returns:
         shape = (batch,).
    """
    # (batch, d1, d2, ..., dn)
    det = jacobian_det(x, spacing)
    det = jnp.clip(det, a_max=0.0)  # negative values are folding
    # (batch,)
    return jnp.mean(-det, axis=tuple(range(1, det.ndim)))  # reverse sign
