"""Metrics to measure foreground area."""
import jax
import numpy as np
from jax import numpy as jnp

MM3_TO_ML = 0.001


def class_proportion(mask: jnp.ndarray) -> jnp.ndarray:
    """Calculate proportion per class.

    This metric does not consider spacing.

    Args:
        mask: shape = (batch, d1, ..., dn, num_classes).

    Returns:
        Proportion, shape = (batch, num_classes).
    """
    reduce_axes = tuple(range(1, mask.ndim - 1))
    volume = jnp.float32(np.prod(mask.shape[1:-1]))
    sqrt_volume = jnp.sqrt(volume)
    mask = jnp.float32(mask)
    # attempt to avoid over/underflow
    return jnp.sum(mask / sqrt_volume, axis=reduce_axes) / sqrt_volume


def get_volume(label: jnp.ndarray, spacing: jnp.ndarray) -> jnp.ndarray:
    """Calculate volume from binary label/mask.

    This metric considers spacing.

    Args:
        label: binary label, of shape (batch, d1, ..., dn).
        spacing: (n,), spacing of each dimension, in mm.

    Returns:
        volume: volume in ml, (batch,).
    """
    volume_per_voxel = jnp.prod(spacing) * MM3_TO_ML  # ml
    return jnp.sum(label, axis=list(range(1, label.ndim))) * volume_per_voxel


def class_volume(mask: jnp.ndarray, spacing: jnp.ndarray) -> jnp.ndarray:
    """Calculate volume per class.

    This metric does consider spacing.

    Args:
        mask: shape = (batch, d1, ..., dn, num_classes).
        spacing: (n,), spacing of each dimension, in mm.

    Returns:
        volume: volume in ml, (batch, num_classes).
    """
    return jax.vmap(get_volume, in_axes=(-1, None), out_axes=-1)(mask, spacing)
