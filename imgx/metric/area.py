"""Metrics to measure foreground area."""

import jax.numpy as jnp


def class_proportion(mask: jnp.ndarray) -> jnp.ndarray:
    """Calculate proportion per class.

    Args:
        mask: shape = (batch, d1, ..., dn, num_classes).

    Returns:
        Proportion, shape = (batch, num_classes).
    """
    reduce_axes = tuple(range(1, mask.ndim - 1))
    volume = jnp.float32(jnp.prod(jnp.array(mask.shape[1:-1])))
    sqrt_volume = jnp.sqrt(volume)
    mask = jnp.float32(mask)
    # attempt to avoid over/underflow
    return jnp.sum(mask / sqrt_volume, axis=reduce_axes) / sqrt_volume
