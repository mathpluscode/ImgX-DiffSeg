"""Image augmentation functions."""
from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from imgx.data import AugmentationFn


def chain_aug_fns(
    fns: Sequence[AugmentationFn],
) -> AugmentationFn:
    """Combine a list of data augmentation functions.

    Args:
        fns: entire config.

    Returns:
        A data augmentation function.
    """

    def aug_fn(key: jax.Array, batch: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        keys = jax.random.split(key, num=len(fns))
        for k, fn in zip(keys, fns):
            batch = fn(k, batch)
        return batch

    return aug_fn
