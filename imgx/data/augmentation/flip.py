"""Flip augmentation for image and label."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from omegaconf import DictConfig

from imgx.data.augmentation import AugmentationFn
from imgx.data.util import get_batch_size
from imgx.datasets import INFO_MAP
from imgx.datasets.constant import FOREGROUND_RANGE, IMAGE, LABEL


def random_flip(x: jnp.ndarray, to_flip: jnp.ndarray) -> jnp.ndarray:
    """Flip an array along an axis.

    Args:
        x: of shape (d1, ..., dn) or (d1, ..., dn, c)
        to_flip: (n, ), with boolean values, True means to flip along that axis.

    Returns:
        Flipped array.
    """
    for i in range(to_flip.size):
        x = lax.select(
            to_flip[i],
            jnp.flip(x, axis=i),
            x,
        )
    return x


def batch_random_flip(
    key: jax.Array, batch: dict[str, jnp.ndarray], num_spatial_dims: int, p: float
) -> dict[str, jnp.ndarray]:
    """Flip an array along an axis.

    Args:
        key: jax random key.
        batch: dict having images or labels, or foreground_range.
            images have shape (batch, d1, ..., dn) or (batch, d1, ..., dn, c)
            labels have shape (batch, d1, ..., dn)
            batch should not have other keys such as UID.
            if foreground_range exists, it's pre-calculated based on label, it's
            pre-calculated because nonzero function is not jittable.
        num_spatial_dims: number of spatial dimensions.
        p: probability to flip for each axis.

    Returns:
        Flipped batch.
    """
    batch_size = get_batch_size(batch)
    to_flip = jax.random.uniform(key=key, shape=(batch_size, num_spatial_dims)) < p

    random_flip_vmap = jax.vmap(random_flip)
    flipped_batch = {}
    for k, v in batch.items():
        if (LABEL in k) or (IMAGE in k):
            flipped_batch[k] = random_flip_vmap(x=v, to_flip=to_flip)
        elif k == FOREGROUND_RANGE:
            flipped_batch[k] = v
        else:
            raise ValueError(f"Unknown key {k} in batch.")
    return flipped_batch


def get_random_flip_augmentation_fn(config: DictConfig) -> AugmentationFn:
    """Return a data augmentation function for random flip.

    Args:
        config: entire config.

    Returns:
        A data augmentation function.
    """
    dataset_info = INFO_MAP[config.data.name]
    da_config = config.data.loader.data_augmentation
    return partial(
        batch_random_flip,
        num_spatial_dims=len(dataset_info.image_spatial_shape),
        p=da_config.p,
    )
