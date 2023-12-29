"""Intensity related data augmentation functions."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from imgx.data.augmentation import AugmentationFn
from imgx.data.util import get_batch_size
from imgx.datasets.constant import IMAGE


def adjust_gamma(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    gain: float = 1.0,
) -> jnp.ndarray:
    """Adjust gamma of input images.

    https://github.com/tensorflow/tensorflow/blob/v2.14.0/tensorflow/python/ops/image_ops_impl.py#L2303-L2366

    Args:
        x: input image.
        gamma: non-negative real number.
        gain: the constant multiplier.

    Returns:
        Adjusted image.
    """
    return gain * x**gamma


def batch_random_adjust_gamma(
    key: jax.Array,
    batch: dict[str, jnp.ndarray],
    max_log_gamma: float,
    p: float = 0.5,
) -> dict[str, jnp.ndarray]:
    """Perform random gamma adjustment on images in batch.

    https://torchio.readthedocs.io/_modules/torchio/transforms/augmentation/intensity/random_gamma.html

    Args:
        key: jax random key.
        batch: dict having images or labels, or foreground_range.
        max_log_gamma: maximum log gamma.
        p: probability of performing the augmentation.

    Returns:
        Augmented dict having image and label, shapes are not changed.
    """
    batch_size = get_batch_size(batch)

    adjusted_batch = {}
    for k, v in batch.items():
        if IMAGE in k:
            key_gamma, key_act, key = jax.random.split(key, 3)
            log_gamma = jax.random.uniform(
                key=key_gamma,
                shape=(batch_size,),
                minval=-max_log_gamma,
                maxval=max_log_gamma,
            )
            gamma = jnp.exp(log_gamma)
            gamma = jnp.where(
                jax.random.uniform(key=key_act, shape=gamma.shape) < p,
                gamma,
                jnp.ones_like(gamma),
            )
            adjusted_batch[k] = jax.vmap(adjust_gamma)(v, gamma)
        else:
            adjusted_batch[k] = v
    return adjusted_batch


def get_random_gamma_augmentation_fn(config: DictConfig) -> AugmentationFn:
    """Return a data augmentation function for random gamma transformation.

    Args:
        config: entire config.

    Returns:
        A data augmentation function.
    """
    da_config = config.data.loader.data_augmentation
    return partial(
        batch_random_adjust_gamma,
        max_log_gamma=da_config.max_log_gamma,
        p=da_config.p,
    )


def rescale_intensity(
    x: jnp.ndarray,
    v_min: float = 0.0,
    v_max: float = 1.0,
) -> jnp.ndarray:
    """Adjust intensity linearly to the desired range.

    Args:
        x: input image, (batch, *spatial_dims, channel).
        v_min: minimum intensity.
        v_max: maximum intensity.

    Returns:
        Adjusted image.
    """
    reduction_axes = tuple(range(x.ndim)[slice(1, -1)])
    x_min = jnp.min(x, axis=reduction_axes, keepdims=True)
    x_max = jnp.max(x, axis=reduction_axes, keepdims=True)
    x = (x - x_min) / (x_max - x_min)
    x = x * (v_max - v_min) + v_min
    return x


def batch_rescale_intensity(
    key: jax.Array,  # noqa: ARG001, pylint: disable=unused-argument
    batch: dict[str, jnp.ndarray],
    v_min: float = 0.0,
    v_max: float = 1.0,
) -> dict[str, jnp.ndarray]:
    """Perform intensity scaling on images in batch.

    Args:
        key: jax random key.
        batch: dict having images or labels, or foreground_range.
        v_min: minimum intensity.
        v_max: maximum intensity.

    Returns:
        Augmented dict having image and label, shapes are not changed.
    """
    adjusted_batch = {}
    for k, v in batch.items():
        if IMAGE in k:
            adjusted_batch[k] = rescale_intensity(v, v_min=v_min, v_max=v_max)
        else:
            adjusted_batch[k] = v
    return adjusted_batch


def get_rescale_intensity_fn(config: DictConfig) -> AugmentationFn:
    """Return a data augmentation function for intensity scaling.

    Args:
        config: entire config.

    Returns:
        A data augmentation function.
    """
    da_config = config.data.loader.data_augmentation
    return partial(
        batch_rescale_intensity,
        v_min=da_config.v_min,
        v_max=da_config.v_max,
    )
