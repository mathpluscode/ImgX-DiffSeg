"""Diffusion based segmentation evaluation."""
from __future__ import annotations

from functools import partial

import jax
from jax import numpy as jnp

from imgx.diffusion import DiffusionSegmentation
from imgx.metric.util import (
    get_jit_segmentation_confidence,
    get_jit_segmentation_metrics,
    get_non_jit_segmentation_metrics_per_step,
)
from imgx_datasets.dataset_info import DatasetInfo


def batch_diffusion_segmentation_inference(
    image: jnp.ndarray,
    num_classes: int,
    sd: DiffusionSegmentation,
    self_conditioning: bool,
) -> jnp.ndarray:
    """Perform diffusion segmentation model inference.

    Args:
        image: (batch, ..., channel).
        num_classes: number of classes including background.
        sd: segmentation diffusion model.
        self_conditioning: whether to use self conditioning.

    Returns:
        logits, of shape (batch, ..., num_classes, num_timesteps_sample).
    """
    # noise_shape = (batch, ..., num_classes)
    noise_shape = (*image.shape[:-1], num_classes)
    # (batch, ..., num_classes)
    x_t = sd.sample_noise(shape=noise_shape, dtype=image.dtype)
    # (batch, ..., num_classes, num_timesteps_sample)
    return jnp.stack(
        list(
            sd.sample_logits_progressive(
                image=image, x_t=x_t, self_conditioning=self_conditioning
            )
        ),
        axis=-1,
    )


def batch_diffusion_segmentation_evaluation(
    logits: jnp.ndarray,
    label: jnp.ndarray,
    dataset_info: DatasetInfo,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Evaluate predictions from diffusion model.

    Args:
        logits: (batch, ..., num_classes, num_timesteps_sample).
        label: (batch, ...).
        spacing: spacing of pixel/voxels along each dimension.
        dataset_info: dataset information.

    Returns:
        Predicted label, with potential post-processing.
        Metrics, each metric value has shape (batch, num_timesteps_sample).
    """
    # (batch, ..., num_classes)
    mask_true = dataset_info.label_to_mask(label, axis=-1)
    # (batch, ..., num_classes, num_timesteps_sample)
    label_pred = dataset_info.logits_to_label_with_post_process(logits, axis=-2)
    mask_pred = dataset_info.label_to_mask(label_pred, axis=-2)

    spacing = jnp.array(dataset_info.image_spacing)
    scalars_confidence_jit = jax.vmap(
        get_jit_segmentation_confidence,
        in_axes=-1,
        out_axes=-1,
    )(logits)
    scalars_jit = jax.vmap(
        partial(
            get_jit_segmentation_metrics,
            mask_true=mask_true,
            spacing=spacing,
        ),
        in_axes=-1,
        out_axes=-1,
    )(mask_pred)
    scalars_non_jit = get_non_jit_segmentation_metrics_per_step(
        mask_pred=mask_pred,
        mask_true=mask_true,
        spacing=spacing,
    )
    return label_pred, {
        **scalars_confidence_jit,
        **scalars_jit,
        **scalars_non_jit,
    }
