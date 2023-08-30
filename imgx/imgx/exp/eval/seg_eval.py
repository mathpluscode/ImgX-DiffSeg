"""Segmentation evaluation."""
from __future__ import annotations

import haiku as hk
from jax import numpy as jnp

from imgx.metric.util import (
    get_jit_segmentation_confidence,
    get_jit_segmentation_metrics,
    get_non_jit_segmentation_metrics,
)
from imgx_datasets.dataset_info import DatasetInfo


def batch_segmentation_inference(
    image: jnp.ndarray,
    model: hk.Module,
) -> jnp.ndarray:
    """Perform segmentation model inference.

    Args:
        image: (batch, ..., ch).
        model: network instance.

    Returns:
        logits, of shape (batch, ..., num_classes).
    """
    # (batch, ..., num_classes)
    logits_maybe_list = model(image=image)
    return logits_maybe_list


def batch_segmentation_evaluation(
    logits: jnp.ndarray,
    label: jnp.ndarray,
    dataset_info: DatasetInfo,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Evaluate binary predictions.

    Args:
        logits: (batch, ..., num_classes).
        label: (batch, ...).
        dataset_info: dataset information.

    Returns:
        Predicted label, with potential post-processing.
        Metrics, each metric value has shape (batch, ).
    """
    # (batch, ..., num_classes)
    mask_true = dataset_info.label_to_mask(label, axis=-1)
    label_pred = dataset_info.logits_to_label_with_post_process(logits, axis=-1)
    mask_pred = dataset_info.label_to_mask(label_pred, axis=-1)

    spacing = jnp.array(dataset_info.image_spacing)
    scalars_confidence_jit = get_jit_segmentation_confidence(logits)
    scalars_jit = get_jit_segmentation_metrics(
        mask_pred=mask_pred, mask_true=mask_true, spacing=spacing
    )
    scalars_non_jit = get_non_jit_segmentation_metrics(
        mask_pred=mask_pred,
        mask_true=mask_true,
        spacing=spacing,
    )
    return label_pred, {
        **scalars_confidence_jit,
        **scalars_jit,
        **scalars_non_jit,
    }
