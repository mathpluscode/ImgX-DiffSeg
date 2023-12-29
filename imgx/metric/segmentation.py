"""Segmentation related metrics."""
from __future__ import annotations

from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np

from imgx.data.warp import get_coordinate_grid
from imgx.datasets import DatasetInfo
from imgx.metric import (
    aggregated_surface_distance,
    centroid_distance,
    class_proportion,
    class_volume,
    dice_score,
    iou,
    normalized_surface_dice_from_distances,
    stability,
)
from imgx.metric.util import flatten_diffusion_metrics


def get_jit_segmentation_confidence(
    logits: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Calculate segmentation confidence metrics.

    Use nanmean in case some classes do not exist.

    Args:
        logits: (batch, ..., num_classes).

    Returns:
        Dict of metrics, each value is of shape (batch,).
    """
    scalars = {}

    # (batch, num_classes)
    stability_bc = stability(logits)
    for i in range(stability_bc.shape[-1]):
        scalars[f"stability_class_{i}"] = stability_bc[:, i]
    scalars["mean_stability"] = jnp.nanmean(stability_bc, axis=1)
    scalars["mean_stability_without_background"] = jnp.nanmean(stability_bc[:, 1:], axis=1)

    return scalars


def get_jit_segmentation_metrics(
    mask_pred: jnp.ndarray, mask_true: jnp.ndarray, spacing: jnp.ndarray
) -> dict[str, jnp.ndarray]:
    """Calculate segmentation metrics.

    Use nanmean in case some classes do not exist.

    Args:
        mask_true: shape = (batch, ..., num_classes).
        mask_pred: shape = (batch, ..., num_classes).
        spacing: spacing of pixel/voxels along each dimension, (3,).

    Returns:
        Dict of metrics, each value is of shape (batch,).
    """
    chex.assert_equal_shape([mask_pred, mask_true])
    scalars = {}
    # binary dice (batch, num_classes)
    dice_score_bc = dice_score(
        mask_pred=mask_pred,
        mask_true=mask_true,
    )
    for i in range(dice_score_bc.shape[-1]):
        scalars[f"binary_dice_score_class_{i}"] = dice_score_bc[:, i]
    scalars["mean_binary_dice_score"] = jnp.nanmean(dice_score_bc, axis=1)
    scalars["mean_binary_dice_score_without_background"] = jnp.nanmean(dice_score_bc[:, 1:], axis=1)

    # IoU (batch, num_classes)
    iou_bc = iou(
        mask_pred=mask_pred,
        mask_true=mask_true,
    )
    for i in range(iou_bc.shape[-1]):
        scalars[f"iou_class_{i}"] = iou_bc[:, i]
    scalars["mean_iou"] = jnp.nanmean(iou_bc, axis=1)
    scalars["mean_iou_without_background"] = jnp.nanmean(iou_bc[:, 1:], axis=1)

    # centroid distance (batch, num_classes)
    grid = get_coordinate_grid(shape=mask_pred.shape[1:-1])
    centroid_dist_bc = centroid_distance(
        mask_pred=mask_pred,
        mask_true=mask_true,
        grid=grid,
        spacing=spacing,
    )
    for i in range(centroid_dist_bc.shape[-1]):
        scalars[f"centroid_dist_class_{i}"] = centroid_dist_bc[:, i]
    scalars["mean_centroid_dist"] = jnp.nanmean(centroid_dist_bc, axis=1)
    scalars["mean_centroid_dist_without_background"] = jnp.nanmean(centroid_dist_bc[:, 1:], axis=1)

    # class proportion without considering spacing (batch, num_classes)
    for mask, mask_name in zip([mask_pred, mask_true], ["pred", "label"]):
        class_prop_bc = class_proportion(mask)
        for i in range(class_prop_bc.shape[-1]):
            scalars[f"class_{i}_proportion_{mask_name}"] = class_prop_bc[:, i]

    # class volume considering spacing (batch, num_classes)
    for mask, mask_name in zip([mask_pred, mask_true], ["pred", "label"]):
        class_volume_bc = class_volume(mask, spacing)
        for i in range(class_volume_bc.shape[-1]):
            scalars[f"class_{i}_volume_{mask_name}"] = class_volume_bc[:, i]

    return scalars


def get_non_jit_segmentation_metrics(
    mask_pred: jnp.ndarray,
    mask_true: jnp.ndarray,
    spacing: jnp.ndarray | None,
) -> dict[str, jnp.ndarray]:
    """Calculate non-jittable segmentation metrics for batch.

    Use nanmean in case some classes do not exist.

    Args:
        mask_pred: (batch, w, h, d, num_classes)
        mask_true: (batch, w, h, d, num_classes)
        spacing: spacing of pixel/voxels along each dimension.

    Returns:
        Dict of metrics, each value is of shape (batch,).
    """
    chex.assert_equal_shape([mask_pred, mask_true])
    batch_scalars = {}

    # (3, batch, num_classes)
    # mean surface distance
    # hausdorff distance, 95 percentile
    # normalised surface dice
    sur_dist_bc = aggregated_surface_distance(
        mask_pred=np.array(mask_pred),
        mask_true=np.array(mask_true),
        agg_fns=[
            np.mean,
            partial(np.percentile, q=95),
            normalized_surface_dice_from_distances,
        ],
        num_args=[1, 1, 2],
        spacing=spacing,
    )
    for i in range(sur_dist_bc.shape[-1]):
        batch_scalars[f"mean_surface_dist_class_{i}"] = sur_dist_bc[0, :, i]
        batch_scalars[f"hausdorff_dist_class_{i}"] = sur_dist_bc[1, :, i]
        batch_scalars[f"normalised_surface_dice_class_{i}"] = sur_dist_bc[2, :, i]
    batch_scalars["mean_mean_surface_dist"] = np.nanmean(sur_dist_bc[0, ...], axis=-1)
    batch_scalars["mean_hausdorff_dist"] = np.nanmean(sur_dist_bc[1, ...], axis=-1)
    batch_scalars["mean_normalised_surface_dice"] = np.nanmean(sur_dist_bc[2, ...], axis=-1)
    batch_scalars["mean_mean_surface_dist_without_background"] = np.nanmean(
        sur_dist_bc[0, :, 1:], axis=-1
    )
    batch_scalars["mean_hausdorff_dist_without_background"] = np.nanmean(
        sur_dist_bc[1, :, 1:], axis=-1
    )
    batch_scalars["mean_normalised_surface_dice_without_background"] = np.nanmean(
        sur_dist_bc[2, :, 1:], axis=-1
    )
    return batch_scalars


def get_non_jit_segmentation_metrics_per_step(
    mask_pred: jnp.ndarray,
    mask_true: jnp.ndarray,
    spacing: jnp.ndarray | None,
) -> dict[str, np.ndarray]:
    """Calculate non-jittable segmentation metrics for batch.

    Cannot use VMAP as it requires jittable functions.

    Args:
        mask_pred: (batch, *spatial_shape, num_classes, num_steps)
        mask_true: (batch, *spatial_shape, num_classes)
        spacing: spacing of pixel/voxels along each dimension.

    Returns:
        Metrics dict, each value of shape (batch, num_steps).
    """
    if mask_pred.ndim != mask_true.ndim + 1:
        raise ValueError("mask_pred must have one more dimension than mask_true")

    lst_scalars = []
    for i in range(mask_pred.shape[-1]):
        scalars = get_non_jit_segmentation_metrics(
            mask_pred=mask_pred[..., i],
            mask_true=mask_true,
            spacing=spacing,
        )
        lst_scalars.append(scalars)
    scalar_keys = lst_scalars[0].keys()
    scalars = {}
    for k in scalar_keys:
        scalars[k] = np.stack([x[k] for x in lst_scalars], axis=-1)
    return scalars


def get_segmentation_metrics(
    logits: jnp.ndarray | None,
    label_pred: jnp.ndarray | None,
    label_true: jnp.ndarray,
    dataset_info: DatasetInfo,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Calculate segmentation metrics.

    This function is not jittable.

    Args:
        logits: (batch, ..., num_classes).
        label_pred: (batch, ...).
        label_true: (batch, ...).
        dataset_info: dataset information.

    Returns:
        Metrics, each metric value has shape (batch, ).
        Predicted label, with potential post-processing.
    """
    if (logits is None) == (label_pred is None):
        raise ValueError("Either logits or label_pred must be None.")

    # (batch, ..., num_classes)
    mask_true = dataset_info.label_to_mask(label_true, axis=-1)
    # post process is non-jiitable and maybe time-consuming
    if label_pred is None:
        label_pred = dataset_info.logits_to_label_with_post_process(logits, axis=-1)
    else:
        label_pred = dataset_info.post_process_label(label_pred)
    mask_pred = dataset_info.label_to_mask(label_pred, axis=-1)

    spacing = jnp.array(dataset_info.image_spacing)
    if logits is None:
        metrics_confidence_jit = {}
    else:
        metrics_confidence_jit = jax.jit(get_jit_segmentation_confidence)(logits)
    metrics_jit = jax.jit(get_jit_segmentation_metrics)(
        mask_pred=mask_pred, mask_true=mask_true, spacing=spacing
    )
    metrics_non_jit = get_non_jit_segmentation_metrics(
        mask_pred=mask_pred,
        mask_true=mask_true,
        spacing=spacing,
    )
    metrics = {
        **metrics_confidence_jit,
        **metrics_jit,
        **metrics_non_jit,
    }
    return metrics, label_pred


def get_segmentation_metrics_per_step(
    logits: jnp.ndarray,
    label: jnp.ndarray,
    dataset_info: DatasetInfo,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Calculate segmentation metrics for diffusion.

    This function is not jittable.

    Args:
        logits: (batch, ..., num_classes, num_timesteps).
        label: (batch, ...).
        dataset_info: dataset information.

    Returns:
        Metrics, each metric value has shape (batch, ).
        Predicted label, with potential post-processing, (batch, ..., num_timesteps).
    """
    # (batch, ..., num_classes)
    mask_true = dataset_info.label_to_mask(label, axis=-1)
    # post process is non-jiitable and maybe time-consuming
    # (batch, ..., num_classes, num_timesteps)
    label_pred = dataset_info.logits_to_label_with_post_process(logits, axis=-2)
    mask_pred = dataset_info.label_to_mask(label_pred, axis=-2)

    spacing = jnp.array(dataset_info.image_spacing)
    metrics_confidence_jit = jax.vmap(
        get_jit_segmentation_confidence,
        in_axes=-1,
        out_axes=-1,
    )(logits)
    metrics_jit = jax.vmap(
        partial(
            get_jit_segmentation_metrics,
            mask_true=mask_true,
            spacing=spacing,
        ),
        in_axes=-1,
        out_axes=-1,
    )(mask_pred)
    metrics_non_jit = get_non_jit_segmentation_metrics_per_step(
        mask_pred=mask_pred,
        mask_true=mask_true,
        spacing=spacing,
    )
    # (batch, num_timesteps_sample) per metric
    metrics = {
        **metrics_confidence_jit,
        **metrics_jit,
        **metrics_non_jit,
    }
    # separate metrics per step
    metrics = flatten_diffusion_metrics(metrics)
    return metrics, label_pred
