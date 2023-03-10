"""Metric functions for image segmentation."""
import jax.numpy as jnp


def dice_score(
    mask_pred: jnp.ndarray,
    mask_true: jnp.ndarray,
) -> jnp.ndarray:
    """Soft Dice score, larger is better.

    Args:
        mask_pred: soft mask with probabilities, (batch, ..., num_classes).
        mask_true: one hot targets, (batch, ..., num_classes).

    Returns:
        Dice score of shape (batch, num_classes).
    """
    reduce_axis = tuple(range(mask_pred.ndim))[1:-1]
    numerator = 2.0 * jnp.sum(mask_pred * mask_true, axis=reduce_axis)
    denominator = jnp.sum(mask_pred + mask_true, axis=reduce_axis)
    return jnp.where(
        condition=denominator > 0,
        x=numerator / denominator,
        y=jnp.nan,
    )


def iou(
    mask_pred: jnp.ndarray,
    mask_true: jnp.ndarray,
) -> jnp.ndarray:
    """IOU (Intersection Over Union), or Jaccard index.

    Args:
        mask_pred: binary mask of predictions, (batch, ..., num_classes).
        mask_true: one hot targets, (batch, ..., num_classes).

    Returns:
        IoU of shape (batch, num_classes).
    """
    reduce_axis = tuple(range(mask_pred.ndim))[1:-1]
    numerator = jnp.sum(mask_pred * mask_true, axis=reduce_axis)
    sum_mask = jnp.sum(mask_pred + mask_true, axis=reduce_axis)
    denominator = sum_mask - numerator
    return jnp.where(
        condition=sum_mask > 0, x=numerator / denominator, y=jnp.nan
    )
