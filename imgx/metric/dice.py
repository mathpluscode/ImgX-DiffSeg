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
    # additions between bools results in errors
    if mask_pred.dtype == jnp.bool_:
        mask_pred = mask_pred.astype(jnp.float32)
    if mask_true.dtype == jnp.bool_:
        mask_true = mask_true.astype(jnp.float32)

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
    # additions between bools results in errors
    if mask_pred.dtype == jnp.bool_:
        mask_pred = mask_pred.astype(jnp.float32)
    if mask_true.dtype == jnp.bool_:
        mask_true = mask_true.astype(jnp.float32)

    reduce_axis = tuple(range(mask_pred.ndim))[1:-1]
    numerator = jnp.sum(mask_pred * mask_true, axis=reduce_axis)
    sum_mask = jnp.sum(mask_pred + mask_true, axis=reduce_axis)
    denominator = sum_mask - numerator
    return jnp.where(condition=sum_mask > 0, x=numerator / denominator, y=jnp.nan)


def stability(
    logits: jnp.ndarray,
    threshold: float = 0.0,
    threshold_offset: float = 1.0,
) -> jnp.ndarray:
    """Calculate stability of predictions.

    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py

    Args:
        logits: shape = (batch, ..., num_classes).
        threshold: threshold for prediction.
        threshold_offset: offset for threshold.

    Returns:
        Stability of shape (batch, num_classes).

    Raises:
        ValueError: if threshold_offset is negative.
    """
    if threshold_offset < 0:
        raise ValueError(f"threshold_offset must be non-negative, got {threshold_offset}.")
    # logits max values is 0
    logits -= jnp.mean(logits, axis=-1, keepdims=True)
    mask_high_threshold = logits >= (threshold + threshold_offset)
    mask_low_threshold = logits >= (threshold - threshold_offset)
    return iou(mask_high_threshold, mask_low_threshold)
