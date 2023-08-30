"""Loss functions for image segmentation."""
import jax
import jax.numpy as jnp


def dice_loss(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
    classes_are_exclusive: bool,
) -> jnp.ndarray:
    """Mean dice loss, smaller is better.

    Losses are not calculated on instance-classes, where there is no label.
    This is to avoid the need of smoothing and potentially nan gradients.

    Args:
        logits: unscaled prediction, (batch, ..., num_classes).
        mask_true: binary masks, (batch, ..., num_classes).
        classes_are_exclusive: classes are exclusive, i.e. no overlap.

    Returns:
        Dice loss value of shape (batch, num_classes).
    """
    mask_pred = jax.lax.cond(
        classes_are_exclusive,
        jax.nn.softmax,
        jax.nn.sigmoid,
        logits,
    )
    reduce_axis = tuple(range(mask_pred.ndim))[1:-1]
    # (batch, num_classes)
    numerator = 2.0 * jnp.sum(mask_pred * mask_true, axis=reduce_axis)
    denominator = jnp.sum(mask_pred + mask_true, axis=reduce_axis)
    not_nan_mask = jnp.sum(mask_true, axis=reduce_axis) > 0
    # nan loss are replaced by 0.0
    return jnp.where(
        condition=not_nan_mask,
        x=1.0 - numerator / denominator,
        y=jnp.nan,
    )
