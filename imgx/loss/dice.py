"""Loss functions for image segmentation."""
import jax
import jax.numpy as jnp


def mean_with_background(batch_cls_loss: jnp.ndarray) -> jnp.ndarray:
    """Return average with background class.

    Args:
        batch_cls_loss: shape (batch, num_classes).

    Returns:
        Mean loss of shape (1,).
    """
    return jnp.nanmean(batch_cls_loss)


def mean_without_background(batch_cls_loss: jnp.ndarray) -> jnp.ndarray:
    """Return average without background class.

    Args:
        batch_cls_loss: shape (batch, num_classes).

    Returns:
        Mean loss of shape (1,).
    """
    return jnp.nanmean(batch_cls_loss[:, 1:])


def dice_loss(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
) -> jnp.ndarray:
    """Mean dice loss, smaller is better.

    Losses are not calculated on instance-classes, where there is no label.
    This is to avoid the need of smoothing and potentially nan gradients.

    Args:
        logits: unscaled prediction, (batch, ..., num_classes).
        mask_true: one hot targets, (batch, ..., num_classes).

    Returns:
        Dice loss value of shape (batch, num_classes).
    """
    mask_pred = jax.nn.softmax(logits)
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


def mean_dice_loss(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
    include_background: bool,
) -> jnp.ndarray:
    """Mean dice loss, smaller is better.

    Losses are not calculated on instance-classes, where there is no label.
    This is to avoid the need of smoothing and potentially nan gradients.

    Args:
        logits: unscaled prediction, (batch, ..., num_classes).
        mask_true: one hot targets, (batch, ..., num_classes).
        include_background: include background as a separate class.

    Returns:
        Dice loss value of shape (1, ).
    """
    loss = dice_loss(logits=logits, mask_true=mask_true)
    return jax.lax.cond(
        include_background,
        mean_with_background,
        mean_without_background,
        loss,
    )
