"""Loss functions for classification."""
import jax
import jax.numpy as jnp
import optax


def softmax_cross_entropy(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
) -> jnp.ndarray:
    """Softmax cross entropy.

    Args:
        logits: unscaled prediction, (batch, ..., num_classes).
        mask_true: one hot targets, (batch, ..., num_classes).

    Returns:
        Cross entropy loss value of shape (batch, ).
    """
    # (batch, ...)
    loss = optax.softmax_cross_entropy(
        logits=logits,
        labels=mask_true,
    )
    return jnp.mean(loss, axis=range(1, loss.ndim))


def sigmoid_binary_cross_entropy(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
) -> jnp.ndarray:
    """Sigmoid binary cross entropy.

    Args:
        logits: unscaled prediction, (batch, ..., num_classes).
        mask_true: multi-hot targets, (batch, ..., num_classes).

    Returns:
        Cross entropy loss value of shape (batch, ).
    """
    # (batch, ..., num_classes)
    loss = optax.sigmoid_binary_cross_entropy(
        logits=logits,
        labels=mask_true,
    )
    return jnp.mean(loss, axis=range(1, loss.ndim))


def cross_entropy(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
    classes_are_exclusive: bool,
) -> jnp.ndarray:
    """Cross entropy.

    Args:
        logits: unscaled prediction, (batch, ..., num_classes).
        mask_true: binary targets, (batch, ..., num_classes).
        classes_are_exclusive: if True, mask_true is one hot encoded.

    Returns:
        Cross entropy loss value of shape (batch, ).
    """
    return jax.lax.cond(
        classes_are_exclusive,
        softmax_cross_entropy,
        sigmoid_binary_cross_entropy,
        logits,
        mask_true,
    )


def softmax_focal_loss(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
    gamma: float = 2.0,
) -> jnp.ndarray:
    """Focal loss with one-hot / mutual exclusive classes.

    https://arxiv.org/abs/1708.02002

    Args:
        logits: unscaled prediction, (batch, ..., num_classes).
        mask_true: one hot targets, (batch, ..., num_classes).
        gamma: adjust class imbalance, 0 is equivalent to cross entropy.

    Returns:
        Focal loss value of shape (batch, ).
    """
    # normalise logits to be the log of probabilities
    # (batch, ..., num_classes)
    log_probs = jax.nn.log_softmax(logits)
    probs = jnp.exp(log_probs)
    loss = -((1 - probs) ** gamma) * log_probs * mask_true
    # (batch, ..., num_classes) -> (batch, ...)
    # label are one hot, just sum over class axis
    loss = jnp.sum(loss, axis=-1)
    return jnp.mean(loss, axis=range(1, loss.ndim))


def sigmoid_focal_loss(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
    gamma: float = 2.0,
) -> jnp.ndarray:
    """Focal loss with multi-hot / non mutual exclusive classes.

    https://arxiv.org/abs/1708.02002

    Args:
        logits: unscaled prediction, (batch, ..., num_classes).
        mask_true: binary targets, (batch, ..., num_classes).
        gamma: adjust class imbalance, 0 is equivalent to cross entropy.

    Returns:
        Focal loss value of shape (batch, ).
    """
    # normalise logits to be the log of probabilities
    # (batch, ..., num_classes)
    log_probs = jax.nn.log_sigmoid(logits)
    # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
    log_not_probs = jax.nn.log_sigmoid(-logits)
    probs = jnp.exp(log_probs)

    loss_foreground = -((1 - probs) ** gamma) * log_probs * mask_true
    loss_background = -(probs**gamma) * log_not_probs * (1 - mask_true)
    loss = loss_foreground + loss_background

    # (batch, ..., num_classes) -> (batch, ...)
    return jnp.mean(loss, axis=range(1, loss.ndim))


def focal_loss(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
    classes_are_exclusive: bool,
    gamma: float = 2.0,
) -> jnp.ndarray:
    """Focal loss.

    https://arxiv.org/abs/1708.02002

    Args:
        logits: unscaled prediction, (batch, ..., num_classes).
        mask_true: binary targets, (batch, ..., num_classes).
        classes_are_exclusive: if True, mask_true is one hot encoded.
        gamma: adjust class imbalance, 0 is equivalent to cross entropy.

    Returns:
        Focal loss value of shape (batch, ).
    """
    return jax.lax.cond(
        classes_are_exclusive,
        softmax_focal_loss,
        sigmoid_focal_loss,
        logits,
        mask_true,
        gamma,
    )
