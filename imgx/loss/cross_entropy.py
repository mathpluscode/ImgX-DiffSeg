"""Loss functions for classification."""
import jax
import jax.numpy as jnp
import optax
from jax import lax


def cross_entropy(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
    classes_are_exclusive: bool,
) -> jnp.ndarray:
    """Cross entropy, supporting soft label.

    optax.softmax_cross_entropy returns (batch, ...).
    optax.sigmoid_binary_cross_entropy returns (batch, ..., num_classes).

    Args:
        logits: unscaled prediction, (batch, ..., num_classes).
        mask_true: probabilities per class, (batch, ..., num_classes).
        classes_are_exclusive: if False, each element can be assigned to multiple classes.

    Returns:
        Cross entropy loss value of shape (batch, ).
    """
    mask_true = mask_true.astype(logits.dtype)
    loss = lax.cond(
        classes_are_exclusive,
        optax.softmax_cross_entropy,
        lambda *args: jnp.sum(optax.sigmoid_binary_cross_entropy(*args), axis=-1),
        logits,
        mask_true,
    )
    return jnp.mean(loss, axis=range(1, loss.ndim))


def softmax_focal_loss(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
    gamma: float = 2.0,
) -> jnp.ndarray:
    """Focal loss with one-hot / mutual exclusive classes.

    https://arxiv.org/abs/1708.02002
    Implementation is similar to optax.softmax_cross_entropy.

    Args:
        logits: unscaled prediction, (batch, ..., num_classes).
        mask_true: probabilities per class, (batch, ..., num_classes).
        gamma: adjust class imbalance, 0 is equivalent to cross entropy.

    Returns:
        Loss of shape (batch, ...).
    """
    log_p = jax.nn.log_softmax(logits)
    p = jnp.exp(log_p)
    return -jnp.sum(((1 - p) ** gamma) * log_p * mask_true, axis=-1)


def sigmoid_focal_loss(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
    gamma: float = 2.0,
) -> jnp.ndarray:
    """Focal loss with multi-hot / non mutual exclusive classes.

    https://arxiv.org/abs/1708.02002
    Implementation is similar to optax.sigmoid_binary_cross_entropy.

    Args:
        logits: unscaled prediction, (batch, ..., num_classes).
        mask_true: probabilities per class, (batch, ..., num_classes).
        gamma: adjust class imbalance, 0 is equivalent to cross entropy.

    Returns:
        Focal loss value of shape (batch, ..., num_classes).
    """
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    p = jnp.exp(log_p)
    return -((1 - p) ** gamma) * log_p * mask_true - (p**gamma) * log_not_p * (1 - mask_true)


def focal_loss(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
    classes_are_exclusive: bool,
    gamma: float = 2.0,
) -> jnp.ndarray:
    """Focal loss.

    https://arxiv.org/abs/1708.02002
    softmax_focal_loss returns (batch, ...).
    sigmoid_focal_loss returns (batch, ..., num_classes).

    Args:
        logits: unscaled prediction, (batch, ..., num_classes).
        mask_true: probabilities per class, (batch, ..., num_classes).
        classes_are_exclusive: if False, each element can be assigned to multiple classes.
        gamma: adjust class imbalance, 0 is equivalent to cross entropy.

    Returns:
        Focal loss value of shape (batch, ).
    """
    mask_true = mask_true.astype(logits.dtype)
    loss = lax.cond(
        classes_are_exclusive,
        softmax_focal_loss,
        lambda *args: jnp.sum(sigmoid_focal_loss(*args), axis=-1),
        logits,
        mask_true,
        gamma,
    )
    return jnp.mean(loss, axis=range(1, loss.ndim))
