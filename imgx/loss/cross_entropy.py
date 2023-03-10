"""Loss functions for classification."""
import jax
import jax.numpy as jnp
import optax


def mean_cross_entropy(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
) -> jnp.ndarray:
    """Cross entropy.

    Args:
        logits: unscaled prediction, (batch, ..., num_classes).
        mask_true: one hot targets, (batch, ..., num_classes).

    Returns:
        Cross entropy loss value of shape (1, ).
    """
    # (batch, ...)
    loss = optax.softmax_cross_entropy(logits=logits, labels=mask_true)
    return jnp.mean(loss)


def mean_focal_loss(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
    gamma: float = 2.0,
) -> jnp.ndarray:
    """Focal loss.

    https://arxiv.org/abs/1708.02002

    Args:
        logits: unscaled prediction, (batch, ..., num_classes).
        mask_true: one hot targets, (batch, ..., num_classes).
        gamma: adjust class imbalance, 0 is equivalent to cross entropy.

    Returns:
        Focal loss value of shape (1, ).
    """
    # normalise logits to be the log of probabilities
    logits = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(logits)
    focal_loss = -((1 - probs) ** gamma) * logits * mask_true
    # (batch, ..., num_classes) -> (batch, ...)
    # label are one hot, just sum over class axis
    focal_loss = jnp.sum(focal_loss, axis=-1)
    return jnp.mean(focal_loss)
