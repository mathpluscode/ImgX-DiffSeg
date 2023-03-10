"""Module for math functions."""

import jax
import jax.numpy as jnp


def logits_to_mask(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Transform logits to one hot mask.

    The one will be on the class having largest logit.

    Args:
        x: logits.
        axis: axis of num_classes.

    Returns:
        One hot probabilities.
    """
    return jax.nn.one_hot(
        x=jnp.argmax(x, axis=axis),
        num_classes=x.shape[axis],
        axis=axis,
    )
