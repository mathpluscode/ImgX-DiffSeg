"""Basic functions and modules."""
import haiku as hk
from jax import numpy as jnp


def layer_norm(x: jnp.ndarray) -> jnp.ndarray:
    """Applies a unique LayerNorm at the last axis.

    Args:
        x: input

    Returns:
        Normalised input.
    """
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)


def instance_norm(x: jnp.ndarray) -> jnp.ndarray:
    """Applies a unique InstanceNorm.

    Args:
        x: input

    Returns:
        Normalised input.
    """
    return hk.InstanceNorm(create_scale=True, create_offset=True)(x)


def dropout(x: jnp.ndarray, dropout_rate: float) -> jnp.ndarray:
    """Applies dropout only if necessary.

    This function is necessary to avoid defining random key for testing.
    Otherwise, calling `hk.dropout` will result in the following error:
        You must pass a non-None PRNGKey to init and/or apply
        if you make use of random numbers.

    Args:
        x: input
        dropout_rate: rate of dropout

    Returns:
        Dropout applied input.
    """
    if dropout_rate == 0.0:  # noqa: PLR2004
        return x
    return hk.dropout(hk.next_rng_key(), dropout_rate, x)


def sinusoidal_positional_embedding(
    x: jnp.ndarray, dim: int, max_period: int = 10000
) -> jnp.ndarray:
    """Create sinusoidal timestep embeddings.

    Half defined by sin, half by cos.
    For position x, the embeddings are (for i = 0,...,half_dim-1)
        sin(x / (max_period ** (i/half_dim)))
        cos(x / (max_period ** (i/half_dim)))

    Args:
        x: (batch, ), with non-negative values.
        dim: embedding dimension, assume to be evenly divided by two.
        max_period: controls the minimum frequency of the embeddings.

    Returns:
        Embedding of size (batch, dim).
    """
    half_dim = dim // 2
    # (half_dim,)
    freq = jnp.arange(0, half_dim, dtype=jnp.float32)
    freq = jnp.exp(-jnp.log(max_period) * freq / half_dim)
    # (batch, half_dim)
    args = x[:, None] * freq[None, :]
    # (batch, dim)
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
