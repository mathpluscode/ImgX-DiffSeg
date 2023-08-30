"""Basic functions and modules."""
from __future__ import annotations

import dataclasses
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp


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


def adaptive_norm(
    x: jnp.ndarray, cond: jnp.ndarray, norm_fn: Callable
) -> jnp.ndarray:
    """Adaptive normalization.

    https://github.com/cientgu/VQ-Diffusion/blob/85813353e6a42f18f74f6f50cdeb4ddf5ee3d1d1/image_synthesis/modeling/transformers/transformer_utils.py#L136

    Usage:
        from functools import partial
        adaptive_layer_norm = partial(adaptive_norm, norm_fn=layer_norm)

    Args:
        x: input, shape (..., emb_size)
        cond: condition, broadcast compatible shape (..., cond_emb_size)
        norm_fn: normalization function which returns the same shape as input.

    Returns:
        Normalised input.
    """
    emb_size = x.shape[-1]
    # (..., emb_size*2)
    cond = hk.Linear(output_size=emb_size * 2)(jax.nn.silu(cond))
    # (..., emb_size)
    scale, shift = jnp.split(cond, 2, axis=-1)
    x = norm_fn(x) * (1 + scale) + shift
    return x


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
    x: jnp.ndarray,
    dim: int,
    max_period: int = 10000,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Create sinusoidal timestep embeddings.

    Half defined by sin, half by cos.
    For position x, the embeddings are (for i = 0,...,half_dim-1)
        sin(x / (max_period ** (i/half_dim)))
        cos(x / (max_period ** (i/half_dim)))

    Args:
        x: (..., ), with values in [0, 1].
        dim: embedding dimension, assume to be evenly divided by two.
        max_period: controls the minimum frequency of the embeddings.

    Returns:
        Embedding of size (..., dim).
    """
    ndim_x = len(x.shape)
    if dim % 2 != 0:
        raise ValueError(f"dim must be evenly divided by two, got {dim}.")
    half_dim = dim // 2
    # (half_dim,)
    freq = jnp.arange(0, half_dim, dtype=dtype)
    freq = jnp.exp(-jnp.log(max_period) * freq / half_dim)
    # (..., half_dim)
    freq = jnp.expand_dims(freq, axis=tuple(range(ndim_x)))
    args = x[..., None] * max_period * freq
    # (..., dim)
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


@dataclasses.dataclass
class MLP(hk.Module):
    """Two-layer MLP."""

    emb_size: int
    output_size: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu
    initializer: hk.initializers.Initializer | None = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: shape (..., in_size)

        Returns:
            shape (..., out_size)
        """
        return hk.Sequential(
            [
                hk.Linear(output_size=self.emb_size, w_init=self.initializer),
                self.activation,
                hk.Linear(
                    output_size=self.output_size, w_init=self.initializer
                ),
            ]
        )(x)
