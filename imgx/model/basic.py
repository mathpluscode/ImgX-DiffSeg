"""Basic functions and modules."""
from __future__ import annotations

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp


class Identity(nn.Module):
    """Identity module."""

    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: input.

        Returns:
            input.
        """
        return x


class InstanceNorm(nn.Module):
    """Instance norm.

    The norm is calculated on axes excluding batch and features.
    """

    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: input with batch axis, (batch, ..., channel).

        Returns:
            Normalised input.
        """
        reduction_axes = tuple(range(x.ndim)[slice(1, -1)])
        return nn.LayerNorm(
            reduction_axes=reduction_axes,
        )(x)


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
        dtype: dtype of the embeddings.

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


class MLP(nn.Module):
    """Two-layer MLP."""

    emb_size: int
    output_size: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu
    kernel_init: Callable[
        [jax.Array, jnp.shape, jnp.dtype], jnp.ndarray
    ] = nn.initializers.lecun_normal()
    remat: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: shape (..., in_size)

        Returns:
            shape (..., out_size)
        """
        dense_cls = nn.remat(nn.Dense) if self.remat else nn.Dense
        x = dense_cls(
            self.emb_size,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.activation(x)
        x = dense_cls(
            self.output_size,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        return x
