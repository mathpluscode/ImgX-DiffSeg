"""Basic functions and modules."""
from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

# flax LayerNorm differs from haiku
LayerNorm = partial(
    nn.LayerNorm,
    epsilon=1e-5,
    use_fast_variance=False,
)


def truncated_normal(
    stddev: float | jnp.ndarray = 1.0,
    mean: float | jnp.ndarray = 0.0,
    dtype: jnp.dtype = jnp.float_,
) -> Callable[[jax.random.PRNGKeyArray, jnp.shape, jnp.dtype], jnp.ndarray]:
    """Truncated normal initializer as in haiku.

    Args:
        stddev: standard deviation of the truncated normal distribution.
        mean: mean of the truncated normal distribution.
        dtype: dtype of the array.

    Returns:
        Initializer function.
    """

    def init(
        key: jax.random.KeyArray, shape: Sequence[int], dtype: jnp.dtype = dtype
    ) -> jnp.ndarray:
        """Init function.

        Args:
            key: random key.
            shape: shape of the array.
            dtype: dtype of the array.
        """
        real_dtype = jnp.finfo(dtype).dtype
        m = jax.lax.convert_element_type(mean, dtype)
        s = jax.lax.convert_element_type(stddev, real_dtype)
        is_complex = jnp.issubdtype(dtype, jnp.complexfloating)
        if is_complex:
            shape = [2, *shape]
        unscaled = jax.random.truncated_normal(key, -2.0, 2.0, shape, real_dtype)
        if is_complex:
            unscaled = unscaled[0] + 1j * unscaled[1]
        return s * unscaled + m

    return init


class InstanceNorm(nn.Module):
    """Instance norm."""

    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: input with batch axis, (batch, ...).

        Returns:
            Normalised input.
        """
        reduction_axes = tuple(range(x.ndim)[slice(1, -1)])
        return LayerNorm(
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
        [jax.random.PRNGKeyArray, jnp.shape, jnp.dtype], jnp.ndarray
    ] = nn.initializers.lecun_normal()
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: shape (..., in_size)

        Returns:
            shape (..., out_size)
        """
        return nn.Sequential(
            [
                nn.Dense(
                    self.emb_size,
                    kernel_init=self.kernel_init,
                    dtype=self.dtype,
                ),
                self.activation,
                nn.Dense(
                    self.output_size,
                    kernel_init=self.kernel_init,
                    dtype=self.dtype,
                ),
            ]
        )(x)
