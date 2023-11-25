"""Module for convolution layers."""
from __future__ import annotations

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from imgx.model.basic import InstanceNorm


class ConvNormAct(nn.Module):
    """Block with conv-norm-act."""

    num_spatial_dims: int
    out_channels: int
    kernel_size: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: (batch, *spatial_shape, in_channels).

        Returns:
            Array.
        """
        return nn.Sequential(
            [
                nn.Conv(
                    features=self.out_channels,
                    kernel_size=(self.kernel_size,) * self.num_spatial_dims,
                    use_bias=False,
                    dtype=self.dtype,
                ),
                InstanceNorm(dtype=self.dtype),
                self.activation,
            ]
        )(x)


class ConvResBlockWithoutTime(nn.Module):
    """Block with two conv-norm-act layers, residual link, but without time."""

    num_spatial_dims: int
    out_channels: int
    kernel_size: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: (batch, *spatial_shape, in_channels).

        Returns:
            Array.
        """
        res = x
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,) * self.num_spatial_dims,
            use_bias=False,
            dtype=self.dtype,
        )(x)
        x = InstanceNorm(dtype=self.dtype)(x)
        x = self.activation(x)
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,) * self.num_spatial_dims,
            use_bias=False,
            dtype=self.dtype,
        )(x)
        x = InstanceNorm(dtype=self.dtype)(x)
        x = self.activation(x + res)
        return x


class ConvResBlockWithTime(nn.Module):
    """Block with two conv-norm-act layers, residual link, and time intput.

    https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
    """

    num_spatial_dims: int
    out_channels: int
    kernel_size: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        t_emb: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: (batch, *spatial_shape, in_channels).
            t_emb: time embedding, (batch, t_channels).

        Returns:
            Array.
        """
        t_emb = jnp.expand_dims(t_emb, axis=range(1, self.num_spatial_dims + 1))
        t_emb = self.activation(t_emb)
        t_emb = nn.Dense(self.out_channels, dtype=self.dtype)(t_emb)

        res = x
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,) * self.num_spatial_dims,
            use_bias=False,
            dtype=self.dtype,
        )(x)
        x = InstanceNorm(dtype=self.dtype)(x)
        x = self.activation(x)
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,) * self.num_spatial_dims,
            use_bias=False,
            dtype=self.dtype,
        )(x)
        x += t_emb
        x = InstanceNorm(dtype=self.dtype)(x)
        x = self.activation(x + res)
        return x


class ConvResBlock(nn.Module):
    """ConvResBlock with optional time embedding input.

    https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
    """

    num_spatial_dims: int
    out_channels: int
    kernel_size: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        t_emb: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: (batch, *spatial_shape, in_channels).
            t_emb: time embedding, if not None, (batch, t_channels).

        Returns:
            Array.
        """
        if t_emb is None:
            return ConvResBlockWithoutTime(
                num_spatial_dims=self.num_spatial_dims,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                activation=self.activation,
                dtype=self.dtype,
            )(x)

        conv_t = ConvResBlockWithTime(
            num_spatial_dims=self.num_spatial_dims,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            activation=self.activation,
            dtype=self.dtype,
        )
        return conv_t(x, t_emb)


class ConvDownSample(nn.Module):
    """Down-sample with Conv."""

    out_channels: int
    scale_factor: tuple[int, ...]
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward.

        Args:
            x: shape (batch, *spatial_shape, in_channels).

        Returns:
            Down-sampled array,
                (batch, *down_sampled_spatial_shape, out_channels).
        """
        return nn.Sequential(
            [
                nn.Conv(
                    features=self.out_channels,
                    kernel_size=self.scale_factor,
                    strides=self.scale_factor,
                    use_bias=False,
                    dtype=self.dtype,
                ),
                InstanceNorm(dtype=self.dtype),
            ]
        )(x)


class ConvUpSample(nn.Module):
    """Up-sample with ConvTranspose."""

    out_channels: int
    scale_factor: tuple[int, ...]
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward.

        Args:
            x: shape (batch, *spatial_shape, in_channels).

        Returns:
            Down-sampled array,
                (batch, *down_sampled_spatial_shape, out_channels).
        """
        return nn.Sequential(
            [
                nn.ConvTranspose(
                    features=self.out_channels,
                    kernel_size=self.scale_factor,
                    strides=self.scale_factor,
                    use_bias=False,
                    dtype=self.dtype,
                ),
                InstanceNorm(dtype=self.dtype),
            ]
        )(x)
