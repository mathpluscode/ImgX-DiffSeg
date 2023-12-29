"""Module for convolution layers."""
from __future__ import annotations

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from imgx.model.basic import InstanceNorm


class ConvNormAct(nn.Module):
    """Block with conv-norm-act."""

    out_channels: int
    kernel_size: tuple[int, ...]
    strides: tuple[int, ...] | int = 1
    padding: str = "SAME"
    feature_group_count: int = 1
    norm: nn.Module = InstanceNorm
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu
    remat: bool = True
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
            (batch, *spatial_shape, out_channels),
            output spatial shape may be different from input.
        """
        conv_cls = nn.remat(nn.Conv) if self.remat else nn.Conv
        norm_cls = nn.remat(self.norm) if self.remat else self.norm
        return nn.Sequential(
            [
                conv_cls(
                    features=self.out_channels,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    feature_group_count=self.feature_group_count,
                    use_bias=False,
                    dtype=self.dtype,
                ),
                norm_cls(dtype=self.dtype),
                self.activation,
            ]
        )(x)


class ConvResBlockWithoutTime(nn.Module):
    """Block with two conv-norm-act layers, residual link, but without time."""

    out_channels: int
    kernel_size: tuple[int, ...]
    norm: nn.Module = InstanceNorm
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu
    dropout: float = 0.0
    remat: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        is_train: bool,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            is_train: whether in training mode.
            x: (batch, *spatial_shape, in_channels).

        Returns:
            (batch, *spatial_shape, out_channels),
            output spatial shape may be different from input.
        """
        conv_cls = nn.remat(nn.Conv) if self.remat else nn.Conv
        norm_cls = nn.remat(self.norm) if self.remat else self.norm
        res = x
        x = conv_cls(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            use_bias=False,
            dtype=self.dtype,
        )(x)
        x = norm_cls(dtype=self.dtype)(x)
        x = self.activation(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not is_train)(x)
        x = conv_cls(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            use_bias=False,
            dtype=self.dtype,
        )(x)
        x = norm_cls(dtype=self.dtype)(x)
        x = self.activation(x + res)
        return x


class ConvResBlockWithTime(nn.Module):
    """Block with two conv-norm-act layers, residual link, and time intput.

    https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
    """

    out_channels: int
    kernel_size: tuple[int, ...]
    norm: nn.Module = InstanceNorm
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu
    dropout: float = 0.0
    remat: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        is_train: bool,
        x: jnp.ndarray,
        t_emb: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            is_train: whether in training mode.
            x: (batch, *spatial_shape, in_channels).
            t_emb: time embedding, (batch, t_channels).

        Returns:
            (batch, *spatial_shape, out_channels),
            output spatial shape may be different from input.
        """
        dense_cls = nn.remat(nn.Dense) if self.remat else nn.Dense
        conv_cls = nn.remat(nn.Conv) if self.remat else nn.Conv
        norm_cls = nn.remat(self.norm) if self.remat else self.norm

        num_spatial_dims = len(self.kernel_size)
        t_emb = jnp.expand_dims(t_emb, axis=range(1, num_spatial_dims + 1))
        t_emb = self.activation(t_emb)
        t_emb = dense_cls(self.out_channels, dtype=self.dtype)(t_emb)

        res = x
        x = conv_cls(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            use_bias=False,
            dtype=self.dtype,
        )(x)
        x = norm_cls(dtype=self.dtype)(x)
        x = self.activation(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not is_train)(x)
        x = conv_cls(
            features=self.out_channels,
            kernel_size=self.kernel_size,
            use_bias=False,
            dtype=self.dtype,
        )(x)
        x += t_emb
        x = norm_cls(dtype=self.dtype)(x)
        x = self.activation(x + res)
        return x


class ConvResBlock(nn.Module):
    """ConvResBlock with optional time embedding input.

    https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
    """

    out_channels: int
    kernel_size: tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu
    dropout: float = 0.0
    remat: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        is_train: bool,
        x: jnp.ndarray,
        t_emb: jnp.ndarray | None,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            is_train: whether in training mode.
            x: (batch, *spatial_shape, in_channels).
            t_emb: time embedding, if not None, (batch, t_channels).

        Returns:
            Array.
        """
        if t_emb is None:
            return ConvResBlockWithoutTime(
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                activation=self.activation,
                dropout=self.dropout,
                remat=self.remat,
                dtype=self.dtype,
            )(is_train, x)

        conv_t = ConvResBlockWithTime(
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            activation=self.activation,
            dropout=self.dropout,
            remat=self.remat,
            dtype=self.dtype,
        )
        return conv_t(is_train, x, t_emb)


class ConvDownSample(nn.Module):
    """Down-sample with Conv."""

    out_channels: int
    scale_factor: tuple[int, ...]
    norm: nn.Module = InstanceNorm
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
                self.norm(dtype=self.dtype),
            ]
        )(x)


class ConvUpSample(nn.Module):
    """Up-sample with ConvTranspose."""

    out_channels: int
    scale_factor: tuple[int, ...]
    norm: nn.Module = InstanceNorm
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
                self.norm(dtype=self.dtype),
            ]
        )(x)
