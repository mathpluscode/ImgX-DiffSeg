"""Module for convolution layers."""
from __future__ import annotations

import dataclasses
from typing import Callable

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from imgx.model.basic import instance_norm


@dataclasses.dataclass
class ConvNDNormAct(hk.Module):
    """Block with conv-norm-act."""

    num_spatial_dims: int
    out_channels: int
    kernel_size: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu

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
        x = hk.ConvND(
            num_spatial_dims=self.num_spatial_dims,
            output_channels=self.out_channels,
            kernel_shape=self.kernel_size,
            with_bias=False,
        )(x)
        x = instance_norm(x)
        x = self.activation(x)
        return x


@dataclasses.dataclass
class ConvNDResBlockWithoutTime(hk.Module):
    """Block with two conv-norm-act layers, residual link, but without time."""

    num_spatial_dims: int
    out_channels: int
    kernel_size: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu

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
        x = hk.ConvND(
            num_spatial_dims=self.num_spatial_dims,
            output_channels=self.out_channels,
            kernel_shape=self.kernel_size,
            with_bias=False,
        )(x)
        x = instance_norm(x)
        x = self.activation(x)
        x = hk.ConvND(
            num_spatial_dims=self.num_spatial_dims,
            output_channels=self.out_channels,
            kernel_shape=self.kernel_size,
            with_bias=False,
        )(x)
        x = instance_norm(x)
        x = self.activation(x + res)
        return x


@dataclasses.dataclass
class ConvNDResBlockWithTime(hk.Module):
    """Block with two conv-norm-act layers, residual link, and time intput.

    https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
    """

    num_spatial_dims: int
    out_channels: int
    kernel_size: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu

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
        t_emb = hk.Linear(output_size=self.out_channels)(t_emb)

        res = x
        x = hk.ConvND(
            num_spatial_dims=self.num_spatial_dims,
            output_channels=self.out_channels,
            kernel_shape=self.kernel_size,
            with_bias=False,
        )(x)
        x = instance_norm(x)
        x = self.activation(x)
        x = hk.ConvND(
            num_spatial_dims=self.num_spatial_dims,
            output_channels=self.out_channels,
            kernel_shape=self.kernel_size,
            with_bias=False,
        )(x)
        x += t_emb
        x = instance_norm(x)
        x = self.activation(x + res)
        return x


@dataclasses.dataclass
class ConvNDResBlock(hk.Module):
    """ConvNDResBlock with optional time embedding input.

    https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
    """

    num_spatial_dims: int
    out_channels: int
    kernel_size: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu
    remat: bool = True

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
            conv = ConvNDResBlockWithoutTime(
                num_spatial_dims=self.num_spatial_dims,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                activation=self.activation,
            )
            conv = hk.remat(conv) if self.remat else conv
            return conv(x)

        conv_t = ConvNDResBlockWithTime(
            num_spatial_dims=self.num_spatial_dims,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            activation=self.activation,
        )
        conv_t = hk.remat(conv_t) if self.remat else conv_t
        return conv_t(x, t_emb)


@dataclasses.dataclass
class ConvNDDownSample(hk.Module):
    """Down-sample with ConvND."""

    num_spatial_dims: int
    out_channels: int
    scale_factor: int
    remat: bool = True

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
        block = hk.Sequential(
            [
                hk.ConvND(
                    num_spatial_dims=self.num_spatial_dims,
                    output_channels=self.out_channels,
                    kernel_shape=self.scale_factor,
                    stride=self.scale_factor,
                    with_bias=False,
                ),
                instance_norm,
            ]
        )
        block = hk.remat(block) if self.remat else block
        return block(x)


@dataclasses.dataclass
class ConvNDUpSample(hk.Module):
    """Up-sample with ConvNDTranspose."""

    num_spatial_dims: int
    out_channels: int
    scale_factor: int
    remat: bool = True

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
        block = hk.Sequential(
            [
                hk.ConvNDTranspose(
                    num_spatial_dims=self.num_spatial_dims,
                    output_channels=self.out_channels,
                    kernel_shape=self.scale_factor,
                    stride=self.scale_factor,
                    with_bias=False,
                ),
                instance_norm,
            ]
        )
        block = hk.remat(block) if self.remat else block
        return block(x)


@dataclasses.dataclass
class PatchEmbedding(hk.Module):
    """Patch embedding for Vision Transformer."""

    patch_shape: tuple[int, ...]
    model_size: int  # patch embedding dimension
    add_position_embedding: bool = True

    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass.

        Supports cases where spatial_shape is not divisible by patch_shape.

        Args:
            x: shape (batch, *spatial_shape, in_channels).

        Returns:
            shape (batch, *spatial_shape, model_size).
        """
        batch, *spatial_shape, _ = x.shape
        patched_shape = tuple(
            (ss + ps - 1) // ps
            for ss, ps in zip(spatial_shape, self.patch_shape)
        )
        num_spatial_dims = len(self.patch_shape)
        num_patches = np.prod(patched_shape)
        chex.assert_rank(x, 2 + num_spatial_dims)

        # split into patches and encode each patch
        conv_layer = hk.ConvND(
            num_spatial_dims=num_spatial_dims,
            output_channels=self.model_size,
            kernel_shape=self.patch_shape,
            stride=self.patch_shape,
        )
        x = conv_layer(x)

        # each pixe/voxel corresponds to one patch before
        # because of the stride
        x = jnp.reshape(x, (batch, num_patches, self.model_size))

        # embed the input tokens and positions.
        # (batch, num_patches, model_size)
        if self.add_position_embedding:
            positional_embeddings = hk.get_parameter(
                name="positional_embeddings",
                shape=[1, num_patches, self.model_size],
                init=hk.initializers.TruncatedNormal(stddev=0.02),
            )
            x += positional_embeddings
        # (batch, *patched_shape, model_size)
        x = x.reshape((batch, *patched_shape, self.model_size))

        return x
