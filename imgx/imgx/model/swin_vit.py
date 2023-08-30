"""Swin Transformer and related functionalities.

https://arxiv.org/abs/2111.09883
https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py
"""
from __future__ import annotations

import dataclasses

import chex
import haiku as hk
import jax.nn
import jax.numpy as jnp
from jax import lax

from imgx.model.basic import MLP, layer_norm
from imgx.model.conv import ConvNDDownSample, PatchEmbedding
from imgx.model.window import (
    get_window_mask,
    get_window_shift_pad_shapes,
    window_partition,
    window_unpartition,
)
from imgx.model.window_attention import WindowMultiHeadAttention


@dataclasses.dataclass
class SwinMultiHeadAttention(hk.Module):
    """Shifted window based multi-head attention.

    Pad - shift - split into windows - mha - merge windows - unshift - unpad.

    TODO: pretrained_window_shape = window_shape for now.

    Modification: for an axis, if the window size is larger than spatial size,
    window is shrunk to avoid unnecessary padding, and no need of shift.
    """

    # swin specific
    window_shape: tuple[int, ...]
    shift_shape: tuple[int, ...]  # set to all zeros to disable shift
    # transformer specific
    num_heads: int
    widening_factor: int = 4
    initializer: hk.initializers.Initializer | None = None

    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: shape (batch, *spatial_shape, model_size).

        Returns:
            Array of shape (batch, *spatial_shape, model_size).
        """
        _, *spatial_shape, model_size = x.shape
        num_spatial_dims = len(spatial_shape)
        (
            window_shape,
            spatial_padding,
            padded_spatial_shape,
            shift_shape,
            neg_shift_shape,
        ) = get_window_shift_pad_shapes(
            spatial_shape=spatial_shape,
            window_shape=self.window_shape,
            shift_shape=self.shift_shape,
        )

        if max(spatial_padding) > 0:
            padding_config = (
                [(0, 0, 0)] + [(0, x, 0) for x in spatial_padding] + [(0, 0, 0)]
            )
            x = lax.pad(
                x,
                padding_config=padding_config,
                padding_value=jnp.array(0.0, dtype=x.dtype),
            )

        # cyclic shift
        # (batch, *padded_spatial_shape, model_size)
        if max(shift_shape) > 0:
            x = jnp.roll(
                x, shift=neg_shift_shape, axis=range(1, 1 + num_spatial_dims)
            )

        # split into windows
        # (batch, num_windows, window_volume, model_size)
        x = window_partition(x=x, window_shape=window_shape)

        # attention
        key_size = model_size
        if model_size >= self.widening_factor:
            key_size = model_size // self.widening_factor
        mha = WindowMultiHeadAttention(
            pretrained_window_shape=window_shape,
            num_heads=self.num_heads,
            key_size=key_size,
            model_size=model_size,
            w_init=self.initializer,
        )
        mask = None
        if max(shift_shape) > 0:
            # (num_windows, window_volume, window_volume)
            mask = get_window_mask(
                spatial_shape=padded_spatial_shape,
                window_shape=window_shape,
                shift_shape=shift_shape,
            )
            # (1, num_windows, 1, window_volume, window_volume)
            mask = mask[None, :, None, :, :]
        # (batch, num_windows, window_volume, model_size)
        x = mha(x=x, mask=mask, window_shape=window_shape)

        # merge windows
        # (batch, *padded_spatial_shape, model_size)
        x = window_unpartition(
            x=x,
            window_shape=window_shape,
            spatial_shape=padded_spatial_shape,
        )

        # reverse cyclic shift
        if max(shift_shape) > 0:
            x = jnp.roll(
                x, shift=shift_shape, axis=range(1, 1 + num_spatial_dims)
            )

        # unpadding
        if max(spatial_padding) > 0:
            x = jax.lax.dynamic_slice(
                x,
                start_indices=(0,) * (num_spatial_dims + 2),
                slice_sizes=(x.shape[0], *spatial_shape, x.shape[-1]),
            )

        return x


@dataclasses.dataclass
class SwinTransformerEncoderLayer(hk.Module):
    """Multiple Swin Transformer encoder blocks.

    BasicLayer and SwinTransformerV2
    https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
    """

    window_shape: tuple[int, ...]
    num_heads: int
    num_layers: int
    widening_factor: int = 4
    initializer: hk.initializers.Initializer | None = None

    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward.

        Args:
            x: shape (batch, *spatial_shape, model_size).

        Returns:
            List of embeddings,
              shape (batch, *spatial_shape, model_size).
        """
        num_spaital_dims = len(self.window_shape)
        chex.assert_rank(x, 2 + num_spaital_dims)
        if self.num_layers % 2 != 0:
            raise ValueError(
                f"Number of layers have to be >2, got {self.num_layers}"
            )
        model_size = x.shape[-1]
        shift_shape = tuple(ws // 2 for ws in self.window_shape)

        for i in range(self.num_layers):
            # define blocks
            attn_block = SwinMultiHeadAttention(
                window_shape=self.window_shape,
                shift_shape=(0,) * num_spaital_dims
                if i % 2 == 0
                else shift_shape,
                num_heads=self.num_heads,
                widening_factor=self.widening_factor,
                initializer=self.initializer,
            )
            ff_block = MLP(
                emb_size=model_size * self.widening_factor,
                output_size=model_size,
                initializer=self.initializer,
            )

            # forward
            x_attn = attn_block(x)
            x_attn = layer_norm(x_attn)
            x += x_attn

            # feed forward
            x_dense = ff_block(x)
            x_dense = layer_norm(x_dense)
            x += x_dense

        return x


@dataclasses.dataclass
class SwinTransformerEncoder(hk.Module):
    """Swin Transformer Encoder."""

    num_layers: int
    num_heads: int
    num_channels: tuple[int, ...]
    patch_shape: tuple[int, ...]
    window_shape: tuple[int, ...]
    add_position_embedding: bool = True
    widening_factor: int = 4
    scale_factor: int = 2  # down-sample factor
    remat: bool = True  # remat reduces memory cost at cost of compute speed

    def __call__(
        self,
        x: jnp.ndarray,
    ) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
        """Forward.

        Notes:
            Original 3D image shape is (H, W, D).
            After patching, shape is
                (H // patch_size, W // patch_size, D // patch_size),
            denoted by (height, width, depth).
            Therefore, num_patch = height*width*depth
            This is further split into windows of shape (wh, ww, wd).

        Args:
            x: shape (batch, *spatial_shape, in_channels).

        Returns:
            - Transformed down-sampled embeddings.
            - Hidden embeddings after each block.
        """
        num_spatial_dims = len(self.patch_shape)
        chex.assert_rank(x, 2 + num_spatial_dims)
        spatial_shape = x.shape[1:-1]
        for i, (ss, ps) in enumerate(zip(spatial_shape, self.patch_shape)):
            if ss % ps != 0:
                raise ValueError(
                    f"spatial_shape[{i}]={ss} must be divisible by "
                    f"patch_shape[{i}]={ps}"
                )

        # split into patch and encode using conv
        # (batch, *patched_shape, model_size)
        model_size = self.num_channels[0]
        patch_emb = PatchEmbedding(
            patch_shape=self.patch_shape,
            model_size=model_size,
        )
        x = patch_emb(x)

        hidden_embeddings = [x]
        for i, _ in enumerate(self.num_channels):
            # transformer
            # num_channels unchanged
            vit = SwinTransformerEncoderLayer(
                window_shape=self.window_shape,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                widening_factor=self.widening_factor,
            )
            vit = hk.remat(vit) if self.remat else vit
            x = vit(x)

            # down-sampling for non-bottom layers
            # spatial shape get halved by 2**(i+1)
            # if scale_factor = 2
            if i < len(self.num_channels) - 1:
                # down-sample
                # num_channels increased
                x = ConvNDDownSample(
                    num_spatial_dims=num_spatial_dims,
                    out_channels=self.num_channels[i + 1],
                    scale_factor=self.scale_factor,
                    remat=self.remat,
                )(x)

                # save intermediate hidden embeddings
                hidden_embeddings.append(x)

        return layer_norm(x), hidden_embeddings
