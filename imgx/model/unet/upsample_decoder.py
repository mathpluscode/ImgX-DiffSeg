"""Upsample encoder for unet."""
from __future__ import annotations

import flax.linen as nn
import jax.lax
import jax.numpy as jnp

from imgx.model.conv import ConvResBlock, ConvUpSample


class UpsampleDecoder(nn.Module):
    """Upsample decoder module with convolutions for unet."""

    num_spatial_dims: int  # 2 or 3
    out_channels: int
    num_channels: tuple[int, ...]  # channel at each depth, including the bottom
    patch_size: tuple[int, ...] | int = 2  # first down sampling layer
    scale_factor: tuple[int, ...] | int = 2  # spatial down-sampling/up-sampling
    num_res_blocks: int = 2  # number of residual blocks
    kernel_size: int = 3  # convolution layer kernel size
    widening_factor: int = 4  # for key size in MHA
    out_kernel_init: jax.nn.initializers.Initializer = nn.linear.default_kernel_init
    remat: bool = True  # remat reduces memory cost at cost of compute speed
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        embeddings: list[jnp.ndarray],
        t_emb: jnp.ndarray | None = None,
    ) -> jnp.ndarray | list[jnp.ndarray]:
        """Decode the embedding and perform prediction.

        Args:
            embeddings: list of embeddings from each layer.
                Starting with the first layer.
            t_emb: array of shape (batch, t_channels).

        Returns:
            Logits (batch, ..., out_channels).
        """
        if len(embeddings) != len(self.num_channels) * (self.num_res_blocks + 1) + 1:
            raise ValueError("MaskDecoderConvUnet input length does not match")
        patch_size = self.patch_size
        scale_factor = self.scale_factor
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * self.num_spatial_dims
        if isinstance(scale_factor, int):
            scale_factor = (scale_factor,) * self.num_spatial_dims

        conv_res_block_cls = nn.remat(ConvResBlock) if self.remat else ConvResBlock
        conv_up_sample_cls = nn.remat(ConvUpSample) if self.remat else ConvUpSample
        conv_cls = nn.remat(nn.Conv) if self.remat else nn.Conv

        # spatial shape get halved by 2**(len(self.num_channels)-1)
        # channel = self.num_channels[-1]
        x = embeddings.pop()

        for i, ch in enumerate(self.num_channels[::-1]):
            # spatial shape get halved by 2**(len(self.num_channels)-1-i)
            # channel = ch
            for _ in range(self.num_res_blocks + 1):
                # add skipped
                # use addition instead of concatenation to reduce memory cost
                skipped = embeddings.pop()
                x += skipped

                # conv
                x = conv_res_block_cls(
                    num_spatial_dims=self.num_spatial_dims,
                    out_channels=ch,
                    kernel_size=self.kernel_size,
                )(x, t_emb)

            if i < len(self.num_channels) - 1:
                # up-sampling
                # skipped.shape <= up-scaled shape
                # as padding may be added when down-sampling
                skipped_shape = embeddings[-1].shape[1:-1]
                # deconv and pad to make emb of same shape as skipped
                x = conv_up_sample_cls(
                    out_channels=self.num_channels[-i - 2],
                    scale_factor=patch_size if i == len(self.num_channels) - 2 else scale_factor,
                )(x)
                x = jax.lax.dynamic_slice(
                    x,
                    start_indices=(0,) * (self.num_spatial_dims + 2),
                    slice_sizes=(x.shape[0], *skipped_shape, x.shape[-1]),
                )
        out = conv_cls(
            features=self.out_channels,
            kernel_size=(1,) * self.num_spatial_dims,
            kernel_init=self.out_kernel_init,
        )(x)
        return out
