"""Image encoder for unet."""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from imgx.model.conv import ConvDownSample, ConvNormAct, ConvResBlock


class ImageEncoderUnet(nn.Module):
    """Image encoder module with convolutions for unet."""

    num_spatial_dims: int  # 2 or 3
    num_channels: tuple[int, ...]  # channel at each depth, including the bottom
    patch_size: int = 2  # first down sampling layer
    scale_factor: int = 2  # spatial down-sampling/up-sampling
    num_res_blocks: int = 2  # number of residual blocks
    kernel_size: int = 3  # convolution layer kernel size
    num_heads: int = 8  # for multi head attention/MHA
    widening_factor: int = 4  # for key size in MHA
    remat: bool = True  # reduces memory cost at cost of compute speed
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        t_emb: jnp.ndarray | None = None,
    ) -> list[jnp.ndarray]:
        """Encoder the image.

        If batch_size = 2, image_shape = (256, 256, 32), num_channels = (1,2,4)
        with num_res_blocks = 2, patch_size = 4
        the embeddings' shape are:
            (2, 256, 256, 32, 1), from first residual block before for loop
            (2, 256, 256, 32, 1), from residual block, i=0
            (2, 256, 256, 32, 1), from residual block, i=0
            (2, 64, 64, 8, 2), from down-sampling block, i=0
            (2, 64, 64, 8, 2), from residual block, i=1
            (2, 64, 64, 8, 2), from residual block, i=1
            (2, 32, 32, 4, 4), from down-sampling block, i=1
            (2, 32, 32, 4, 4), from residual block, i=2
            (2, 32, 32, 4, 4), from residual block, i=2

        Args:
            x: array of shape (batch, *spatial_shape, in_channels).
            t_emb: array of shape (batch, t_channels).

        Returns:
            List of embeddings from each layer.
        """
        conv_norm_act_cls = nn.remat(ConvNormAct) if self.remat else ConvNormAct
        conv_res_block_cls = nn.remat(ConvResBlock) if self.remat else ConvResBlock
        conv_down_sample_cls = nn.remat(ConvDownSample) if self.remat else ConvDownSample
        # encoder raw input
        x = conv_norm_act_cls(
            num_spatial_dims=self.num_spatial_dims,
            out_channels=self.num_channels[0],
            kernel_size=self.kernel_size,
        )(x)

        # encoding
        embeddings = [x]
        for i, ch in enumerate(self.num_channels):
            # residual blocks
            # spatial shape get halved by 2**i
            for _ in range(self.num_res_blocks):
                x = conv_res_block_cls(
                    num_spatial_dims=self.num_spatial_dims,
                    out_channels=ch,
                    kernel_size=self.kernel_size,
                )(x, t_emb)
                embeddings.append(x)

            # down-sampling for non-bottom layers
            # spatial shape get halved by 2**(i+1)
            if i < len(self.num_channels) - 1:
                scale_factor = self.patch_size if i == 0 else self.scale_factor
                x = conv_down_sample_cls(
                    num_spatial_dims=self.num_spatial_dims,
                    out_channels=self.num_channels[i + 1],
                    scale_factor=scale_factor,
                )(x)
                embeddings.append(x)

        return embeddings
