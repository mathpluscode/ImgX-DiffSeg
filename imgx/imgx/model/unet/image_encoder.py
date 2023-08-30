"""Image encoder for unet."""
from __future__ import annotations

import dataclasses

import haiku as hk
import jax.numpy as jnp

from imgx.model.conv import ConvNDDownSample, ConvNDNormAct, ConvNDResBlock


@dataclasses.dataclass
class ImageEncoderUnet(hk.Module):
    """Image encoder module with convolutions for unet."""

    num_spatial_dims: int  # 2 or 3
    num_channels: tuple[int, ...]  # channel at each depth, including the bottom
    patch_size: int = 2  # first down sampling layer
    scale_factor: int = 2  # spatial down-sampling/up-sampling
    num_res_blocks: int = 2  # number of residual blocks
    kernel_size: int = 3  # convolution layer kernel size
    num_heads: int = 8  # for multi head attention/MHA
    widening_factor: int = 4  # for key size in MHA
    remat: bool = True  # remat reduces memory cost at cost of compute speed

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
        # encoder raw input
        conv = ConvNDNormAct(
            num_spatial_dims=self.num_spatial_dims,
            out_channels=self.num_channels[0],
            kernel_size=self.kernel_size,
        )
        conv = hk.remat(conv) if self.remat else conv
        x = conv(x)

        # encoding
        embeddings = [x]
        for i, ch in enumerate(self.num_channels):
            # residual blocks
            # spatial shape get halved by 2**i
            for _ in range(self.num_res_blocks):
                x = ConvNDResBlock(
                    num_spatial_dims=self.num_spatial_dims,
                    out_channels=ch,
                    kernel_size=self.kernel_size,
                    remat=self.remat,
                )(x, t_emb)
                embeddings.append(x)

            # down-sampling for non-bottom layers
            # spatial shape get halved by 2**(i+1)
            if i < len(self.num_channels) - 1:
                scale_factor = self.patch_size if i == 0 else self.scale_factor
                x = ConvNDDownSample(
                    num_spatial_dims=self.num_spatial_dims,
                    out_channels=self.num_channels[i + 1],
                    scale_factor=scale_factor,
                    remat=self.remat,
                )(x)
                embeddings.append(x)

        return embeddings
