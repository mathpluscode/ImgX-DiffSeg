"""Image encoder for unet."""
from __future__ import annotations

import dataclasses

import haiku as hk
import jax.numpy as jnp

from imgx.model.conv import ConvNDResBlock
from imgx.model.transformer import TransformerEncoder


@dataclasses.dataclass
class BottomImageEncoderUnet(hk.Module):
    """Image encoder module with convolutions for unet."""

    kernel_size: int = 3  # convolution layer kernel size
    num_heads: int = 8  # for multi head attention
    widening_factor: int = 4  # for key size in MHA
    remat: bool = True  # remat reduces memory cost at cost of compute speed

    def __call__(
        self,
        image_emb: jnp.ndarray,
        t_emb: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Encoder the image.

        Args:
            image_emb: shape (batch, *spatial_shape, model_size).
            t_emb: time embedding, (batch, t_channels).

        Returns:
            image_emb: (batch, *spatial_shape, model_size).
        """
        batch_size, *spatial_shape, model_size = image_emb.shape
        num_spatial_dims = len(spatial_shape)

        # conv before attention
        # image_emb (batch, *spatial_shape, image_emb_size)
        image_emb = ConvNDResBlock(
            num_spatial_dims=num_spatial_dims,
            out_channels=model_size,
            kernel_size=self.kernel_size,
            remat=self.remat,
        )(image_emb, t_emb)

        # attention
        image_emb = image_emb.reshape((batch_size, -1, model_size))
        transformer = TransformerEncoder(
            num_heads=self.num_heads,
            num_layers=1,
            autoregressive=False,
            widening_factor=self.widening_factor,
        )
        image_emb, _ = transformer(image_emb)
        image_emb = image_emb.reshape((batch_size, *spatial_shape, model_size))

        # conv after attention
        # image_emb (batch, *spatial_shape, image_emb_size)
        image_emb = ConvNDResBlock(
            num_spatial_dims=num_spatial_dims,
            out_channels=model_size,
            kernel_size=self.kernel_size,
            remat=self.remat,
        )(image_emb, t_emb)

        return image_emb
