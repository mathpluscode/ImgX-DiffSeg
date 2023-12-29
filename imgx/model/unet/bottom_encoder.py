"""Image encoder for unet."""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from imgx.model.attention import TransformerEncoder
from imgx.model.conv import ConvResBlock


class BottomImageEncoderUnet(nn.Module):
    """Image encoder module with convolutions for unet."""

    kernel_size: tuple[int, ...]  # convolution layer kernel size
    dropout: float = 0.0  # for resnet block
    num_heads: int = 8  # for multi head attention
    num_layers: int = 1  # for transformer encoder
    widening_factor: int = 4  # for key size in MHA
    remat: bool = True  # reduces memory cost at cost of compute speed
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        is_train: bool,
        image_emb: jnp.ndarray,
        t_emb: jnp.ndarray | None,
    ) -> jnp.ndarray:
        """Encoder the image.

        Args:
            is_train: whether in training mode.
            image_emb: shape (batch, *spatial_shape, model_size).
            t_emb: time embedding, (batch, t_channels).

        Returns:
            image_emb: (batch, *spatial_shape, model_size).
        """
        model_size = image_emb.shape[-1]

        # conv before attention
        # image_emb (batch, *spatial_shape, image_emb_size)
        image_emb = ConvResBlock(
            out_channels=model_size,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            remat=self.remat,
        )(is_train, image_emb, t_emb)

        # attention
        image_emb = TransformerEncoder(
            num_heads=self.num_heads,
            widening_factor=self.widening_factor,
            dropout=self.dropout,
            remat=self.remat,
            dtype=self.dtype,
        )(is_train, image_emb)

        # conv after attention
        # image_emb (batch, *spatial_shape, image_emb_size)
        image_emb = ConvResBlock(
            out_channels=model_size,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            remat=self.remat,
        )(is_train, image_emb, t_emb)

        return image_emb
