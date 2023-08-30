"""Mask encoder for unet."""
from __future__ import annotations

import dataclasses

import haiku as hk
import jax.lax
import jax.numpy as jnp

from imgx.model.conv import ConvNDResBlock, ConvNDUpSample


def out_head(
    x: jnp.ndarray,
    out_channels: int,
    remat: bool,
) -> jnp.ndarray:
    """Decode the embedding and perform prediction.

    Args:
        x: (batch, *spatial_shape, channel).
        out_channels: number of output channels.
        remat: remat reduces memory cost at cost of compute speed.

    Returns:
        Logits (batch, *spatial_shape, out_channels).
    """
    num_spatial_dims = len(x.shape) - 2
    # (batch, *spatial_shape, out_channels)
    conv = hk.ConvND(
        num_spatial_dims=num_spatial_dims,
        output_channels=out_channels,
        kernel_shape=1,
    )
    conv = hk.remat(conv) if remat else conv
    x = conv(x)

    return x


@dataclasses.dataclass
class MaskDecoderUnet(hk.Module):
    """Mask decoder module with convolutions for unet."""

    num_spatial_dims: int  # 2 or 3
    out_channels: int
    num_channels: tuple[int, ...]  # channel at each depth, including the bottom
    patch_size: int = 2  # first down sampling layer
    scale_factor: int = 2  # spatial down-sampling/up-sampling
    num_res_blocks: int = 2  # number of residual blocks
    kernel_size: int = 3  # convolution layer kernel size
    widening_factor: int = 4  # for key size in MHA
    remat: bool = True  # remat reduces memory cost at cost of compute speed

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
        if (
            len(embeddings)
            != len(self.num_channels) * (self.num_res_blocks + 1) + 1
        ):
            raise ValueError("MaskDecoderConvUnet input length does not match")

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
                x = ConvNDResBlock(
                    num_spatial_dims=self.num_spatial_dims,
                    out_channels=ch,
                    kernel_size=self.kernel_size,
                    remat=self.remat,
                )(x, t_emb)

            if i < len(self.num_channels) - 1:
                # up-sampling
                # skipped.shape <= up-scaled shape
                # as padding may be added when down-sampling
                skipped_shape = embeddings[-1].shape[1:-1]
                # deconv and pad to make emb of same shape as skipped
                scale_factor = (
                    self.patch_size
                    if i == len(self.num_channels) - 2
                    else self.scale_factor
                )
                x = ConvNDUpSample(
                    num_spatial_dims=self.num_spatial_dims,
                    out_channels=self.num_channels[-i - 2],
                    scale_factor=scale_factor,
                    remat=self.remat,
                )(x)
                x = jax.lax.dynamic_slice(
                    x,
                    start_indices=(0,) * (self.num_spatial_dims + 2),
                    slice_sizes=(x.shape[0], *skipped_shape, x.shape[-1]),
                )

        out = out_head(x, out_channels=self.out_channels, remat=self.remat)
        return out
