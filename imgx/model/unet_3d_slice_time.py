"""UNet for segmentation."""
import dataclasses
from typing import Callable, List, Tuple

import haiku as hk
import jax
from jax import numpy as jnp

from imgx.model.basic import instance_norm, sinusoidal_positional_embedding
from imgx.model.unet_3d_slice import Conv2dNormAct, Conv2dPool


@dataclasses.dataclass
class TimeConv2dResBlock(hk.Module):
    """Conv2dResBlock with time embedding input.

    This class is defined separately to use remat, as remat does not allow
    condition loop (if / else).

    https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
    """

    out_channels: int
    kernel_size: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu

    def __call__(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: tensor to be up-sampled, (batch, w, h, in_channels).
            t: time embedding, (batch, t_channels).

        Returns:
            Tensor.
        """
        res = x
        x = hk.Conv2D(
            output_channels=self.out_channels,
            kernel_shape=self.kernel_size,
            with_bias=False,
        )(x)
        x = instance_norm(x)
        x = self.activation(x)
        x = hk.Conv2D(
            output_channels=self.out_channels,
            kernel_shape=self.kernel_size,
            with_bias=False,
        )(x)
        t = self.activation(t[:, None, None, :])
        t = hk.Linear(output_size=self.out_channels)(t)
        x += t
        x = instance_norm(x)
        x = self.activation(x + res)
        return x


@dataclasses.dataclass
class Unet3dSliceTime(hk.Module):
    """2D UNet for 3D images.

    https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/basic_unet.py
    """

    in_shape: Tuple[int, int, int]  # spatial shape
    in_channels: int  # input channels
    out_channels: int
    num_channels: Tuple[int, ...]  # channel at each depth, including the bottom
    num_timesteps: int  # T
    kernel_size: int = 3
    scale_factor: int = 2  # spatial down-sampling/up-sampling
    remat: bool = False  # remat reduces memory cost at cost of compute speed

    def encoder(
        self,
        image: jnp.ndarray,
        t: jnp.ndarray,
    ) -> List[jnp.ndarray]:
        """Encoder the image.

        Args:
            image: image tensor of shape (batch, H, W, in_channels).
            t: time embedding of shape (batch, t_channels).

        Returns:
            List of embeddings from each layer.
        """
        conv = Conv2dNormAct(
            out_channels=self.num_channels[0],
            kernel_size=self.kernel_size,
        )
        conv = hk.remat(conv) if self.remat else conv
        emb = conv(image)
        conv_t = TimeConv2dResBlock(
            out_channels=self.num_channels[0],
            kernel_size=self.kernel_size,
        )
        conv_t = hk.remat(conv_t) if self.remat else conv_t
        emb = conv_t(x=emb, t=t)

        embeddings = [emb]
        for ch in self.num_channels:
            conv = Conv2dPool(out_channels=ch, scale_factor=self.scale_factor)
            conv = hk.remat(conv) if self.remat else conv
            emb = conv(emb)

            conv_t = TimeConv2dResBlock(
                out_channels=ch,
                kernel_size=self.kernel_size,
            )
            conv_t = hk.remat(conv_t) if self.remat else conv_t
            emb = conv_t(x=emb, t=t)

            embeddings.append(emb)

        return embeddings

    def decoder(
        self,
        embeddings: List[jnp.ndarray],
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Decode the embedding and perform prediction.

        Args:
            embeddings: list of embeddings from each layer.
                Starting with the first layer.
            t: time embedding of shape (batch, t_channels).

        Returns:
            Unnormalized logits.
        """
        if len(embeddings) != len(self.num_channels) + 1:
            raise ValueError("UNet decoder input length does not match")
        emb = embeddings[-1]
        # calculate up-sampled channel
        # [32, 64, 128, 256] -> [32, 32, 64, 128]
        channels = self.num_channels[:1] + self.num_channels[:-1]
        for i, ch in enumerate(channels[::-1]):
            # skipped.shape <= up-scaled shape
            # as padding may be added when down-sampling
            skipped = embeddings[-i - 2]
            skipped_shape = skipped.shape[-3:-1]
            # deconv and pad to make emb of same shape as skipped
            conv = hk.Conv2DTranspose(
                output_channels=ch,
                kernel_shape=self.scale_factor,
                stride=self.scale_factor,
            )
            conv = hk.remat(conv) if self.remat else conv
            emb = conv(emb)
            emb = emb[
                ...,
                : skipped_shape[0],
                : skipped_shape[1],
                :,
            ]
            # add skipped
            emb += skipped
            # conv
            conv_t = TimeConv2dResBlock(
                out_channels=ch,
                kernel_size=self.kernel_size,
            )
            conv_t = hk.remat(conv_t) if self.remat else conv_t
            emb = conv_t(emb, t)

        conv = hk.Conv2D(output_channels=self.out_channels, kernel_shape=1)
        conv = hk.remat(conv) if self.remat else conv
        out = conv(emb)
        return out

    def __call__(  # type: ignore[no-untyped-def]
        self,
        image: jnp.ndarray,
        t: jnp.ndarray,
        **kwargs,  # noqa: ARG002
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            image: (batch, h, w, d, in_channels).
            t: (batch, ).
            kwargs: unused arguments.

        Returns:
            Predictions (batch, h, w, d, out_channels).

        Raises:
            ValueError: if input shape does not match.
        """
        if image.shape[-4:] != (*self.in_shape, self.in_channels):
            raise ValueError(
                f"Input shape {image.shape[-4:]} does not match"
                f" configs {(*self.in_shape, self.in_channels)}"
            )

        # (batch, h, w, d, in_channels) -> (batch, d, h, w, in_channels)
        image = jnp.transpose(image, (0, 3, 1, 2, 4))
        # (batch, d, h, w, in_channels) -> (batch*d, h, w, in_channels)
        image = jnp.reshape(image, (-1, *self.in_shape[:2], self.in_channels))
        # (batch, ) -> (batch*d,)
        t = jnp.repeat(t, repeats=self.in_shape[2], axis=0)

        dim_t = self.num_channels[0] * 4
        t = sinusoidal_positional_embedding(x=t, dim=dim_t)
        embeddings = self.encoder(image=image, t=t)
        out = self.decoder(embeddings=embeddings, t=t)

        # (batch*d, h, w, out_channels) -> (batch, d, h, w, out_channels)
        out = jnp.reshape(
            out, (-1, self.in_shape[2], *self.in_shape[:2], self.out_channels)
        )
        # (batch, d, h, w, out_channels) -> (batch, h, w, d, out_channels)
        out = jnp.transpose(out, (0, 2, 3, 1, 4))
        return out
