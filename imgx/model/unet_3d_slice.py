"""UNet for segmentation."""
import dataclasses
from typing import Callable, List, Tuple

import haiku as hk
import jax
from jax import numpy as jnp

from imgx.model.basic import instance_norm


@dataclasses.dataclass
class Conv2dNormAct(hk.Module):
    """Block with conv2d-norm-act."""

    out_channels: int
    kernel_size: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu

    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: tensor to be up-sampled.

        Returns:
            Tensor.
        """
        x = hk.Conv2D(
            output_channels=self.out_channels,
            kernel_shape=self.kernel_size,
            with_bias=False,
        )(x)
        x = instance_norm(x)
        x = self.activation(x)
        return x


@dataclasses.dataclass
class Conv2dResBlock(hk.Module):
    """Block with two conv2d-norm-act layers and residual link."""

    out_channels: int
    kernel_size: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu

    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: tensor to be up-sampled.

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
        x = instance_norm(x)
        x = self.activation(x + res)
        return x


@dataclasses.dataclass
class Conv2dPool(hk.Module):
    """Patch merging, a down-sample layer."""

    out_channels: int
    scale_factor: int

    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward.

        Args:
            x: shape (batch, h, w, d, in_channels).

        Returns:
            Down-sampled array.
        """
        x = hk.Conv2D(
            output_channels=self.out_channels,
            kernel_shape=self.scale_factor,
            stride=self.scale_factor,
            with_bias=False,
        )(x)
        x = instance_norm(x)
        return x


@dataclasses.dataclass
class Unet3dSlice(hk.Module):
    """2D UNet for 3D images.

    https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/basic_unet.py
    """

    in_shape: Tuple[int, int, int]  # spatial shape
    in_channels: int  # input channels
    out_channels: int
    num_channels: Tuple[int, ...]  # channel at each depth, including the bottom
    kernel_size: int = 3
    scale_factor: int = 2  # spatial down-sampling/up-sampling
    remat: bool = False  # remat reduces memory cost at cost of compute speed

    def encoder(
        self,
        image: jnp.ndarray,
    ) -> List[jnp.ndarray]:
        """Encoder the image.

        Args:
            image: image tensor of shape (batch, H, W, C).

        Returns:
            List of embeddings from each layer.
        """
        conv = hk.Sequential(
            [
                Conv2dNormAct(
                    out_channels=self.num_channels[0],
                    kernel_size=self.kernel_size,
                ),
                Conv2dResBlock(
                    out_channels=self.num_channels[0],
                    kernel_size=self.kernel_size,
                ),
            ]
        )
        conv = hk.remat(conv) if self.remat else conv
        emb = conv(image)
        embeddings = [emb]
        for ch in self.num_channels:
            conv = hk.Sequential(
                [
                    Conv2dPool(out_channels=ch, scale_factor=self.scale_factor),
                    Conv2dResBlock(
                        out_channels=ch,
                        kernel_size=self.kernel_size,
                    ),
                ]
            )
            conv = hk.remat(conv) if self.remat else conv
            emb = conv(emb)
            embeddings.append(emb)

        return embeddings

    def decoder(
        self,
        embeddings: List[jnp.ndarray],
    ) -> jnp.ndarray:
        """Decode the embedding and perform prediction.

        Args:
            embeddings: list of embeddings from each layer.
                Starting with the first layer.

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
            conv = Conv2dResBlock(
                out_channels=ch,
                kernel_size=self.kernel_size,
            )
            conv = hk.remat(conv) if self.remat else conv
            emb = conv(emb)

        conv = hk.Conv2D(output_channels=self.out_channels, kernel_shape=1)
        conv = hk.remat(conv) if self.remat else conv
        out = conv(emb)
        return out

    def __call__(  # type: ignore[no-untyped-def]
        self,
        image: jnp.ndarray,
        **kwargs,  # noqa: ARG002
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            image: (batch, h, w, d, in_channels).
            kwargs: unused arguments.

        Returns:
            Predictions (batch, h, w, d, out_channels).

        Raises:
            ValueError: if input shape does not match.
        """
        if image.shape[-4:] != (*self.in_shape, self.in_channels):
            raise ValueError(
                f"Input shape {image.shape[-3:]} does not match"
                f" configs {(*self.in_shape, self.in_channels)}"
            )

        # (batch, h, w, d, in_channels) -> (batch, d, h, w, in_channels)
        image = jnp.transpose(image, (0, 3, 1, 2, 4))
        # (batch, d, h, w, in_channels) -> (batch*d, h, w, in_channels)
        image = jnp.reshape(image, (-1, *self.in_shape[:2], self.in_channels))

        embeddings = self.encoder(image=image)
        out = self.decoder(embeddings=embeddings)

        # (batch*d, h, w, out_channels) -> (batch, d, h, w, out_channels)
        out = jnp.reshape(
            out, (-1, self.in_shape[2], *self.in_shape[:2], self.out_channels)
        )
        # (batch, d, h, w, out_channels) -> (batch, h, w, d, out_channels)
        out = jnp.transpose(out, (0, 2, 3, 1, 4))
        return out
