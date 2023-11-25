"""UNet for segmentation."""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from imgx.model.basic import sinusoidal_positional_embedding
from imgx.model.slice import merge_spatial_dim_into_batch, split_spatial_dim_from_batch
from imgx.model.unet.bottom_encoder import BottomImageEncoderUnet
from imgx.model.unet.downsample_encoder import DownsampleEncoder
from imgx.model.unet.upsample_decoder import UpsampleDecoder


class Unet(nn.Module):
    """UNet with optional mask and timesteps inputs.

    https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
    """

    num_spatial_dims: int  # 2 or 3
    out_channels: int
    num_channels: tuple[int, ...]  # channel at each depth, including the bottom
    patch_size: int = 2  # first down sampling layer
    scale_factor: int = 2  # spatial down-sampling/up-sampling
    num_res_blocks: int = 2  # number of residual blocks
    kernel_size: int = 3  # convolution layer kernel size
    num_heads: int = 8  # for multi head attention/MHA
    widening_factor: int = 4  # for key size in MHA
    num_transform_layers: int = 1  # for transformer encoder
    out_kernel_init: jax.nn.initializers.Initializer = nn.linear.default_kernel_init
    remat: bool = True  # remat reduces memory cost at cost of compute speed
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        image: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        t: jnp.ndarray | None = None,
    ) -> jnp.ndarray | list[jnp.ndarray]:
        """Forward pass.

        For vanilla UNet, mask and t are None.

        Args:
            image: (batch, *spatial_shape, image_channels).
            mask: (batch, *spatial_shape, mask_channels).
            t: (batch, ), values in [0, 1).

        Returns:
            Logits (batch, ..., out_channels).

        Raises:
            ValueError: if input shape does not match.
        """
        if image.ndim < self.num_spatial_dims + 2:
            raise ValueError(
                f"Input image has shape {image.shape},"
                f"but num_spatial_dims = {self.num_spatial_dims}."
            )
        if (mask is None) != (t is None):
            raise ValueError("mask and t must be both None or both not None.")

        # concatenate image and mask
        if mask is not None:
            image = jnp.concatenate([image, mask], axis=-1)

        # merge spatial dimensions into batch if needed
        merge_spatial_dim = image.ndim > self.num_spatial_dims + 2
        if merge_spatial_dim:
            batch_size = image.shape[0]
            spatial_shape = image.shape[1:-1]
            image = merge_spatial_dim_into_batch(x=image, num_spatial_dims=self.num_spatial_dims)
            # time steps needs to be expanded too
            if t is not None:
                t = jnp.repeat(t, repeats=image.shape[0] // batch_size, axis=0)

        # time encoder
        t_emb = None
        if t is not None:
            t_emb = sinusoidal_positional_embedding(
                x=t, dim=self.num_channels[-1], dtype=image.dtype
            )

        # image encoder
        embeddings = DownsampleEncoder(
            num_spatial_dims=self.num_spatial_dims,
            num_channels=self.num_channels,
            patch_size=self.patch_size,
            scale_factor=self.scale_factor,
            num_res_blocks=self.num_res_blocks,
            kernel_size=self.kernel_size,
            num_heads=self.num_heads,
            widening_factor=self.widening_factor,
            remat=self.remat,
            dtype=self.dtype,
        )(x=image, t_emb=t_emb)
        image_emb = embeddings[-1]

        # bottom encoder
        image_emb = BottomImageEncoderUnet(
            kernel_size=self.kernel_size,
            num_heads=self.num_heads,
            widening_factor=self.widening_factor,
            num_layers=self.num_transform_layers,
            remat=self.remat,
            dtype=self.dtype,
        )(image_emb=image_emb, t_emb=t_emb)
        embeddings.append(image_emb)

        # mask decoder
        out = UpsampleDecoder(
            num_spatial_dims=self.num_spatial_dims,
            out_channels=self.out_channels,
            num_channels=self.num_channels,
            patch_size=self.patch_size,
            scale_factor=self.scale_factor,
            num_res_blocks=self.num_res_blocks,
            kernel_size=self.kernel_size,
            widening_factor=self.widening_factor,
            out_kernel_init=self.out_kernel_init,
            remat=self.remat,
            dtype=self.dtype,
        )(embeddings=embeddings, t_emb=t_emb)

        # split batch back to original spatial dimensions
        if merge_spatial_dim:
            return split_spatial_dim_from_batch(
                x=out,
                num_spatial_dims=self.num_spatial_dims,
                batch_size=batch_size,
                spatial_shape=spatial_shape,
            )
        return out
