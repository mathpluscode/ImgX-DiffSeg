"""Module for diffusion segmentation."""
from __future__ import annotations

from collections.abc import Iterator

from jax import numpy as jnp

from imgx.diffusion.diffusion import Diffusion


class DiffusionSegmentation(Diffusion):
    """Base class for segmentation."""

    def mask_to_x(self, mask: jnp.ndarray) -> jnp.ndarray:
        """Convert mask to x.

        Args:
            mask: boolean segmentation mask.

        Returns:
            array in diffusion space.
        """
        raise NotImplementedError

    def x_to_mask(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert x to mask.

        Args:
            x: array in diffusion space.

        Returns:
            boolean segmentation mask.
        """
        raise NotImplementedError

    def x_to_logits(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert x into model output space, which is logits.

        Args:
            x: array in diffusion space.

        Returns:
            unnormalised logits.
        """
        raise NotImplementedError

    def model_out_to_logits_start(
        self, model_out: jnp.ndarray, x_t: jnp.ndarray, t_index: jnp.ndarray
    ) -> jnp.ndarray:
        """Convert model outputs to logits at time 0, noiseless.

        Args:
            model_out: model outputs.
            x_t: noisy x at time t.
            t_index: storing index values < self.num_timesteps.

        Returns:
            logits.
        """
        raise NotImplementedError

    def sample_logits_progressive(
        self,
        image: jnp.ndarray,
        x_t: jnp.ndarray,
        self_conditioning: bool,
    ) -> Iterator[jnp.ndarray]:
        """Generate segmentation mask logits conditioned on image.

        The noise here is defined on segmentation mask.

        Args:
            image: image to be segmented, shape = (batch, ..., C).
            x_t: noisy x at time t.
            self_conditioning: whether to use self conditioning.

        Yields:
            logits.
        """
        batch_size = x_t.shape[0]
        mask_pred = jnp.zeros_like(self.x_to_mask(x_t))
        for t_index_scalar in reversed(range(self.num_timesteps)):
            # (batch, )
            t_index = jnp.full((batch_size,), t_index_scalar, dtype=jnp.int32)
            t = self.t_index_to_t(t_index)
            mask = self.x_to_mask(x_t)
            if self_conditioning:
                mask = jnp.concatenate([mask, mask_pred], axis=-1)
            model_out = self.model(
                image=image,
                mask=mask,
                t=t,
            )
            x_t, x_start = self.sample(
                model_out=model_out,
                x_t=x_t,
                t_index=t_index,
            )
            mask_pred = self.x_to_mask(x_start)
            yield self.x_to_logits(x_start)
