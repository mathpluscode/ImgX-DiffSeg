"""Module for diffusion segmentation."""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from imgx.diffusion.diffusion import Diffusion


@dataclass
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
