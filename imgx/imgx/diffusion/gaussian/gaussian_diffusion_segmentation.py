"""Diffusion model for segmentation."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random

from imgx import EPS
from imgx.diffusion.diffusion_segmentation import DiffusionSegmentation
from imgx.diffusion.gaussian.gaussian_diffusion import (
    GaussianDiffusion,
    GaussianDiffusionModelOutputType,
)


class GaussianDiffusionSegmentation(GaussianDiffusion, DiffusionSegmentation):
    # pylint: disable=abstract-method
    """Class for segmentation diffusion sampling.

    x is probabilities scaled in [-1, 1].
    model_out is logits.
    """

    def __init__(  # type: ignore[no-untyped-def]
        self, classes_are_exclusive: bool, **kwargs
    ) -> None:
        """Initialise the class.

        Args:
            classes_are_exclusive: whether classes are exclusive.
            **kwargs: keyword arguments for GaussianDiffusion.
        """
        super().__init__(**kwargs)
        self.classes_are_exclusive = classes_are_exclusive

    def model_out_to_x(self, model_out: jnp.ndarray) -> jnp.ndarray:
        """Convert model outputs to x space.

        Args:
            model_out: unnormalised values,
                classes are assumed to be in the last axis.
                shape = (..., num_classes).

        Returns:
            Probabilities scaled to [-1, 1].
        """
        fn = jax.nn.softmax if self.classes_are_exclusive else jax.nn.sigmoid
        x = fn(model_out)
        x = x * 2.0 - 1.0
        return x

    def mask_to_x(self, mask: jnp.ndarray) -> jnp.ndarray:
        """Convert mask to x.

        Args:
            mask: boolean segmentation mask, shape = (batch, ..., num_classes).

        Returns:
            x, shape = (batch, ..., num_classes), of values in [-1, 1].
        """
        return mask * 2 - 1

    def x_to_mask(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert x to mask.

        Args:
            x: shape = (batch, ..., num_classes), of values in [-1, 1].

        Returns:
            boolean segmentation mask, shape = (batch, ..., num_classes).
        """
        x = jnp.clip(x, -1.0, 1.0)
        return (x + 1) / 2

    def x_to_logits(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert x into model output space, which is logits.

        Args:
            x: probabilities scaled to [-1, 1].

        Returns:
            unnormalised logits.
        """
        probs = (x + 1) / 2
        probs = jnp.clip(probs, EPS, 1.0)
        return jnp.log(probs)

    def model_out_to_logits_start(
        self, model_out: jnp.ndarray, x_t: jnp.ndarray, t_index: jnp.ndarray
    ) -> jnp.ndarray:
        """Convert model outputs to logits at time 0, noiseless.

        Args:
            model_out: unnormalised values,
                classes are assumed to be in the last axis.
                shape = (..., num_classes).
            x_t: label at time t of shape (..., num_classes).
            t_index: time of shape (...,).

        Returns:
            logits, shape = (..., num_classes).
        """
        if self.model_out_type == GaussianDiffusionModelOutputType.X_START:
            # model output is logits
            return model_out
        if self.model_out_type == GaussianDiffusionModelOutputType.NOISE:
            x_start = self.predict_xstart_from_noise_xt(
                x_t=x_t, noise=model_out, t_index=t_index
            )
            return self.x_to_logits(x_start)

        raise ValueError(
            f"Unknown DiffusionModelOutputType {self.model_out_type}."
        )
