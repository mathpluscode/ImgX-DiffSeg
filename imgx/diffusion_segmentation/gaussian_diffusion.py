"""Diffusion model for segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jax.random

from imgx import EPS
from imgx.diffusion.gaussian.gaussian_diffusion import (
    GaussianDiffusion,
    get_gaussian_diffusion_attributes,
)
from imgx.diffusion.gaussian.sampler import DDIMSampler, DDPMSampler
from imgx.diffusion_segmentation.diffusion import DiffusionSegmentation


@dataclass
class GaussianDiffusionSegmentation(GaussianDiffusion, DiffusionSegmentation):
    # pylint: disable=abstract-method
    """Class for segmentation diffusion sampling.

    x is probabilities scaled in [-1, 1].
    model_out is logits.
    """
    classes_are_exclusive: bool

    @classmethod
    def create(  # type: ignore[no-untyped-def]
        cls: type[GaussianDiffusionSegmentation],
        num_timesteps: int,  # T
        num_timesteps_beta: int,
        beta_schedule: str,
        beta_start: float,
        beta_end: float,
        model_out_type: str,
        model_var_type: str,
        **kwargs,
    ) -> GaussianDiffusionSegmentation:
        """Create a new instance.

        Args:
            num_timesteps: number of diffusion steps.
            num_timesteps_beta: number of steps when defining beta schedule.
            beta_schedule: schedule for betas.
            beta_start: beta for t=0.
            beta_end: beta for t=T.
            model_out_type: type of model output.
            model_var_type: type of variance for p(x_{t-1} | x_t).
            kwargs: arguments, including classes_are_exclusive.

        Returns:
            Instance of GaussianDiffusionSegmentation.
        """
        # sanity check for string variables
        if model_out_type not in ["x_start", "noise"]:
            raise ValueError(
                f"Unknown DiffusionModelOutputType {model_out_type}, should be x_start or noise."
            )
        if model_var_type not in [
            "fixed_small",
            "fixed_large",
            "learned",
            "learned_range",
        ]:
            raise ValueError(
                f"Unknown DiffusionModelVarianceType {model_var_type},"
                f"should be fixed_small, fixed_large, learned or learned_range."
            )

        # set variance schedule
        attr_dict = get_gaussian_diffusion_attributes(
            num_timesteps=num_timesteps,
            num_timesteps_beta=num_timesteps_beta,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        return cls(
            num_timesteps=num_timesteps,
            noise_fn=jax.random.normal,
            num_timesteps_beta=num_timesteps_beta,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            model_out_type=model_out_type,
            model_var_type=model_var_type,
            **attr_dict,
            **kwargs,
        )

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
        if self.model_out_type == "x_start":
            # model output is logits
            return model_out
        if self.model_out_type == "noise":
            x_start = self.predict_xstart_from_noise_xt(x_t=x_t, noise=model_out, t_index=t_index)
            return self.x_to_logits(x_start)

        raise ValueError(f"Unknown DiffusionModelOutputType {self.model_out_type}.")


@dataclass
class DDPMSegmentationSampler(GaussianDiffusionSegmentation, DDPMSampler):
    """DDPM for segmentation."""


@dataclass
class DDIMSegmentationSampler(GaussianDiffusionSegmentation, DDIMSampler):
    """DDIM for segmentation."""
