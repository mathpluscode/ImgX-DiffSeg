"""Module for sampling."""
from __future__ import annotations

import jax.numpy as jnp

from imgx.diffusion.gaussian.gaussian_diffusion import GaussianDiffusion
from imgx.diffusion.gaussian.gaussian_diffusion_segmentation import (
    GaussianDiffusionSegmentation,
)
from imgx.diffusion.util import expand, extract_and_expand


class DDPMSampler(GaussianDiffusion):
    """DDPM https://arxiv.org/abs/2006.11239."""

    def sample(
        self,
        model_out: jnp.ndarray,
        x_t: jnp.ndarray,
        t_index: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample x_{t-1} ~ p(x_{t-1} | x_t) using DDPM.

        https://arxiv.org/abs/2006.11239

        Args:
            model_out: model predicted output.
                If model estimates variance, the last axis will be split.
            x_t: noisy input, shape (batch, ...).
            t_index: storing index values < self.num_timesteps,
                shape (batch, ) or broadcast-compatible to x_start shape.

        Returns:
            sample: x_{t-1}, same shape as x_t.
            x_start_pred: same shape as x_t.
        """
        x_start_pred, mean, log_variance = self.p_mean_variance(
            model_out=model_out,
            x_t=x_t,
            t_index=t_index,
        )
        noise = self.sample_noise(shape=x_t.shape, dtype=x_t.dtype)

        # no noise when t=0
        # mean + exp(log(sigma**2)/2) * noise = mean + sigma * noise
        nonzero_mask = jnp.array(t_index != 0, dtype=noise.dtype)
        nonzero_mask = expand(nonzero_mask, noise.ndim)
        sample = mean + nonzero_mask * jnp.exp(0.5 * log_variance) * noise

        return sample, x_start_pred


class DDIMSampler(GaussianDiffusion):
    """DDIM https://arxiv.org/abs/2010.02502.

    https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim/pipeline_ddim.py
    """

    def sample(
        self,
        model_out: jnp.ndarray,
        x_t: jnp.ndarray,
        t_index: jnp.ndarray,
        eta: float = 0.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample x_{t-1} ~ p(x_{t-1} | x_t) using DDIM.

        https://arxiv.org/abs/2010.02502

        Args:
            model_out: model predicted output.
                If model estimates variance, the last axis will be split.
            x_t: noisy input, shape (batch, ...).
            t_index: storing index values < self.num_timesteps,
                shape (batch, ) or broadcast-compatible to x_start shape.
            eta: control the noise level in sampling.

        Returns:
            sample: x_{t-1}, same shape as x_t.
            x_start_pred: same shape as x_t.
        """
        # prepare constants
        x_start_pred, _ = self.p_mean(
            model_out=model_out,
            x_t=x_t,
            t_index=t_index,
        )
        noise = self.predict_noise_from_xstart_xt(
            x_t=x_t, x_start=x_start_pred, t_index=t_index
        )
        alphas_cumprod_prev = extract_and_expand(
            self.alphas_cumprod_prev, t_index=t_index, ndim=x_t.ndim
        )
        coeff_start = jnp.sqrt(alphas_cumprod_prev)
        log_variance = (
            extract_and_expand(
                self.posterior_log_variance_clipped,
                t_index=t_index,
                ndim=x_t.ndim,
            )
            * eta
        )
        coeff_noise = jnp.sqrt(1.0 - alphas_cumprod_prev - log_variance**2)
        mean = coeff_start * x_start_pred + coeff_noise * noise

        # deterministic for t_index > 0
        nonzero_mask = jnp.array(t_index != 0, dtype=x_t.dtype)
        nonzero_mask = expand(nonzero_mask, x_t.ndim)

        # sample
        noise = self.sample_noise(shape=x_t.shape, dtype=x_t.dtype)
        sample = mean + nonzero_mask * log_variance * noise
        return sample, x_start_pred


class DDPMSegmentationSampler(GaussianDiffusionSegmentation, DDPMSampler):
    """DDPM for segmentation."""


class DDIMSegmentationSampler(GaussianDiffusionSegmentation, DDIMSampler):
    """DDIM for segmentation."""
