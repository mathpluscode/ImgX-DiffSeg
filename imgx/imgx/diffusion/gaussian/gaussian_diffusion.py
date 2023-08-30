"""Gaussian diffusion related functions.

https://github.com/WuJunde/MedSegDiff/blob/master/guided_diffusion/gaussian_diffusion.py
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py
"""
from __future__ import annotations

import enum

import haiku as hk
import jax.numpy as jnp
import jax.random
from absl import logging

from imgx.diffusion.diffusion import Diffusion
from imgx.diffusion.util import extract_and_expand
from imgx.diffusion.variance_schedule import (
    DiffusionBetaSchedule,
    downsample_beta_schedule,
    get_beta_schedule,
)
from imgx.metric.distribution import (
    discretized_gaussian_log_likelihood,
    normal_kl,
)


class GaussianDiffusionModelOutputType(enum.Enum):
    """Class to define model's output meaning.

    - X_START: model predicts x_0.
    - X_PREVIOUS: model predicts x_{t-1}.
    - NOISE: model predicts noise epsilon.
    """

    X_START = enum.auto()
    NOISE = enum.auto()


class GaussianDiffusionModelVarianceType(enum.Enum):
    r"""Class to define p(x_{t-1} | x_t) variance.

    - FIXED_SMALL: a smaller variance,
        \tilde{beta}_t = (1-\bar{alpha}_{t-1})/(1-\bar{alpha}_{t})*beta_t.
    - FIXED_LARGE: a larger variance, beta_t.
    - LEARNED: model outputs an array with channel=2, for mean and variance.
    - LEARNED_RANGE: model outputs an array with channel=2, for mean and
        variance. But the variance is not raw values, it's a coefficient to
        control the value between FIXED_SMALL and FIXED_LARGE.
    """

    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED = enum.auto()
    LEARNED_RANGE = enum.auto()


class GaussianDiffusion(Diffusion):
    # pylint: disable=too-many-public-methods, abstract-method
    """Class for Gaussian diffusion sampling.

    https://github.com/WuJunde/MedSegDiff/blob/master/guided_diffusion/gaussian_diffusion.py
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        model: hk.Module,
        num_timesteps: int,  # T
        num_timesteps_beta: int,
        beta_schedule: str,
        beta_start: float,
        beta_end: float,
        model_out_type: str,
        model_var_type: str,
        **kwargs,  # noqa: ARG002, pylint: disable=unused-argument
    ) -> None:
        """Init.

        q(x_t | x_{t-1}) ~ Normal(sqrt(1-beta_t)*x_{t-1}, beta_t*I)

        Args:
            model: haiku model.
            num_timesteps: number of diffusion steps.
            num_timesteps_beta: number of steps when defining beta schedule.
            beta_schedule: schedule for betas.
            beta_start: beta for t=0.
            beta_end: beta for t=T.
            model_out_type: type of model output.
            model_var_type: type of variance for p(x_{t-1} | x_t).
            sampler: sampler type.
            **kwargs: potential additional arguments.
        """
        super().__init__(
            model=model,
            num_timesteps=num_timesteps,
            noise_fn=jax.random.normal,
        )
        self.num_timesteps_beta = num_timesteps_beta
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.model_out_type = GaussianDiffusionModelOutputType[model_out_type]
        self.model_var_type = GaussianDiffusionModelVarianceType[model_var_type]

        self.set_beta_schedule(
            num_timesteps=num_timesteps,
            num_timesteps_beta=num_timesteps_beta,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.set_diffusion_constants()

    def set_beta_schedule(
        self,
        num_timesteps: int,  # T
        num_timesteps_beta: int,
        beta_schedule: str,
        beta_start: float,
        beta_end: float,
    ) -> None:
        """Set beta schedule.

        Args:
            num_timesteps: down-sampled number of diffusion steps.
            num_timesteps_beta: original number of steps.
            beta_schedule: schedule for betas.
            beta_start: beta for t=0.
            beta_end: beta for t=num_timesteps-1.
        """
        if num_timesteps > num_timesteps_beta:
            raise ValueError(
                f"num_timesteps {num_timesteps} > "
                f"num_timesteps_beta {num_timesteps_beta}."
            )

        # if called multiple times, the schedule will be different
        self.num_timesteps = num_timesteps

        # (num_timesteps_beta,)
        betas = get_beta_schedule(
            num_timesteps=num_timesteps_beta,
            beta_schedule=DiffusionBetaSchedule[beta_schedule],
            beta_start=beta_start,
            beta_end=beta_end,
        )
        # (num_timesteps,)
        self.betas = downsample_beta_schedule(
            betas=betas,
            num_timesteps=num_timesteps_beta,
            num_timesteps_to_keep=num_timesteps,
        )

    def set_diffusion_constants(self) -> None:
        """Set constants with defined."""
        alphas = 1.0 - self.betas  # alpha_t
        self.alphas_cumprod = jnp.cumprod(alphas)  # \bar{alpha}_t
        self.alphas_cumprod_prev = jnp.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = jnp.append(self.alphas_cumprod[1:], 0.0)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jnp.log(1.0 - self.alphas_cumprod)
        # last value is inf as last value of alphas_cumprod is zero
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod_minus_one = jnp.sqrt(
            1.0 / self.alphas_cumprod - 1
        )

        # q(x_{t-1} | x_t, x_0)
        # mean = coeff_start * x_0 + coeff_t * x_t
        # first values are nan
        self.posterior_mean_coeff_start = (
            self.betas
            * jnp.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coeff_t = (
            jnp.sqrt(alphas)
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        # variance
        # log calculation clipped because the posterior variance is 0 at t=0
        # alphas_cumprod_prev has 1.0 appended in front
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        # posterior_variance first value is zero
        self.posterior_log_variance_clipped = jnp.log(
            jnp.append(self.posterior_variance[1], self.posterior_variance[1:])
        )

    def reset_beta_schedule(
        self,
        num_inference_timesteps: int,
    ) -> None:
        """Reset variance schedules for inference.

        Args:
            num_inference_timesteps: number of inference steps.
        """
        if num_inference_timesteps != self.num_timesteps:
            # reset variance schedule if needed
            self.set_beta_schedule(
                num_timesteps=num_inference_timesteps,
                num_timesteps_beta=self.num_timesteps_beta,
                beta_schedule=self.beta_schedule,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
            )
            self.set_diffusion_constants()
            logging.info(
                f"Reset variance schedule to {self.num_timesteps} steps."
            )

    def q_mean_log_variance(
        self, x_start: jnp.ndarray, t_index: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get the distribution q(x_t | x_0).

        Args:
            x_start: noiseless input, shape (batch, ...).
            t_index: storing index values < self.num_timesteps,
                shape (batch, ) or broadcast-compatible to x_start shape.

        Returns:
            mean: shape (batch, ...), expanded axes have dim 1.
            log_variance: shape (batch, ...), expanded axes have dim 1.
        """
        mean = (
            extract_and_expand(
                self.sqrt_alphas_cumprod, t_index=t_index, ndim=x_start.ndim
            )
            * x_start
        )
        log_variance = extract_and_expand(
            self.log_one_minus_alphas_cumprod,
            t_index=t_index,
            ndim=x_start.ndim,
        )
        return mean, log_variance

    def q_posterior_mean(
        self, x_start: jnp.ndarray, x_t: jnp.ndarray, t_index: jnp.ndarray
    ) -> jnp.ndarray:
        """Get mean of the distribution q(x_{t-1} | x_t, x_0).

        Args:
            x_start: noiseless input, shape (batch, ...).
            x_t: noisy input, same shape as x_start.
            t_index: storing index values < self.num_timesteps,
                shape (batch, ) or broadcast-compatible to x_start shape.

        Returns:
            mean: same shape as x_start.
        """
        return (
            extract_and_expand(
                self.posterior_mean_coeff_start,
                t_index=t_index,
                ndim=x_start.ndim,
            )
            * x_start
            + extract_and_expand(
                self.posterior_mean_coeff_t, t_index=t_index, ndim=x_start.ndim
            )
            * x_t
        )

    def q_posterior_mean_variance(
        self, x_start: jnp.ndarray, x_t: jnp.ndarray, t_index: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get the distribution q(x_{t-1} | x_t, x_0).

        Args:
            x_start: noiseless input, shape (batch, ...).
            x_t: noisy input, same shape as x_start.
            t_index: storing index values < self.num_timesteps,
                shape (batch, ) or broadcast-compatible to x_start shape.

        Returns:
            mean: same shape as x_start.
            log_variance: shape (batch, ...), expanded axes have dim 1.
        """
        mean = self.q_posterior_mean(x_start, x_t, t_index)
        log_variance = extract_and_expand(
            self.posterior_log_variance_clipped,
            t_index=t_index,
            ndim=x_start.ndim,
        )
        return mean, log_variance

    def p_log_variance(
        self,
        model_out: jnp.ndarray,
        x_t: jnp.ndarray,
        t_index: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get log_variance of distribution p(x_{t-1} | x_t).

        Args:
            model_out: model predicted output.
                If model estimates variance, the last axis will be split.
            x_t: noisy input, shape (batch, ...).
            t_index: storing index values < self.num_timesteps,
                shape (batch, ) or broadcast-compatible to x_start shape.

        Returns:
            model_out: potentially updated model_out.
            log_variance: broadcast-compatible shape to x_t.
        """
        if (
            self.model_var_type
            == GaussianDiffusionModelVarianceType.FIXED_SMALL
        ):
            log_variance = extract_and_expand(
                self.posterior_log_variance_clipped,
                t_index=t_index,
                ndim=x_t.ndim,
            )
            return model_out, log_variance

        if (
            self.model_var_type
            == GaussianDiffusionModelVarianceType.FIXED_LARGE
        ):
            # TODO why appending?
            variance = jnp.append(self.posterior_variance[1], self.betas[1:])
            log_variance = extract_and_expand(
                jnp.log(variance), t_index=t_index, ndim=x_t.ndim
            )
            return model_out, log_variance

        if self.model_var_type == GaussianDiffusionModelVarianceType.LEARNED:
            model_out, log_variance = jnp.split(
                model_out, indices_or_sections=2, axis=-1
            )
            return model_out, log_variance

        if (
            self.model_var_type
            == GaussianDiffusionModelVarianceType.LEARNED_RANGE
        ):
            # var_coeff are not normalised
            model_out, var_coeff = jnp.split(
                model_out, indices_or_sections=2, axis=-1
            )

            # get min and max of log variance
            log_min_variance = self.posterior_log_variance_clipped
            log_max_variance = jnp.log(self.betas)
            log_min_variance = extract_and_expand(
                log_min_variance, t_index=t_index, ndim=x_t.ndim
            )
            log_max_variance = extract_and_expand(
                log_max_variance, t_index=t_index, ndim=x_t.ndim
            )

            # interpolate between min and max
            var_coeff = jax.nn.sigmoid(var_coeff)  # [0, 1]
            log_variance = (
                var_coeff * log_max_variance
                + (1 - var_coeff) * log_min_variance
            )
            return model_out, log_variance
        raise ValueError(
            f"Unknown DiffusionModelVarianceType {self.model_var_type}."
        )

    def p_mean(
        self,
        model_out: jnp.ndarray,
        x_t: jnp.ndarray,
        t_index: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get mean of distribution p(x_{t-1} | x_t).

        Args:
            model_out: model predicted output.
                If model estimates variance, the last axis will be split.
            x_t: noisy input, shape (batch, ...).
            t_index: storing index values < self.num_timesteps,
                shape (batch, ) or broadcast-compatible to x_start shape.

        Returns:
            x_start: predicted, same shape as x_t.
            mean: same shape as x_t.
        """
        if self.model_out_type == GaussianDiffusionModelOutputType.X_START:
            # q(x_{t-1} | x_t, x_0)
            x_start = self.model_out_to_x(model_out)
            mean = self.q_posterior_mean(
                x_start=x_start, x_t=x_t, t_index=t_index
            )
            return x_start, mean
        if self.model_out_type == GaussianDiffusionModelOutputType.NOISE:
            x_start = self.predict_xstart_from_noise_xt(
                x_t=x_t, noise=model_out, t_index=t_index
            )
            mean = self.q_posterior_mean(
                x_start=x_start, x_t=x_t, t_index=t_index
            )
            return x_start, mean
        raise ValueError(
            f"Unknown DiffusionModelOutputType {self.model_out_type}."
        )

    def p_mean_variance(
        self,
        model_out: jnp.ndarray,
        x_t: jnp.ndarray,
        t_index: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Get the distribution p(x_{t-1} | x_t).

        Args:
            model_out: model predicted output.
                If model estimates variance, the last axis will be split.
            x_t: noisy input, shape (batch, ...).
            t_index: storing index values < self.num_timesteps,
                shape (batch, ) or broadcast-compatible to x_start shape.

        Returns:
            x_start: predicted, same shape as x_t, values are clipped.
            mean: same shape as x_t.
            log_variance: compatible shape to x_t.
        """
        model_out, log_variance = self.p_log_variance(model_out, x_t, t_index)
        x_start, mean = self.p_mean(model_out, x_t, t_index)
        return x_start, mean, log_variance

    def q_sample(
        self,
        x_start: jnp.ndarray,
        noise: jnp.ndarray,
        t_index: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample from q(x_t | x_0).

        Args:
            x_start: noiseless input, shape (batch, ...).
            noise: same shape as x_start.
            t_index: storing index values < self.num_timesteps,
                shape (batch, ) or broadcast-compatible to x_start shape.

        Returns:
            Noisy array with same shape as x_start.
        """
        mean = (
            extract_and_expand(
                self.sqrt_alphas_cumprod, t_index=t_index, ndim=x_start.ndim
            )
            * x_start
        )
        var = extract_and_expand(
            self.sqrt_one_minus_alphas_cumprod,
            t_index=t_index,
            ndim=x_start.ndim,
        )
        x_t = mean + var * noise
        return x_t

    def predict_xprev_from_xstart_xt(
        self, x_start: jnp.ndarray, x_t: jnp.ndarray, t_index: jnp.ndarray
    ) -> jnp.ndarray:
        """Get x_{t-1} from x_0 and x_t.

        The mean of q(x_{t-1} | x_t, x_0) is coeff_start * x_0 + coeff_t * x_t.
        So x_{t-1} = coeff_start * x_0 + coeff_t * x_t.

        Args:
            x_start: noisy input at t, shape (batch, ...).
            x_t: noisy input, same shape as x_start.
            t_index: storing index values < self.num_timesteps, shape (batch, ).

        Returns:
            predicted x_0, same shape as x_prev.
        """
        coeff_start = extract_and_expand(
            self.posterior_mean_coeff_start, t_index, x_t.ndim
        )
        coeff_t = extract_and_expand(
            self.posterior_mean_coeff_t,
            t_index,
            x_t.ndim,
        )
        return coeff_start * x_start + coeff_t * x_t

    def predict_xstart_from_noise_xt(
        self, x_t: jnp.ndarray, noise: jnp.ndarray, t_index: jnp.ndarray
    ) -> jnp.ndarray:
        """Get x_0 from noise epsilon.

        The reparameterization gives:
            x_t = sqrt(alphas_cumprod) * x_0
                + sqrt(1-alphas_cumprod) * epsilon
        so,
        x_0 = 1/sqrt(alphas_cumprod) * x_t
            - sqrt(1-alphas_cumprod)/sqrt(alphas_cumprod) * epsilon
            = 1/sqrt(alphas_cumprod) * x_t
            - sqrt(1/alphas_cumprod - 1) * epsilon

        Args:
            x_t: noisy input, shape (batch, ...).
            noise: noise, shape (batch, ...), expanded axes have dim 1.
            t_index: storing index values < self.num_timesteps,
                shape (batch, ) or broadcast-compatible to x_start shape.

        Returns:
            predicted x_0, same shape as x_t.
        """
        coeff_t = extract_and_expand(
            self.sqrt_recip_alphas_cumprod, t_index=t_index, ndim=x_t.ndim
        )
        coeff_noise = extract_and_expand(
            self.sqrt_recip_alphas_cumprod_minus_one,
            t_index=t_index,
            ndim=x_t.ndim,
        )
        return coeff_t * x_t - coeff_noise * noise

    def predict_noise_from_xstart_xt(
        self, x_t: jnp.ndarray, x_start: jnp.ndarray, t_index: jnp.ndarray
    ) -> jnp.ndarray:
        """Get noise epsilon from x_0 and x_t.

        The reparameterization gives:
            x_t = sqrt(alphas_cumprod) * x_0
                + sqrt(1-alphas_cumprod) * epsilon
        so,
        epsilon = (x_t - sqrt(alphas_cumprod) * x_0) / sqrt(1-alphas_cumprod)
                = (1/sqrt(alphas_cumprod) * x_t - x_0)
                  /sqrt(1/alphas_cumprod-1)

        Args:
            x_t: noisy input, shape (batch, ...).
            x_start: predicted x_0, same shape as x_t.
            t_index: storing index values < self.num_timesteps,
                shape (batch, ) or broadcast-compatible to x_start shape.

        Returns:
            predicted x_0, same shape as x_t.
        """
        coeff_t = extract_and_expand(
            self.sqrt_recip_alphas_cumprod, t_index=t_index, ndim=x_t.ndim
        )
        denominator = extract_and_expand(
            self.sqrt_recip_alphas_cumprod_minus_one,
            t_index=t_index,
            ndim=x_t.ndim,
        )
        return (coeff_t * x_t - x_start) / denominator

    def predict_xstart_from_model_out_xt(
        self,
        model_out: jnp.ndarray,
        x_t: jnp.ndarray,
        t_index: jnp.ndarray,
    ) -> jnp.ndarray:
        """Predict x_0 from model output and x_t.

        Args:
            model_out: model output.
            x_t: noisy input.
            t_index: storing index values < self.num_timesteps.

        Returns:
            x_start, same shape as x_t.
        """
        return self.p_mean(model_out, x_t, t_index)[0]

    def predict_noise_from_model_out_xt(
        self,
        model_out: jnp.ndarray,
        x_t: jnp.ndarray,
        t_index: jnp.ndarray,
    ) -> jnp.ndarray:
        """Get noise from model output and x_t.

        Args:
            model_out: unnormalised values.
            x_t: noisy input.
            t_index: time of shape (...,).

        Returns:
            noise, same shape as x_t.
        """
        if self.model_out_type == GaussianDiffusionModelOutputType.X_START:
            x_start = self.model_out_to_x(model_out)
            return self.predict_noise_from_xstart_xt(
                x_start=x_start, x_t=x_t, t_index=t_index
            )

        if self.model_out_type == GaussianDiffusionModelOutputType.NOISE:
            return model_out

        raise ValueError(
            f"Unknown DiffusionModelOutputType {self.model_out_type}."
        )

    def variational_lower_bound(
        self,
        model_out: jnp.ndarray,
        x_start: jnp.ndarray,
        x_t: jnp.ndarray,
        t_index: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Variational lower-bound, smaller is better.

        For t_index > 0, loss is the KL divergence between
            q(x_{t-1} | x_t, x_0) and p(x_{t-1} | x_t).
        For t_index = 0, loss is q(x_0 | x_t, x_0).

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        Args:
            model_out: model predicted output, may contain variance,
                shape (batch, ...).
            x_start: cleaned, same shape as x_t.
            x_t: noisy input, shape (batch, ...).
            t_index: storing index values < self.num_timesteps,
                shape (batch, ) or broadcast-compatible to x_start shape.

        Returns:
            - lower bounds of shape (batch, ).
            - model_out without variance.
        """
        # split variance from model_out
        # stop-gradient to prevent this loss change mean prediction
        if self.model_var_type in [
            GaussianDiffusionModelVarianceType.LEARNED,
            GaussianDiffusionModelVarianceType.LEARNED_RANGE,
        ]:
            # model_out (batch, ..., num_classes)
            model_out, log_variance = jnp.split(
                model_out, indices_or_sections=2, axis=-1
            )
            # apply a stop-gradient to the mean output for the vlb to prevent
            # this loss change mean prediction
            model_out_vlb = jax.lax.stop_gradient(model_out)
            # model_out (batch, ..., num_classes*2)
            model_out_vlb = jnp.concatenate(
                [model_out_vlb, log_variance], axis=-1
            )
        else:
            model_out_vlb = jax.lax.stop_gradient(model_out)

        # same shape as x_t or broadcast-compatible
        # q(x_{t-1} | x_t, x_0)
        q_mean, q_log_variance = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t_index=t_index
        )
        # p(x_{t-1} | x_t)
        _, p_mean, p_log_variance = self.p_mean_variance(
            model_out=model_out_vlb,
            x_t=x_t,
            t_index=t_index,
        )

        # same shape as x_t or broadcast-compatible
        # if not learning variance, the difference between variance may
        # dominate the kl divergence
        kl = normal_kl(
            q_mean=q_mean,
            q_log_variance=q_log_variance,
            p_mean=p_mean,
            p_log_variance=p_log_variance,
        )
        nll = -discretized_gaussian_log_likelihood(
            x_start, mean=q_mean, log_variance=q_log_variance
        )

        # (batch, )
        reduce_axis = tuple(range(x_t.ndim))[1:]
        kl = jnp.mean(kl, axis=reduce_axis) / jnp.log(2.0)
        nll = jnp.mean(nll, axis=reduce_axis) / jnp.log(2.0)

        # return neg-log-likelihood for t = 0
        return jnp.where(t_index == 0, nll, kl), model_out

    def model_out_to_x(self, model_out: jnp.ndarray) -> jnp.ndarray:
        """Transform model outputs to x space.

        Args:
            model_out: model output without variance.

        Returns:
            Array in the same space as x_start.
        """
        logging.info("Model output and x are assumed to be in the same space")
        return model_out

    def diffusion_loss(
        self,
        x_start: jnp.ndarray,
        x_t: jnp.ndarray,
        t_index: jnp.ndarray,
        noise: jnp.ndarray,
        model_out: jnp.ndarray,
    ) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
        """Diffusion-specific loss function.

        Args:
            x_start: noiseless input.
            x_t: noisy input.
            t_index: storing index values < self.num_timesteps.
            noise: sampled noise, same shape as x_t.
            model_out: model output.

        Returns:
            scalars: dict of losses, each of shape (batch, ).
            model_out: same shape as x_start.
        """
        scalars = {}
        # VLB / ELBO
        # remove potential variance in model_out
        vlb_loss_batch, model_out = self.variational_lower_bound(
            model_out=model_out,
            x_start=x_start,
            x_t=x_t,
            t_index=t_index,
        )
        scalars["vlb_loss"] = vlb_loss_batch

        # mse loss on noise
        noise_pred = self.predict_noise_from_model_out_xt(
            model_out=model_out, x_t=x_t, t_index=t_index
        )
        mse_loss_batch = jnp.mean(
            (noise_pred - noise) ** 2, axis=range(1, noise.ndim)
        )
        scalars["mse_loss"] = mse_loss_batch

        return scalars, model_out
