"""Gaussian diffusion related functions.

https://github.com/WuJunde/MedSegDiff/blob/master/guided_diffusion/gaussian_diffusion.py
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jax.random
from absl import logging

from imgx.diffusion.diffusion import Diffusion
from imgx.diffusion.gaussian.variance_schedule import downsample_beta_schedule, get_beta_schedule
from imgx.diffusion.util import extract_and_expand
from imgx.metric.distribution import discretized_gaussian_log_likelihood, normal_kl


def get_gaussian_diffusion_attributes(
    num_timesteps: int,  # T
    num_timesteps_beta: int,
    beta_schedule: str,
    beta_start: float,
    beta_end: float,
) -> dict[str, jnp.ndarray]:
    """Setup variance schedule and create instance.

    Args:
        num_timesteps: number of diffusion steps.
        num_timesteps_beta: number of steps when defining beta schedule.
        beta_schedule: schedule for betas.
        beta_start: beta for t=0.
        beta_end: beta for t=T.

    Returns:
        Dict of attributes.
    """
    if num_timesteps > num_timesteps_beta:
        raise ValueError(
            f"num_timesteps {num_timesteps} > num_timesteps_beta {num_timesteps_beta}."
        )
    # set variance schedule
    # (num_timesteps_beta,)
    betas = get_beta_schedule(
        num_timesteps=num_timesteps_beta,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
    )
    # (num_timesteps,)
    betas = downsample_beta_schedule(
        betas=betas,
        num_timesteps=num_timesteps_beta,
        num_timesteps_to_keep=num_timesteps,
    )

    # Set constants with defined.
    alphas = 1.0 - betas  # alpha_t
    alphas_cumprod = jnp.cumprod(alphas)  # \bar{alpha}_t
    alphas_cumprod_prev = jnp.append(1.0, alphas_cumprod[:-1])
    alphas_cumprod_next = jnp.append(alphas_cumprod[1:], 0.0)
    sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
    log_one_minus_alphas_cumprod = jnp.log(1.0 - alphas_cumprod)
    # last value is inf as last value of alphas_cumprod is zero
    sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod)
    sqrt_recip_alphas_cumprod_minus_one = jnp.sqrt(1.0 / alphas_cumprod - 1)

    # q(x_{t-1} | x_t, x_0)
    # mean = coeff_start * x_0 + coeff_t * x_t
    # first values are nan
    posterior_mean_coeff_start = betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coeff_t = jnp.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    # variance
    # log calculation clipped because the posterior variance is 0 at t=0
    # alphas_cumprod_prev has 1.0 appended in front
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    # posterior_variance first value is zero
    posterior_log_variance_clipped = jnp.log(
        jnp.append(posterior_variance[1], posterior_variance[1:])
    )

    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "alphas_cumprod_next": alphas_cumprod_next,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "log_one_minus_alphas_cumprod": log_one_minus_alphas_cumprod,
        "sqrt_recip_alphas_cumprod": sqrt_recip_alphas_cumprod,
        "sqrt_recip_alphas_cumprod_minus_one": sqrt_recip_alphas_cumprod_minus_one,
        "posterior_mean_coeff_start": posterior_mean_coeff_start,
        "posterior_mean_coeff_t": posterior_mean_coeff_t,
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": posterior_log_variance_clipped,
    }


@dataclass
class GaussianDiffusion(Diffusion):
    # pylint: disable=too-many-public-methods, abstract-method
    """Class for Gaussian diffusion sampling.

    https://github.com/WuJunde/MedSegDiff/blob/master/guided_diffusion/gaussian_diffusion.py
    """
    # additional config to Diffusion
    num_timesteps_beta: int  # number of steps when defining beta schedule
    beta_schedule: str
    beta_start: float
    beta_end: float
    model_out_type: str  # x_start, noise
    model_var_type: str  # fixed_small, fixed_large, learned, learned_range
    # variables
    betas: jnp.ndarray
    alphas_cumprod: jnp.ndarray
    alphas_cumprod_prev: jnp.ndarray
    alphas_cumprod_next: jnp.ndarray
    sqrt_alphas_cumprod: jnp.ndarray
    sqrt_one_minus_alphas_cumprod: jnp.ndarray
    log_one_minus_alphas_cumprod: jnp.ndarray
    sqrt_recip_alphas_cumprod: jnp.ndarray
    sqrt_recip_alphas_cumprod_minus_one: jnp.ndarray
    posterior_mean_coeff_start: jnp.ndarray
    posterior_mean_coeff_t: jnp.ndarray
    posterior_variance: jnp.ndarray
    posterior_log_variance_clipped: jnp.ndarray

    @classmethod
    def create(
        cls: type[GaussianDiffusion],
        num_timesteps: int,  # T
        num_timesteps_beta: int,
        beta_schedule: str,
        beta_start: float,
        beta_end: float,
        model_out_type: str,
        model_var_type: str,
    ) -> GaussianDiffusion:
        """Setup variance schedule and create instance.

        Args:
            num_timesteps: number of diffusion steps.
            num_timesteps_beta: number of steps when defining beta schedule.
            beta_schedule: schedule for betas.
            beta_start: beta for t=0.
            beta_end: beta for t=T.
            model_out_type: type of model output.
            model_var_type: type of variance for p(x_{t-1} | x_t).

        Returns:
            Instance of GaussianDiffusion.
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
            extract_and_expand(self.sqrt_alphas_cumprod, t_index=t_index, ndim=x_start.ndim)
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
            + extract_and_expand(self.posterior_mean_coeff_t, t_index=t_index, ndim=x_start.ndim)
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
        if self.model_var_type == "fixed_small":
            log_variance = extract_and_expand(
                self.posterior_log_variance_clipped,
                t_index=t_index,
                ndim=x_t.ndim,
            )
            return model_out, log_variance

        if self.model_var_type == "fixed_large":
            # TODO why appending?
            variance = jnp.append(self.posterior_variance[1], self.betas[1:])
            log_variance = extract_and_expand(jnp.log(variance), t_index=t_index, ndim=x_t.ndim)
            return model_out, log_variance

        if self.model_var_type == "learned":
            model_out, log_variance = jnp.split(model_out, indices_or_sections=2, axis=-1)
            return model_out, log_variance

        if self.model_var_type == "learned_range":
            # var_coeff are not normalised
            model_out, var_coeff = jnp.split(model_out, indices_or_sections=2, axis=-1)

            # get min and max of log variance
            log_min_variance = self.posterior_log_variance_clipped
            log_max_variance = jnp.log(self.betas)
            log_min_variance = extract_and_expand(log_min_variance, t_index=t_index, ndim=x_t.ndim)
            log_max_variance = extract_and_expand(log_max_variance, t_index=t_index, ndim=x_t.ndim)

            # interpolate between min and max
            var_coeff = jax.nn.sigmoid(var_coeff)  # [0, 1]
            log_variance = var_coeff * log_max_variance + (1 - var_coeff) * log_min_variance
            return model_out, log_variance
        raise ValueError(f"Unknown DiffusionModelVarianceType {self.model_var_type}.")

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
        if self.model_out_type == "x_start":
            # q(x_{t-1} | x_t, x_0)
            x_start = self.model_out_to_x(model_out)
            mean = self.q_posterior_mean(x_start=x_start, x_t=x_t, t_index=t_index)
            return x_start, mean
        if self.model_out_type == "noise":
            x_start = self.predict_xstart_from_noise_xt(x_t=x_t, noise=model_out, t_index=t_index)
            mean = self.q_posterior_mean(x_start=x_start, x_t=x_t, t_index=t_index)
            return x_start, mean
        raise ValueError(f"Unknown DiffusionModelOutputType {self.model_out_type}.")

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
            extract_and_expand(self.sqrt_alphas_cumprod, t_index=t_index, ndim=x_start.ndim)
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
        coeff_start = extract_and_expand(self.posterior_mean_coeff_start, t_index, x_t.ndim)
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
        coeff_t = extract_and_expand(self.sqrt_recip_alphas_cumprod, t_index=t_index, ndim=x_t.ndim)
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
        coeff_t = extract_and_expand(self.sqrt_recip_alphas_cumprod, t_index=t_index, ndim=x_t.ndim)
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
        if self.model_out_type == "x_start":
            x_start = self.model_out_to_x(model_out)
            return self.predict_noise_from_xstart_xt(x_start=x_start, x_t=x_t, t_index=t_index)

        if self.model_out_type == "noise":
            return model_out

        raise ValueError(f"Unknown DiffusionModelOutputType {self.model_out_type}.")

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
            "learned",
            "learned_range",
        ]:
            # model_out (batch, ..., num_classes)
            model_out, log_variance = jnp.split(model_out, indices_or_sections=2, axis=-1)
            # apply a stop-gradient to the mean output for the vlb to prevent
            # this loss change mean prediction
            model_out_vlb = jax.lax.stop_gradient(model_out)
            # model_out (batch, ..., num_classes*2)
            model_out_vlb = jnp.concatenate([model_out_vlb, log_variance], axis=-1)
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
        mse_loss_batch = jnp.mean((noise_pred - noise) ** 2, axis=range(1, noise.ndim))
        scalars["mse_loss"] = mse_loss_batch

        return scalars, model_out
