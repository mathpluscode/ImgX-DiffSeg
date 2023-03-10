"""Gaussian diffusion related functions.

https://github.com/WuJunde/MedSegDiff/blob/master/guided_diffusion/gaussian_diffusion.py
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py
"""
import dataclasses
import enum
from typing import Callable, Iterator, Sequence, Tuple, Union

import haiku as hk
import jax.numpy as jnp
import jax.random
import numpy as np

from imgx import EPS
from imgx.metric.distribution import (
    discretized_gaussian_log_likelihood,
    normal_kl,
)


class DiffusionBetaSchedule(enum.Enum):
    """Class to define beta schedule."""

    LINEAR = enum.auto()
    QUADRADIC = enum.auto()
    COSINE = enum.auto()
    WARMUP10 = enum.auto()
    WARMUP50 = enum.auto()


class DiffusionModelOutputType(enum.Enum):
    """Class to define model's output meaning.

    - X_START: model predicts x_0.
    - X_PREVIOUS: model predicts x_{t-1}.
    - EPSILON: model predicts noise epsilon.
    """

    X_START = enum.auto()
    X_PREVIOUS = enum.auto()
    EPSILON = enum.auto()


class DiffusionModelVarianceType(enum.Enum):
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


class DiffusionSpace(enum.Enum):
    """Class to define the meaning of x.

    Model always outputs logits.
    """

    SCALED_PROBS = enum.auto()  # values will be [-1, 1]
    LOGITS = enum.auto()


def extract_and_expand(
    arr: jnp.ndarray, t: jnp.ndarray, ndim: int
) -> jnp.ndarray:
    """Extract values from a 1D array and expand.

    This function is not jittable.

    Args:
        arr: 1D of shape (num_timesteps, ).
        t: storing index values < self.num_timesteps, shape (batch, ).
        ndim: number of dimensions for an array of shape (batch, ...).

    Returns:
        Expanded array of shape (batch, ...), expanded axes have dim 1.
    """
    return jnp.expand_dims(arr[t], axis=tuple(range(1, ndim)))


def get_beta_schedule(
    num_timesteps: int,
    beta_schedule: DiffusionBetaSchedule,
    beta_start: float,
    beta_end: float,
) -> jnp.ndarray:
    """Get variance (beta) schedule for q(x_t | x_{t-1}).

        TODO: open-source code used float64 for beta.

    Args:
        num_timesteps: number of time steps in total, T.
        beta_schedule: schedule for beta.
        beta_start: beta for t=0.
        beta_end: beta for t=T.

    Raises:
        ValueError: for unknown schedule.
    """
    if beta_schedule == DiffusionBetaSchedule.LINEAR:
        return jnp.linspace(
            beta_start,
            beta_end,
            num_timesteps,
        )
    if beta_schedule == DiffusionBetaSchedule.QUADRADIC:
        return (
            jnp.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_timesteps,
            )
            ** 2
        )
    if beta_schedule == DiffusionBetaSchedule.COSINE:

        def alphas_cumprod(t: float) -> float:
            """Eq 17 in https://arxiv.org/abs/2102.09672."""
            return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2

        max_beta = 0.999
        betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            beta = min(1 - alphas_cumprod(t2) / alphas_cumprod(t1), max_beta)
            betas.append(beta)
        return jnp.array(betas)

    if beta_schedule == DiffusionBetaSchedule.WARMUP10:
        num_timesteps_warmup = max(num_timesteps // 10, 1)
        betas_warmup = (
            jnp.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_timesteps_warmup,
            )
            ** 2
        )
        return jnp.concatenate(
            [
                betas_warmup,
                jnp.ones((num_timesteps - num_timesteps_warmup,)) * beta_end,
            ]
        )
    if beta_schedule == DiffusionBetaSchedule.WARMUP50:
        num_timesteps_warmup = max(num_timesteps // 2, 1)
        betas_warmup = (
            jnp.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_timesteps_warmup,
            )
            ** 2
        )
        return jnp.concatenate(
            [
                betas_warmup,
                jnp.ones((num_timesteps - num_timesteps_warmup,)) * beta_end,
            ]
        )
    raise ValueError(f"Unknown beta_schedule {beta_schedule}.")


@dataclasses.dataclass
class GaussianDiffusion(hk.Module):
    """Class for Gaussian diffusion sampling.

    https://github.com/WuJunde/MedSegDiff/blob/master/guided_diffusion/gaussian_diffusion.py

    TODO: split segmentation related functions to a sub-class.
    """

    def __init__(
        self,
        model: hk.Module,
        num_timesteps: int,  # T
        num_timesteps_beta: int,
        beta_schedule: DiffusionBetaSchedule,
        beta_start: float,
        beta_end: float,
        model_out_type: DiffusionModelOutputType,
        model_var_type: DiffusionModelVarianceType,
        x_space: DiffusionSpace,
        x_limit: float,
        use_ddim: bool,
        noise_fn: Callable = jax.random.normal,
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
            x_space: x is logits or scaled_probs.
            x_limit: x_t has values in [-x_limit, x_limit], the range has to be
                symmetric, as for T, the distribution is centered at zero.
            use_ddim: use ddim_sample.
            noise_fn: a function that gets noise of the same shape as x_t.
        """
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.num_timesteps_beta = num_timesteps_beta
        self.use_ddim = use_ddim
        self.model_out_type = model_out_type
        self.model_var_type = model_var_type
        self.x_space = x_space
        self.x_limit = x_limit
        self.noise_fn = noise_fn

        # shape are all (T,)
        # corresponding to 0, ..., T-1, where 0 means one step
        self.betas = get_beta_schedule(
            num_timesteps=num_timesteps_beta,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        if num_timesteps_beta % num_timesteps != 0:
            raise ValueError(
                f"num_timesteps_beta={num_timesteps_beta} "
                f"can't be evenly divided by num_timesteps={num_timesteps}."
            )
        if num_timesteps != num_timesteps_beta:
            # adjust beta
            step_scale = num_timesteps_beta // num_timesteps
            alphas = 1.0 - self.betas
            alphas_cumprod = jnp.cumprod(alphas)
            alphas_cumprod = alphas_cumprod[step_scale - 1 :: step_scale]
            self.betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]

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

    def q_mean_log_variance(
        self, x_start: jnp.ndarray, t: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get the distribution q(x_t | x_0).

        Args:
            x_start: noiseless input, shape (batch, ...).
            t: storing index values < self.num_timesteps, shape (batch, ).

        Returns:
            mean: shape (batch, ...), expanded axes have dim 1.
            log_variance: shape (batch, ...), expanded axes have dim 1.
        """
        mean = (
            extract_and_expand(self.sqrt_alphas_cumprod, t, x_start.ndim)
            * x_start
        )
        log_variance = extract_and_expand(
            self.log_one_minus_alphas_cumprod, t, x_start.ndim
        )
        return mean, log_variance

    def q_sample(
        self,
        x_start: jnp.ndarray,
        noise: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample from q(x_t | x_0).

        Args:
            x_start: noiseless input, shape (batch, ...).
            noise: same shape as x_start.
            t: storing index values < self.num_timesteps, shape (batch, ).

        Returns:
            Noisy array with same shape as x_start.
        """
        mean = (
            extract_and_expand(self.sqrt_alphas_cumprod, t, x_start.ndim)
            * x_start
        )
        var = extract_and_expand(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.ndim
        )
        x_t = mean + var * noise
        x_t = self.clip_x(x_t)
        return x_t

    def q_posterior_mean(
        self, x_start: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """Get mean of the distribution q(x_{t-1} | x_t, x_0).

        Args:
            x_start: noiseless input, shape (batch, ...).
            x_t: noisy input, same shape as x_start.
            t: storing index values < self.num_timesteps, shape (batch, ).

        Returns:
            mean: same shape as x_start.
        """
        return (
            extract_and_expand(self.posterior_mean_coeff_start, t, x_start.ndim)
            * x_start
            + extract_and_expand(self.posterior_mean_coeff_t, t, x_start.ndim)
            * x_t
        )

    def q_posterior_mean_variance(
        self, x_start: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get the distribution q(x_{t-1} | x_t, x_0).

        Args:
            x_start: noiseless input, shape (batch, ...).
            x_t: noisy input, same shape as x_start.
            t: storing index values < self.num_timesteps, shape (batch, ).

        Returns:
            mean: same shape as x_start.
            log_variance: shape (batch, ...), expanded axes have dim 1.
        """
        mean = self.q_posterior_mean(x_start, x_t, t)
        log_variance = extract_and_expand(
            self.posterior_log_variance_clipped, t, x_start.ndim
        )
        return mean, log_variance

    def p_mean_variance(  # pylint:disable=R0912
        self,
        model_out: jnp.ndarray,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Get the distribution p(x_{t-1} | x_t).

        Args:
            model_out: model predicted output.
                If model estimates variance, shape (batch, ..., 2*num_classes),
                else shape (batch, ..., num_classes).
            x_t: noisy input, shape (batch, ..., num_classes).
            t: storing index values < self.num_timesteps, shape (batch, ).

        Returns:
            x_start: predicted, same shape as x_t, values are clipped.
            mean: same shape as x_t.
            log_variance: compatible shape to (batch, ..., num_classes).
        """
        # variance
        if self.model_var_type == DiffusionModelVarianceType.FIXED_SMALL:
            log_variance = self.posterior_log_variance_clipped

            # extend shape
            log_variance = extract_and_expand(log_variance, t, x_t.ndim)

        elif self.model_var_type == DiffusionModelVarianceType.FIXED_LARGE:
            # TODO why appending?
            variance = jnp.append(self.posterior_variance[1], self.betas[1:])
            log_variance = jnp.log(variance)

            # extend shape
            log_variance = extract_and_expand(log_variance, t, x_t.ndim)

        elif self.model_var_type == DiffusionModelVarianceType.LEARNED:
            # model_out (batch, ..., num_classes*2)
            model_out, log_variance = jnp.split(
                model_out, indices_or_sections=2, axis=-1
            )

        elif self.model_var_type == DiffusionModelVarianceType.LEARNED_RANGE:
            # model_out (batch, ..., num_classes*2)
            model_out, var_coeff = jnp.split(
                model_out, indices_or_sections=2, axis=-1
            )
            log_min_variance = self.posterior_log_variance_clipped
            log_max_variance = jnp.log(self.betas)
            log_min_variance = extract_and_expand(log_min_variance, t, x_t.ndim)
            log_max_variance = extract_and_expand(log_max_variance, t, x_t.ndim)
            # var_coeff values are in [-1, 1] for [min_var, max_var].
            var_coeff = jnp.clip(var_coeff, -1.0, 1.0)
            var_coeff = (var_coeff + 1) / 2
            log_variance = (
                var_coeff * log_max_variance
                + (1 - var_coeff) * log_min_variance
            )
        else:
            raise ValueError(
                f"Unknown DiffusionModelVarianceType {self.model_var_type}."
            )

        # mean
        if self.model_out_type == DiffusionModelOutputType.X_START:
            # q(x_{t-1} | x_t, x_0)
            x_start = self.logits_to_x(model_out)
            x_start = self.clip_x(x_start)
            mean = self.q_posterior_mean(x_start=x_start, x_t=x_t, t=t)
        elif self.model_out_type == DiffusionModelOutputType.X_PREVIOUS:
            # x_{t-1}
            x_prev = self.logits_to_x(model_out)
            x_prev = self.clip_x(x_prev)
            mean = x_prev
            x_start = self.predict_xstart_from_xprev_xt(
                x_prev=x_prev, x_t=x_t, t=t
            )
            x_start = self.clip_x(x_start)
        elif self.model_out_type == DiffusionModelOutputType.EPSILON:
            x_start = self.predict_xstart_from_epsilon_xt(
                x_t=x_t, epsilon=model_out, t=t
            )
            x_start = self.clip_x(x_start)
            mean = self.q_posterior_mean(x_start=x_start, x_t=x_t, t=t)
        else:
            raise ValueError(
                f"Unknown DiffusionModelOutputType {self.model_out_type}."
            )
        return x_start, mean, log_variance

    def p_sample(
        self,
        model_out: jnp.ndarray,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample x_{t-1} ~ p(x_{t-1} | x_t).

        Args:
            model_out: model predicted output.
                If model estimates variance, shape (batch, ..., 2*num_classes),
                else shape (batch, ..., num_classes).
            x_t: noisy input, shape (batch, ..., num_classes).
            t: storing index values < self.num_timesteps, shape (batch, ).

        Returns:
            sample: x_{t-1}, same shape as x_t.
            x_start_pred: same shape as x_t.
        """
        x_start_pred, mean, log_variance = self.p_mean_variance(
            model_out=model_out,
            x_t=x_t,
            t=t,
        )
        noise = self.noise_sample(shape=x_t.shape, dtype=x_t.dtype)

        # no noise when t=0
        # mean + exp(log(sigma**2)/2) * noise = mean + sigma * noise
        nonzero_mask = jnp.expand_dims(
            jnp.array(t != 0, dtype=noise.dtype),
            axis=tuple(range(1, noise.ndim)),
        )
        sample = mean + nonzero_mask * jnp.exp(0.5 * log_variance) * noise
        # clip as the value may be out of range
        sample = self.clip_x(sample)

        return sample, x_start_pred

    def ddim_sample(
        self,
        model_out: jnp.ndarray,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        eta: float = 0.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample x_{t-1} ~ p(x_{t-1} | x_t).

        TODO: what are the differences between p_sample / ddim_sample?

        Args:
            model_out: model predicted output.
                If model estimates variance, shape (batch, ..., 2*num_classes),
                else shape (batch, ..., num_classes).
            x_t: noisy input, shape (batch, ..., num_classes).
            t: storing index values < self.num_timesteps, shape (batch, ).
            eta: control the noise level in sampling.

        Returns:
            sample: x_{t-1}, same shape as x_t.
            x_start_pred: same shape as x_t.
        """
        # TODO why not using log variance output here?
        x_start_pred, _, _ = self.p_mean_variance(
            model_out=model_out,
            x_t=x_t,
            t=t,
        )
        noise = self.noise_sample(shape=x_t.shape, dtype=x_t.dtype)
        epsilon = self.predict_epsilon_from_xstart_xt(
            x_t=x_t, x_start=x_start_pred, t=t
        )

        alphas_cumprod_prev = extract_and_expand(
            self.alphas_cumprod_prev, t, x_t.ndim
        )
        coeff_start = alphas_cumprod_prev
        log_variance = (
            extract_and_expand(self.posterior_log_variance_clipped, t, x_t.ndim)
            * eta
        )
        coeff_epsilon = jnp.sqrt(1.0 - alphas_cumprod_prev - log_variance**2)
        mean = coeff_start * x_start_pred + coeff_epsilon * epsilon
        nonzero_mask = jnp.expand_dims(
            jnp.array(t != 0, dtype=noise.dtype),
            axis=tuple(range(1, noise.ndim)),
        )
        sample = mean + nonzero_mask * log_variance * noise

        # clip as the value may be out of range
        sample = self.clip_x(sample)

        return sample, x_start_pred

    def sample_mask(
        self,
        image: jnp.ndarray,
        x_t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Generate segmentation mask from the model conditioned on image.

        The noise here is defined on segmentation mask.
        x_t is considered as logits.

        Args:
            image: image to be segmented, shape = (batch, ..., C).
            x_t: segmentation logits to be refined,
                 shape = (batch, ..., num_classes).

        Returns:
            Sampled segmentation logits, shape = (batch, ..., num_classes).
        """
        for x_start_t in self.sample_mask_progressive(image=image, x_t=x_t):
            x_start = x_start_t
        return x_start

    def sample_mask_progressive(
        self,
        image: jnp.ndarray,
        x_t: jnp.ndarray,
    ) -> Iterator[jnp.ndarray]:
        """Generate segmentation mask from the model conditioned on image.

        The noise here is defined on segmentation mask.
        x_t is considered as logits.

        Args:
            image: image to be segmented, shape = (batch, ..., C).
            x_t: segmentation logits to be refined,
                 shape = (batch, ..., num_classes).

        Yields:
            x_start of shape = (batch, ..., num_classes, T).
        """
        for t in reversed(range(self.num_timesteps)):
            # (batch, )
            t_batch = jnp.array(
                [t] * x_t.shape[0],
                dtype=jnp.int16,
            )
            # (batch, ..., ch_input + num_classes)
            model_in = jnp.concatenate([image, x_t], axis=-1)
            # (batch, ..., num_classes) or (batch, ..., 2*num_classes)
            model_out = self.model(model_in, t_batch)
            if self.use_ddim:
                x_t, x_start = self.ddim_sample(
                    model_out=model_out,
                    x_t=x_t,
                    t=t_batch,
                )
            else:
                x_t, x_start = self.p_sample(
                    model_out=model_out,
                    x_t=x_t,
                    t=t_batch,
                )
            yield x_start

    def predict_xstart_from_xprev_xt(
        self, x_prev: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """Get x_0 from x_{t-1} and x_t.

        The mean of q(x_{t-1} | x_t, x_0) is coeff_start * x_0 + coeff_t * x_t.
        So x_{t-1} = coeff_start * x_0 + coeff_t * x_t.
        x_0 = (x_{t-1} - coeff_t * x_t) / coeff_start
            = 1/coeff_start * x_{t-1} - coeff_t/coeff_start * x_t

        Args:
            x_prev: noisy input at t-1, shape (batch, ...).
            x_t: noisy input, same shape as x_prev.
            t: storing index values < self.num_timesteps, shape (batch, ).

        Returns:
            predicted x_0, same shape as x_prev.
        """
        coeff_prev = extract_and_expand(
            1.0 / self.posterior_mean_coeff_start, t, x_t.ndim
        )
        coeff_t = extract_and_expand(
            self.posterior_mean_coeff_t / self.posterior_mean_coeff_start,
            t,
            x_t.ndim,
        )
        return coeff_prev * x_prev - coeff_t * x_t

    def predict_xprev_from_xstart_xt(
        self, x_start: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """Get x_{t-1} from x_0 and x_t.

        The mean of q(x_{t-1} | x_t, x_0) is coeff_start * x_0 + coeff_t * x_t.
        So x_{t-1} = coeff_start * x_0 + coeff_t * x_t.

        Args:
            x_start: noisy input at t, shape (batch, ...).
            x_t: noisy input, same shape as x_start.
            t: storing index values < self.num_timesteps, shape (batch, ).

        Returns:
            predicted x_0, same shape as x_prev.
        """
        coeff_start = extract_and_expand(
            self.posterior_mean_coeff_start, t, x_t.ndim
        )
        coeff_t = extract_and_expand(
            self.posterior_mean_coeff_t,
            t,
            x_t.ndim,
        )
        return coeff_start * x_start + coeff_t * x_t

    def sample_xprev_from_xstart_xt(
        self, x_start: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """Sample x_{t-1} from q(x_{t-1} | x_0, x_t).

        The mean of q(x_{t-1} | x_t, x_0) is coeff_start * x_0 + coeff_t * x_t.
        So x_{t-1} = coeff_start * x_0 + coeff_t * x_t.

        Args:
            x_start: noisy input at t, shape (batch, ...).
            x_t: noisy input, same shape as x_start.
            t: storing index values < self.num_timesteps, shape (batch, ).

        Returns:
            predicted x_0, same shape as x_prev.
        """
        x_prev = self.predict_xprev_from_xstart_xt(
            x_start=x_start,
            x_t=x_t,
            t=t,
        )
        noise = self.noise_sample(shape=x_t.shape, dtype=x_t.dtype)
        log_variance = extract_and_expand(
            self.posterior_log_variance_clipped, t, x_t.ndim
        )
        sample = x_prev + noise * log_variance
        return self.clip_x(sample)

    def predict_xstart_from_epsilon_xt(
        self, x_t: jnp.ndarray, epsilon: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """Get x_0 from epsilon.

        The reparameterization gives:
            x_t = sqrt(alphas_cumprod) * x_0
                + sqrt(1-alphas_cumprod) * epsilon
        so,
        x_0 = 1/sqrt(alphas_cumprod) * x_t
            - sqrt(1-alphas_cumprod)/sqrt(alphas_cumprod) * epsilon
            = 1/sqrt(alphas_cumprod) * x_t
            - sqrt(1/alphas_cumprod - 1) * epsilon

        Args:
            x_t: noisy input at t-1, shape (batch, ...).
            epsilon: noise, shape (batch, ...), expanded axes have dim 1.
            t: storing index values < self.num_timesteps, shape (batch, ).

        Returns:
            predicted x_0, same shape as x_t.
        """
        coeff_t = extract_and_expand(
            self.sqrt_recip_alphas_cumprod, t, x_t.ndim
        )
        coeff_epsilon = extract_and_expand(
            self.sqrt_recip_alphas_cumprod_minus_one, t, x_t.ndim
        )
        return coeff_t * x_t - coeff_epsilon * epsilon

    def predict_epsilon_from_xstart_xt(
        self, x_t: jnp.ndarray, x_start: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """Get epsilon from x_0 and x_t.

        The reparameterization gives:
            x_t = sqrt(alphas_cumprod) * x_0
                + sqrt(1-alphas_cumprod) * epsilon
        so,
        epsilon = (x_t - sqrt(alphas_cumprod) * x_0) / sqrt(1-alphas_cumprod)
                = (1/sqrt(alphas_cumprod) * x_t - x_0)
                  /sqrt(1/alphas_cumprod-1)

        Args:
            x_t: noisy input at t-1, shape (batch, ...).
            x_start: predicted x_0, same shape as x_t.
            t: storing index values < self.num_timesteps, shape (batch, ).

        Returns:
            predicted x_0, same shape as x_t.
        """
        coeff_t = extract_and_expand(
            self.sqrt_recip_alphas_cumprod, t, x_t.ndim
        )
        denominator = extract_and_expand(
            self.sqrt_recip_alphas_cumprod_minus_one, t, x_t.ndim
        )
        return (coeff_t * x_t - x_start) / denominator

    def sample_timestep(
        self, batch_size: int, min_val: Union[int, jnp.ndarray] = 0
    ) -> jnp.ndarray:
        """Sample t of shape (batch, ).

        Define this function to avoid defining randon key.

        Args:
            batch_size: number of steps.
            min_val: minimum value, inclusive.

        Returns:
            Time steps with value between 0 and T-1, both sides inclusive.
        """
        min_val = jnp.minimum(min_val, self.num_timesteps - 1)
        return jax.random.randint(
            hk.next_rng_key(),
            shape=(batch_size,),
            minval=min_val,  # inclusive
            maxval=self.num_timesteps,  # exclusive
        )

    def noise_sample(
        self, shape: Sequence[int], dtype: jnp.dtype
    ) -> jnp.ndarray:
        """Return a noise of the same shape as input.

        Define this function to avoid defining randon key.

        Args:
            shape: array shape.
            dtype: data type.

        Returns:
            Noise of the same shape and dtype as x.
        """
        return self.noise_fn(key=hk.next_rng_key(), shape=shape, dtype=dtype)

    def clip_x(self, x: jnp.ndarray) -> jnp.ndarray:
        """Clip the x_start/x_t to desired range.

        TODO: where should clip be used?

        Args:
            x: any array.

        Returns:
            Clipped array.
        """
        if self.x_limit <= 0:
            return x
        return jnp.clip(x, -self.x_limit, self.x_limit)

    def logits_to_x(self, logits: jnp.ndarray) -> jnp.ndarray:
        """Map logits to x space.

        Args:
            logits: unnormalised logits.

        Returns:
            Array in the same space as x_start.
        """
        if self.x_space == DiffusionSpace.LOGITS:
            return logits
        if self.x_space == DiffusionSpace.SCALED_PROBS:
            x = jax.nn.softmax(logits, axis=-1)
            x = x * 2.0 - 1.0
            return x
        raise ValueError(f"Unknown x space {self.x_space}.")

    def x_to_logits(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map x to logits.

        Args:
            x: in the same space as x_start.

        Returns:
            Logits.
        """
        if self.x_space == DiffusionSpace.LOGITS:
            return x
        if self.x_space == DiffusionSpace.SCALED_PROBS:
            probs = (x + 1) / 2
            probs = jnp.clip(probs, EPS, 1.0)
            return jnp.log(probs)
        raise ValueError(f"Unknown x space {self.x_space}.")

    def variational_lower_bound(
        self,
        model_out: jnp.ndarray,
        x_start: jnp.ndarray,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Variational lower-bound, smaller is better.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        Args:
            model_out: model predicted output, may present different things,
                shape (batch, ...).
            x_start: cleaned, same shape as x_t.
            x_t: noisy input, shape (batch, ...).
            t: storing index values < self.num_timesteps, shape (batch, ).

        Returns:
            lower bounds of shape (batch, ).
        """
        reduce_axis = tuple(range(x_t.ndim))[1:]

        # q(x_{t-1} | x_t, x_0)
        q_mean, q_log_variance = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        # p(x_{t-1} | x_t)
        _, p_mean, p_log_variance = self.p_mean_variance(
            model_out=model_out,
            x_t=x_t,
            t=t,
        )

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
        kl = jnp.mean(kl, axis=reduce_axis) / jnp.log(2.0)
        nll = jnp.mean(nll, axis=reduce_axis) / jnp.log(2.0)

        # return neg-log-likelihood for t = 0
        return jnp.where(t == 0, nll, kl)
