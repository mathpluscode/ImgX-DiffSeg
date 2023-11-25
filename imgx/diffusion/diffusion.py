"""Base diffusion class."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import jax.random


@dataclass
class Diffusion:
    """Base class for diffusion."""

    num_timesteps: int
    noise_fn: Callable[..., jnp.ndarray]

    def sample_noise(self, key: jax.Array, shape: Sequence[int], dtype: jnp.dtype) -> jnp.ndarray:
        """Return a noise of the same shape as input.

        Define this function to avoid defining randon key.

        Args:
            key: random key.
            shape: array shape.
            dtype: data type.

        Returns:
            Noise of the same shape and dtype as x.
        """
        return self.noise_fn(key=key, shape=shape, dtype=dtype)

    def t_index_to_t(self, t_index: jnp.ndarray) -> jnp.ndarray:
        """Convert t_index to t.

        t_index = 0 corresponds to t = 1 / num_timesteps.
        t_index = num_timesteps - 1 corresponds to t = 1.

        Args:
            t_index: t_index, shape (batch, ).

        Returns:
            t: t, shape (batch, ).
        """
        return jnp.asarray(t_index + 1, jnp.float32) / self.num_timesteps

    def q_sample(
        self,
        x_start: jnp.ndarray,
        noise: jnp.ndarray,
        t_index: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample from q(x_t | x_0).

        Args:
            x_start: noiseless input.
            noise: same shape as x_start.
            t_index: storing index values < self.num_timesteps.

        Returns:
            Noisy array with same shape as x_start.
        """
        raise NotImplementedError

    def predict_xprev_from_xstart_xt(
        self, x_start: jnp.ndarray, x_t: jnp.ndarray, t_index: jnp.ndarray
    ) -> jnp.ndarray:
        """Get x_{t-1} from x_0 and x_t.

        Args:
            x_start: noisy input at t, shape (batch, ...).
            x_t: noisy input, same shape as x_start.
            t_index: storing index values < self.num_timesteps, shape (batch, ).

        Returns:
            predicted x_0, same shape as x_prev.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def variational_lower_bound(
        self,
        model_out: jnp.ndarray,
        x_start: jnp.ndarray,
        x_t: jnp.ndarray,
        t_index: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Variational lower-bound, ELBO, smaller is better.

        Args:
            model_out: raw model output, may contain additional parameters.
            x_start: noiseless input.
            x_t: noisy input, same shape as x_start.
            t_index: storing index values < self.num_timesteps.

        Returns:
            - lower bounds, shape (batch, ).
            - model_out with the same shape as x_start.
        """
        raise NotImplementedError

    def sample(
        self,
        key: jax.Array,
        model_out: jnp.ndarray,
        x_t: jnp.ndarray,
        t_index: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample x_{t-1} ~ p(x_{t-1} | x_t).

        Args:
            key: random key.
            model_out: model predicted output.
                If model estimates variance, the last axis will be split.
            x_t: noisy x at time t.
            t_index: storing index values < self.num_timesteps.

        Returns:
            sample: x_{t-1}, same shape as x_t.
            x_start_pred: same shape as x_t.
        """
        raise NotImplementedError

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
            model_out: model output, may contain additional parameters.

        Returns:
            scalars: dict of losses, each with shape (batch, ).
            model_out: same shape as x_start.
        """
        raise NotImplementedError
