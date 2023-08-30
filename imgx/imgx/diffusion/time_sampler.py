"""Module for sampling t, t_index for diffusion."""
from __future__ import annotations

import haiku as hk
import jax.numpy as jnp
import jax.random
from jax import lax


def scatter_add(
    x: jnp.ndarray, indices: jnp.ndarray, updates: jnp.ndarray
) -> jnp.ndarray:
    """Use lax.scatter_add to add value at indices.

    Args:
        x: array to be updated, (n, ).
        indices: index values < n, (batch, ).
        updates: value to be added, (batch, ).

    Returns:
        Updated array x.
    """
    dim_num = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    return lax.scatter_add(
        x,
        scatter_indices=indices[:, None],
        updates=updates,
        dimension_numbers=dim_num,
    )


def scatter_set(
    x: jnp.ndarray, indices: jnp.ndarray, updates: jnp.ndarray
) -> jnp.ndarray:
    """Use lax.scatte to set value at indices.

    When indices are not unique, values at the same index is overwritten
    by the last value.

    Args:
        x: array to be updated, (n, ).
        indices: index values < n, (batch, ).
        updates: value to be added, (batch, ).

    Returns:
        Updated array x.
    """
    dim_num = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    return lax.scatter(
        x,
        scatter_indices=indices[:, None],
        updates=updates,
        dimension_numbers=dim_num,
    )


class TimeSampler(hk.Module):
    """Time sampler for diffusion.

    Each device has its own time sampler.
    """

    def __init__(
        self,
        num_timesteps: int,
        uniform_time_sampling: bool,
        warmup_steps: int = 10,
        decay: float = 0.9,
        uniform_prob: float = 0.01,
    ) -> None:
        """Initialise the class.

        Args:
            num_timesteps: number of diffusion steps.
            uniform_time_sampling: sample time uniformly.
            warmup_steps: number of warmup steps, perform importance sampling
                only if each time step has been sampled warmup_steps times.
            decay: decay rate for loss_sq_hist.
            uniform_prob: probability of sampling t uniformly,
                even if using importance sampling.
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        self.uniform_time_sampling = uniform_time_sampling
        self.warmup_steps = warmup_steps
        self.decay = decay
        self.uniform_prob = uniform_prob

        # init states
        hk.get_state(
            "loss_sq_hist",
            shape=(self.num_timesteps,),
            dtype=jnp.float32,
            init=jnp.zeros,
        )
        hk.get_state(
            "loss_count_hist",
            shape=(self.num_timesteps,),
            dtype=jnp.int32,
            init=jnp.zeros,
        )

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

    def update_loss_sq_hist(
        self,
        loss_batch: jnp.ndarray,
        t_index: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """Update the loss_sq_hist and loss_count_hist.

        Args:
            loss_batch: loss of the current batch, shape (batch, ).
            t_index: t_index of the current batch, shape (batch, ).

        Returns:
            metrics: metrics dict.
        """
        # retrieve the states
        loss_sq_hist = hk.get_state("loss_sq_hist")
        loss_count_hist = hk.get_state("loss_count_hist")

        # (batch, )
        loss_sq_batch = loss_batch**2
        # (batch, )
        loss_sq_prev = loss_sq_hist[t_index]
        loss_sq_updated = (
            1 - self.decay
        ) * loss_sq_batch + self.decay * loss_sq_prev
        loss_sq_hist = scatter_set(
            x=loss_sq_hist, indices=t_index, updates=loss_sq_updated
        )
        loss_count_hist = scatter_add(
            x=loss_count_hist,
            indices=t_index,
            updates=jnp.ones_like(t_index),
        )
        hk.set_state("loss_sq_hist", loss_sq_hist)
        hk.set_state("loss_count_hist", loss_count_hist)

        scalars = {}
        probs = self.t_probs_from_loss_sq(loss_sq_hist)
        entropy = -jnp.sum(probs * jnp.log(probs))
        scalars["loss_hist_entropy"] = entropy

        scalars["mean_loss_count_hist"] = jnp.mean(loss_count_hist)
        scalars["min_loss_count_hist"] = jnp.min(loss_count_hist)
        scalars["max_loss_count_hist"] = jnp.max(loss_count_hist)
        scalars["mean_loss_sq_hist"] = jnp.mean(loss_sq_hist)
        scalars["min_loss_sq_hist"] = jnp.min(loss_sq_hist)
        scalars["max_loss_sq_hist"] = jnp.max(loss_sq_hist)
        return scalars

    def sample_uniformly(
        self,
        t_index_minval: jnp.ndarray,
        t_index_maxval: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample noise and time uniformly.

        Args:
            t_index_minval: minium value of t_index, in [0, num_timesteps),
                shape (batch, ), the values are inclusive.
            t_index_maxval: maximum value of t_index, in [0, num_timesteps),
                shape (batch, ), the values are exclusive.

        Returns:
            t, values between [0, 1), shape (batch, ).
            t_index, values between [0, num_timesteps), shape (batch, ).
            probs_t, probability of sampled t, shape (batch, ).
        """
        t_index = jax.random.randint(
            key=hk.next_rng_key(),
            shape=t_index_minval.shape,
            minval=t_index_minval,  # inclusive
            maxval=t_index_maxval,  # exclusive
        )
        t = self.t_index_to_t(t_index)
        probs_t = jnp.ones_like(t) / self.num_timesteps
        return t, t_index, probs_t

    def t_probs_from_loss_sq(self, loss_sq_hist: jnp.ndarray) -> jnp.ndarray:
        """Get probability of sampling t from loss_sq_hist.

        Args:
            loss_sq_hist: loss_sq_hist, shape (num_timesteps, ).

        Returns:
            probs: probability of sampling each t, shape (num_timesteps, ).
        """
        probs = jnp.sqrt(loss_sq_hist)
        probs /= probs.sum()
        probs *= 1 - self.uniform_prob
        probs += self.uniform_prob / self.num_timesteps
        return probs

    def t_probs_from_loss_count(
        self,
        loss_count_hist: jnp.ndarray,
    ) -> jnp.ndarray:
        """Get probability of sampling t from loss_count_hist.

        Args:
            loss_count_hist: loss_count_hist, shape (num_timesteps, ).

        Returns:
            probs: probability of sampling each t, shape (num_timesteps, ).
        """
        probs = self.warmup_steps - loss_count_hist
        probs = jnp.maximum(probs, 0)
        probs *= 1 - self.uniform_prob
        probs += self.uniform_prob / self.num_timesteps
        probs /= probs.sum()
        return probs

    def sample_with_importance(
        self,
        t_index_minval: jnp.ndarray,
        t_index_maxval: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample noise and time via importance sampling.

        https://arxiv.org/abs/2102.09672
        https://github.com/microsoft/muzic/blob/61e4436516c61b6b358ef709b446de539817decf/getmusic/getmusic/modeling/roformer/diffusion_roformer.py#L376
        https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/resample.py#L70

        Args:
            t_index_minval: minium value of t_index, in [0, num_timesteps),
                shape (batch, ), the values are inclusive.
            t_index_maxval: maximum value of t_index, in [0, num_timesteps),
                shape (batch, ), the values are exclusive.

        Returns:
            t, values between [0, 1), shape (batch, ).
            t_index, values between [0, num_timesteps), shape (batch, ).
            probs_t, probability of sampled t, shape (batch, ).
        """
        batch_size = t_index_minval.shape[0]
        loss_sq_hist = hk.get_state("loss_sq_hist")
        probs = self.t_probs_from_loss_sq(loss_sq_hist=loss_sq_hist)

        # extend the probs to the batch size
        # (batch_size, num_timesteps)
        probs_batch = jnp.tile(probs, (batch_size, 1))

        # mask out the timesteps out of range
        # and renomalize the probs
        mask = jnp.tile(jnp.arange(self.num_timesteps), (batch_size, 1))
        mask = jnp.logical_and(
            mask >= t_index_minval[:, None],
            mask < t_index_maxval[:, None],
        )
        probs_batch = jnp.where(mask, probs_batch, 0.0)
        probs_batch /= probs_batch.sum(axis=1, keepdims=True)

        # sample timesteps
        logits = jnp.log(probs_batch)
        t_index = jax.random.categorical(
            key=hk.next_rng_key(),
            logits=logits,
            shape=(batch_size,),
        )
        t = self.t_index_to_t(t_index)
        probs_t = probs[t_index]

        return t, t_index, probs_t

    def sample(
        self,
        batch_size: int,
        t_index_min: int,
        t_index_max: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample noise and time.

        Each time step in consideration should be sampled enough.

        Args:
            batch_size: batch size.
            t_index_min: minium value of t_index, in [0, num_timesteps),
                the values are inclusive.
            t_index_max: maximum value of t_index, in [0, num_timesteps),
                the values are exclusive.

        Returns:
            t, values between [0, 1), shape (batch, ).
            t_index, values between [0, num_timesteps), shape (batch, ).
            probs_t, probability of sampled t, shape (batch, ).
        """
        t_index_minval = jnp.full((batch_size,), t_index_min, dtype=jnp.int32)
        t_index_maxval = jnp.full((batch_size,), t_index_max, dtype=jnp.int32)
        if self.uniform_time_sampling:
            return self.sample_uniformly(
                t_index_minval=t_index_minval,
                t_index_maxval=t_index_maxval,
            )

        loss_count_hist = hk.get_state("loss_count_hist")
        min_loss_count_hist = jnp.min(loss_count_hist[t_index_min:t_index_max])
        return hk.cond(
            min_loss_count_hist < self.warmup_steps,
            self.sample_uniformly,
            self.sample_with_importance,
            t_index_minval,
            t_index_maxval,
        )
