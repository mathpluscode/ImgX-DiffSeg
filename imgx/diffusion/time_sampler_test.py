"""Test time sampler class and related functions."""


import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.diffusion.time_sampler import TimeSampler, scatter_add, scatter_set


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestScatter(chex.TestCase):
    """Test scatter_add and scatter_set."""

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "add to zeros",
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0, 3, 0]),
            np.array([-1.0, 2.1, 1.2]),
            np.array([0.2, 0.0, 0.0, 2.1, 0.0]),
        ),
        (
            "add to non zeros",
            np.array([0.0, 1.0, 0.0, 2.0, 0.0]),
            np.array([0, 1, 2]),
            np.array([-1.0, 2.1, 1.2]),
            np.array([-1.0, 3.1, 1.2, 2.0, 0.0]),
        ),
    )
    def test_scatter_add(
        self,
        x: np.ndarray,
        indices: np.ndarray,
        updates: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        """Test scatter_add."""
        got = self.variant(scatter_add)(
            x,
            indices,
            updates,
        )
        chex.assert_trees_all_close(got, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "set to zeros",
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0, 3, 1]),
            np.array([-1.0, 2.1, 1.2]),
            np.array([-1.0, 1.2, 0.0, 2.1, 0.0]),
        ),
        (
            "set to non zeros",
            np.array([0.0, 1.0, 0.0, 2.0, 4.0]),
            np.array([0, 3, 1]),
            np.array([-1.0, 2.1, 1.2]),
            np.array([-1.0, 1.2, 0.0, 2.1, 4.0]),
        ),
    )
    def test_scatter_set(
        self,
        x: np.ndarray,
        indices: np.ndarray,
        updates: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        """Test scatter_set."""
        got = self.variant(scatter_set)(
            x,
            indices,
            updates,
        )
        chex.assert_trees_all_close(got, expected)


class TestTimeSampler(chex.TestCase):
    """Test TimeSampler."""

    num_timesteps = 4
    batch_size = 2

    @parameterized.named_parameters(
        (
            "uniform",
            True,
            0,
            3,
        ),
        (
            "importance sampling",
            False,
            0,
            3,
        ),
    )
    def test_shape(
        self,
        uniform_time_sampling: bool,
        t_index_min: int,
        t_index_max: int,
    ) -> None:
        """Test output shape."""
        key = jax.random.PRNGKey(0)
        sampler = TimeSampler(
            num_timesteps=self.num_timesteps,
            uniform_time_sampling=uniform_time_sampling,
        )
        loss_count_hist = jnp.ones((self.num_timesteps,), dtype=jnp.int32)
        loss_sq_hist = jnp.ones((self.num_timesteps,), dtype=jnp.float32)
        t, t_index, probs_t = sampler.sample(
            key, self.batch_size, t_index_min, t_index_max, loss_count_hist, loss_sq_hist
        )
        chex.assert_shape(t, (self.batch_size,))
        chex.assert_shape(t_index, (self.batch_size,))
        chex.assert_shape(probs_t, (self.batch_size,))

    @parameterized.named_parameters(
        (
            "uniform zero",
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ),
        (
            "uniform",
            np.array([0.1, 0.1, 0.1, 0.1]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ),
        (
            "non-uniform one hot",
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.9925, 0.0025, 0.0025, 0.0025]),
        ),
        (
            "non-uniform",
            np.array([9.0, 4.0, 0.0, 0.0]),
            np.array([0.0025 + 0.6 * 0.99, 0.0025 + 0.4 * 0.99, 0.0025, 0.0025]),
        ),
    )
    def test_t_probs_from_loss_sq(self, loss_sq_hist: np.ndarray, expected: np.ndarray) -> None:
        """Test t_probs_from_loss_sq."""
        sampler = TimeSampler(
            num_timesteps=self.num_timesteps,
            uniform_time_sampling=False,
        )
        got = sampler.t_probs_from_loss_sq(jnp.array(loss_sq_hist))
        chex.assert_trees_all_close(got, jnp.array(expected))

    @parameterized.named_parameters(
        (
            "uniform",
            np.array([10.0, 10.0, 10.0, 10.0]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ),
        (
            "uniform zero",
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ),
        (
            "non-uniform",
            np.array([4.0, 6.0, 5.0, 10.0]),
            np.array(
                [
                    6.0 / 15.0,
                    4.0 / 15.0,
                    5.0 / 15.0,
                    0.0,
                ]
            ),
        ),
    )
    def test_t_probs_from_loss_count(
        self, loss_count_hist: np.ndarray, expected: np.ndarray
    ) -> None:
        """Test t_probs_from_loss_count."""
        sampler = TimeSampler(
            num_timesteps=self.num_timesteps,
            uniform_time_sampling=False,
        )
        got = sampler.t_probs_from_loss_count(jnp.array(loss_count_hist))
        chex.assert_trees_all_close(got, jnp.array(expected))

    def test_sample(self) -> None:
        """Test sample make sure that all time steps are sampled after enough steps."""
        batch_size = 4
        sampler = TimeSampler(
            num_timesteps=self.num_timesteps,
            uniform_time_sampling=False,
        )
        loss_count_hist = jnp.zeros((self.num_timesteps,), dtype=jnp.int32)
        loss_sq_hist = jnp.zeros((self.num_timesteps,), dtype=jnp.float32)
        # the coefficient 1.1 is to ensure over-sampling since probs has 0.01 uniform noise
        for i in range(int(sampler.warmup_steps * self.num_timesteps // batch_size * 1.1)):
            _, t_index, probs_t = sampler.sample(
                key=jax.random.PRNGKey(i),
                batch_size=batch_size,
                t_index_min=0,
                t_index_max=self.num_timesteps,
                loss_count_hist=loss_count_hist,
                loss_sq_hist=loss_sq_hist,
            )
            loss_count_hist, loss_sq_hist = sampler.update_stats(
                loss_batch=jnp.ones((batch_size,)),
                t_index=t_index,
                loss_count_hist=loss_count_hist,
                loss_sq_hist=loss_sq_hist,
            )
            min_loss_count_hist = jnp.min(loss_count_hist)
            if min_loss_count_hist < sampler.warmup_steps:
                chex.assert_trees_all_close(probs_t, jnp.ones_like(probs_t) / self.num_timesteps)
        min_loss_count_hist = jnp.min(loss_count_hist)
        assert min_loss_count_hist >= sampler.warmup_steps
        max_loss_count_hist = jnp.max(loss_count_hist)
        assert max_loss_count_hist > sampler.warmup_steps
