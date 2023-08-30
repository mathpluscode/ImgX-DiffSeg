"""Test Gaussian diffusion related classes and functions."""


import chex
import jax.numpy as jnp
from absl.testing import parameterized
from chex._src import fake

from imgx.diffusion.variance_schedule import (
    DiffusionBetaSchedule,
    downsample_beta_schedule,
    get_beta_schedule,
)


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestGetBetaSchedule(chex.TestCase):
    """Test get_beta_schedule."""

    @parameterized.product(
        num_timesteps=[1, 4],
        beta_schedule=[
            DiffusionBetaSchedule.LINEAR,
            DiffusionBetaSchedule.QUADRADIC,
            DiffusionBetaSchedule.COSINE,
            DiffusionBetaSchedule.WARMUP10,
            DiffusionBetaSchedule.WARMUP50,
        ],
    )
    def test_shapes(
        self,
        num_timesteps: int,
        beta_schedule: DiffusionBetaSchedule,
    ) -> None:
        """Test output shape."""
        beta_start = 0.0
        beta_end = 0.2
        got = get_beta_schedule(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        chex.assert_shape(got, (num_timesteps,))

        assert got[0] == beta_start
        if num_timesteps > 1:
            chex.assert_trees_all_close(got[-1], beta_end)


class TestDownsampleBetaSchedule(chex.TestCase):
    """Test downsample_beta_schedule."""

    @parameterized.named_parameters(
        ("same 1001 steps", 1001, 1001),
        ("same 101 steps", 101, 101),
        ("downsample 21 to 5", 21, 5),
        ("downsample 101 to 5", 101, 5),
        ("downsample 11 to 3", 11, 3),
    )
    def test_values(
        self,
        num_timesteps: int,
        num_timesteps_to_keep: int,
    ) -> None:
        """Test output values and shapes."""
        betas = jnp.linspace(1e-4, 0.02, num_timesteps)
        alphas_cumprod = jnp.cumprod(1.0 - betas)
        got = downsample_beta_schedule(
            betas, num_timesteps, num_timesteps_to_keep
        )
        alphas_cumprod_got = jnp.cumprod(1.0 - got)
        chex.assert_shape(got, (num_timesteps_to_keep,))
        chex.assert_trees_all_close(alphas_cumprod_got[0], alphas_cumprod[0])
        chex.assert_trees_all_close(alphas_cumprod_got[-1], alphas_cumprod[-1])
