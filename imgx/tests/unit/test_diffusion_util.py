"""Test Gaussian diffusion related classes and functions."""


import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from chex._src import fake

from imgx.diffusion.util import extract_and_expand


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestExtractAndExpand(chex.TestCase):
    """Test extract_and_expand."""

    @chex.variants(without_jit=True, with_device=True, without_device=True)
    @parameterized.named_parameters(
        (
            "1d",
            1,
        ),
        (
            "2d",
            2,
        ),
        (
            "3d",
            3,
        ),
    )
    def test_shapes(
        self,
        ndim: int,
    ) -> None:
        """Test output shape.

        Args:
            ndim: number of dimensions.
        """
        batch_size = 2
        betas = jnp.array([0, 0.2, 0.5, 1.0])
        num_timesteps = len(betas)
        rng = jax.random.PRNGKey(0)
        t_index = jax.random.randint(
            rng, shape=(batch_size,), minval=0, maxval=num_timesteps
        )
        got = self.variant(extract_and_expand)(
            arr=betas, t_index=t_index, ndim=ndim
        )
        expected_shape = (batch_size,) + (1,) * (ndim - 1)
        chex.assert_shape(got, expected_shape)
