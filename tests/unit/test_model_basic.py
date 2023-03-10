"""Test basic functions for model."""
import chex
import jax
from absl.testing import parameterized
from chex._src import fake

from imgx.model.basic import sinusoidal_positional_embedding


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestSinusoidalPositionalEmbedding(chex.TestCase):
    """Test the function sinusoidal_positional_embedding."""

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, without_device=True
    )
    @parameterized.named_parameters(
        ("case 1", 2, 4, 5),
        (
            "case 2",
            3,
            8,
            10000,
        ),
    )
    def test_shapes(self, batch_size: int, dim: int, max_period: int) -> None:
        """Test output shapes under different device condition.

        Args:
            batch_size: batch size.
            dim: embedding dimension, assume to be evenly divided by two.
            max_period: controls the minimum frequency of the embeddings.
        """
        rng = jax.random.PRNGKey(0)
        x = jax.random.uniform(
            rng,
            shape=(batch_size,),
        )
        out = self.variant(
            sinusoidal_positional_embedding, static_argnums=(1, 2)
        )(
            x,
            dim=dim,
            max_period=max_period,
        )
        chex.assert_shape(out, (batch_size, dim))
