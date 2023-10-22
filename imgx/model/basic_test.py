"""Test basic functions for model."""


from functools import partial

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from chex._src import fake

from imgx.model.basic import MLP, InstanceNorm, sinusoidal_positional_embedding


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestInstanceNorm(chex.TestCase):
    """Test the function sinusoidal_positional_embedding."""

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d",
            (2,),
        ),
        (
            "2d",
            (2, 3),
        ),
    )
    def test_shapes(
        self,
        in_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes under different device condition.

        Args:
            in_shape: input shape.
        """
        rng = {"params": jax.random.PRNGKey(0)}
        norm = InstanceNorm()
        x = jax.random.uniform(
            jax.random.PRNGKey(0),
            shape=in_shape,
        )
        out, _ = self.variant(norm.init_with_output)(rng, x)
        chex.assert_shape(out, in_shape)


class TestSinusoidalPositionalEmbedding(chex.TestCase):
    """Test the function sinusoidal_positional_embedding."""

    @chex.all_variants()
    @parameterized.named_parameters(
        ("1d case 1", (2,), 4, 5),
        (
            "1d case 2",
            (2,),
            8,
            10000,
        ),
        (
            "2d",
            (2, 3),
            8,
            10000,
        ),
    )
    def test_shapes(self, in_shape: tuple[int, ...], dim: int, max_period: int) -> None:
        """Test output shapes under different device condition.

        Args:
            in_shape: input shape.
            dim: embedding dimension, assume to be evenly divided by two.
            max_period: controls the minimum frequency of the embeddings.
        """
        rng = jax.random.PRNGKey(0)
        x = jax.random.uniform(
            rng,
            shape=in_shape,
        )
        out = self.variant(
            partial(sinusoidal_positional_embedding, dim=dim, max_period=max_period)
        )(x)
        chex.assert_shape(out, (*in_shape, dim))


class TestMLP(chex.TestCase):
    """Test MLP."""

    emb_size: int = 4
    output_size: int = 8

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d",
            (5,),
        ),
        (
            "2d",
            (5, 4),
        ),
        (
            "3d",
            (3, 5, 4),
        ),
    )
    def test_shape(
        self,
        in_shape: tuple[int, ...],
    ) -> None:
        """Test FeedForwardBlock output shapes.

        Args:
            in_shape: input tensor shape.
        """
        rng = {"params": jax.random.PRNGKey(0)}
        mlp = MLP(
            emb_size=self.emb_size,
            output_size=self.output_size,
        )
        out, _ = self.variant(mlp.init_with_output)(rng, jnp.ones(in_shape))
        chex.assert_shape(out, (*in_shape[:-1], self.output_size))
