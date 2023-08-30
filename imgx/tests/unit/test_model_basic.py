"""Test basic functions for model."""


from functools import partial

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from chex._src import fake

from imgx.model.basic import (
    MLP,
    adaptive_norm,
    layer_norm,
    sinusoidal_positional_embedding,
)


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


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
    def test_shapes(
        self, in_shape: tuple[int, ...], dim: int, max_period: int
    ) -> None:
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
            partial(
                sinusoidal_positional_embedding, dim=dim, max_period=max_period
            )
        )(x)
        chex.assert_shape(out, (*in_shape, dim))


class TestAdaptiveNorm(chex.TestCase):
    """Test adaptive_norm."""

    emb_size: int = 4

    @chex.all_variants()
    @parameterized.named_parameters(
        ("1d", (5,), (5,)),
        (
            "2d",
            (5, 4),
            (5, 4),
        ),
        (
            "3d - cond compatible",
            (3, 5, 4),
            (3, 1, 4),
        ),
    )
    def test_shape(
        self,
        x_shape: tuple[int, ...],
        cond_shape: tuple[int, ...],
    ) -> None:
        """Test FeedForwardBlock output shapes.

        Args:
            x_shape: shape of input x.
            cond_shape: shape of input cond.
        """

        adaptive_layer_norm = partial(adaptive_norm, norm_fn=layer_norm)

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            x: jnp.ndarray,
            cond: jnp.ndarray,
        ) -> jnp.ndarray:
            return adaptive_layer_norm(x, cond)

        key = jax.random.PRNGKey(0)
        dummy_x = jax.random.uniform(key, shape=x_shape)
        dummy_cond = jax.random.uniform(key, shape=cond_shape)
        out = forward(dummy_x, dummy_cond)
        chex.assert_shape(out, x_shape)


class TestMLP(chex.TestCase):
    """Test MLP."""

    emb_size: int = 4
    output_size: int = 8
    initializer: hk.initializers.Initializer = hk.initializers.VarianceScaling(
        1.0
    )

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

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            x: jnp.ndarray,
        ) -> jnp.ndarray:
            """Forward function.

            Args:
                x: input.

            Returns:
                Network prediction.
            """
            mlp = MLP(
                emb_size=self.emb_size,
                output_size=self.output_size,
                initializer=self.initializer,
            )
            return mlp(x)

        key = jax.random.PRNGKey(0)
        dummy_input = jax.random.uniform(key, shape=in_shape)
        out = forward(dummy_input)
        chex.assert_shape(out, (*in_shape[:-1], self.output_size))
