"""Test Unet bottom encoder related classes and functions."""

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from chex._src import fake

from imgx.model.unet.bottom_encoder import BottomImageEncoderUnet


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestBottomEncoderUnet(chex.TestCase):
    """Test the class BottomEncoderUnet."""

    batch_size = 2
    model_size = 4

    @chex.all_variants()
    @parameterized.named_parameters(
        ("2D", (5, 6)),
        ("3D", (5, 6, 7)),
    )
    def test_output_shape(
        self,
        spatial_shape: tuple[int, ...],
    ) -> None:
        """Test output shape."""

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            image_emb: jnp.ndarray,
        ) -> jnp.ndarray:
            """Forward."""
            return BottomImageEncoderUnet()(image_emb)

        rng = jax.random.PRNGKey(0)
        dummy_image = jax.random.uniform(
            rng, shape=(self.batch_size, *spatial_shape, self.model_size)
        )
        out_image_emb = forward(dummy_image)

        chex.assert_shape(
            out_image_emb, (self.batch_size, *spatial_shape, self.model_size)
        )
