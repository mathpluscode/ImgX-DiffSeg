"""Test Unet bottom encoder related classes and functions."""

import chex
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
    model_size = 16
    num_heads = 2
    t_size = 3

    @chex.all_variants()
    @parameterized.product(
        spatial_shape=[(5, 6), (5, 6, 7)],
        with_time=[True, False],
        is_train=[True, False],
    )
    def test_output_shape(
        self,
        spatial_shape: tuple[int, ...],
        with_time: bool,
        is_train: bool,
    ) -> None:
        """Test output shape."""
        rng = {"params": jax.random.PRNGKey(0)}
        kernel_size = (3,) * len(spatial_shape)
        encoder = BottomImageEncoderUnet(num_heads=self.num_heads, kernel_size=kernel_size)
        image_emb = jnp.ones((self.batch_size, *spatial_shape, self.model_size))
        t_emb = jnp.ones((self.batch_size, self.t_size)) if with_time else None
        out, _ = self.variant(encoder.init_with_output, static_argnums=(1,))(
            rng, is_train, image_emb, t_emb
        )
        chex.assert_shape(out, (self.batch_size, *spatial_shape, self.model_size))
