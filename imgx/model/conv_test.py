"""Test conv layers."""

import chex
import jax
from absl.testing import parameterized
from chex._src import fake

from imgx.model.conv import ConvDownSample, ConvUpSample


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestConvDownSample(chex.TestCase):
    """Test ConvDownSample."""

    batch = 2

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d",
            (12,),
            (2,),
            (6,),
        ),
        (
            "2d",
            (12, 13),
            (2, 2),
            (6, 7),
        ),
        (
            "2d - different scale factors",
            (12, 13),
            (4, 2),
            (3, 7),
        ),
        (
            "3d - large scale factor",
            (2, 4, 8),
            (4, 4, 4),
            (1, 1, 2),
        ),
        (
            "3d",
            (12, 13, 14),
            (2, 2, 2),
            (6, 7, 7),
        ),
    )
    def test_shapes(
        self,
        in_shape: tuple[int, ...],
        scale_factor: tuple[int, ...],
        out_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes under different device condition.

        Args:
            in_shape: input shape, without batch, channel.
            scale_factor: downsample factor.
            out_shape: output shape, without batch, channel.
        """
        in_channels = 1
        out_channels = 1
        rng = {"params": jax.random.PRNGKey(0)}
        conv = ConvDownSample(
            out_channels=out_channels,
            scale_factor=scale_factor,
        )
        x = jax.random.uniform(
            jax.random.PRNGKey(0),
            shape=(self.batch, *in_shape, in_channels),
        )
        out, _ = self.variant(conv.init_with_output)(rng, x)
        chex.assert_shape(out, (self.batch, *out_shape, out_channels))


class TestConvUpSample(chex.TestCase):
    """Test ConvUpSample."""

    batch = 2

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d",
            (3,),
            (2,),
            (6,),
        ),
        (
            "2d",
            (3, 4),
            (2, 2),
            (6, 8),
        ),
        (
            "2d - different scale factors",
            (3, 4),
            (4, 2),
            (12, 8),
        ),
        (
            "3d",
            (2, 3, 4),
            (2, 2, 2),
            (4, 6, 8),
        ),
    )
    def test_shapes(
        self,
        in_shape: tuple[int, ...],
        scale_factor: tuple[int, ...],
        out_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes under different device condition.

        Args:
            in_shape: input shape, without batch, channel.
            scale_factor: up-sampler factor.
            out_shape: output shape, without batch, channel.
        """
        in_channels = 1
        out_channels = 1
        rng = {"params": jax.random.PRNGKey(0)}
        conv = ConvUpSample(
            out_channels=out_channels,
            scale_factor=scale_factor,
        )
        x = jax.random.uniform(
            jax.random.PRNGKey(0),
            shape=(self.batch, *in_shape, in_channels),
        )
        out, _ = self.variant(conv.init_with_output)(rng, x)
        chex.assert_shape(out, (self.batch, *out_shape, out_channels))
