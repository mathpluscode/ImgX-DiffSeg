"""Test conv layers."""

import chex
import jax
from absl.testing import parameterized
from chex._src import fake

from imgx.model.conv import ConvDownSample, ConvResBlock, ConvUpSample


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


class TestConvResBlock(chex.TestCase):
    """Test ConvResBlock."""

    batch = 2

    @chex.all_variants()
    @parameterized.product(
        in_shape=[(12,), (12, 13), (12, 13, 14)],
        has_t=[True, False],
        dropout=[0.0, 0.5, 1.0],
        is_train=[True, False],
        remat=[True, False],
    )
    def test_shapes(
        self,
        in_shape: tuple[int, ...],
        has_t: bool,
        dropout: float,
        is_train: bool,
        remat: bool,
    ) -> None:
        """Test output shapes.

        Args:
            in_shape: input shape, without batch, channel.
            has_t: whether has time embedding.
            dropout: dropout rate.
            is_train: whether in training mode.
            remat: remat or not.
        """
        kernel_size = (3,) * len(in_shape)
        in_channels = 1
        t_channels = 2
        out_channels = 1
        rng = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}
        conv = ConvResBlock(
            out_channels=out_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            remat=remat,
        )
        x = jax.random.uniform(
            jax.random.PRNGKey(0),
            shape=(self.batch, *in_shape, in_channels),
        )
        t_emb = None
        if has_t:
            t_emb = jax.random.uniform(
                jax.random.PRNGKey(0),
                shape=(self.batch, t_channels),
            )
        out, _ = self.variant(conv.init_with_output, static_argnums=(1,))(rng, is_train, x, t_emb)
        chex.assert_shape(out, (self.batch, *in_shape, out_channels))
