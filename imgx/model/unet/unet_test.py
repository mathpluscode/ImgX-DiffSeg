"""Test Unet related classes and functions."""

import chex
import jax
import jax.numpy as jnp
import optax
import pytest
from absl.testing import parameterized
from chex._src import fake

from imgx.model import Unet


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestUnet(chex.TestCase):
    """Test the class Unet."""

    batch_size = 2
    in_channels = 1
    out_channels = 2
    num_res_blocks = 1
    num_timesteps = 2
    num_heads = 2

    @parameterized.product(
        (
            {
                "in_shape": (15, 16),
                "kernel_size": 3,
                "scale_factor": 2,
                "num_spatial_dims": 2,
            },
            {
                "in_shape": (15, 16, 17),
                "kernel_size": 3,
                "scale_factor": 2,
                "num_spatial_dims": 2,
            },
            {
                "in_shape": (15, 16, 17),
                "kernel_size": 3,
                "scale_factor": 2,
                "num_spatial_dims": 3,
            },
            {
                "in_shape": (13, 14, 15),
                "kernel_size": 5,
                "scale_factor": 1,
                "num_spatial_dims": 2,
            },
        ),
        with_time=[True, False],
        patch_size=[2, 4],
    )
    def test_output_shape(
        self,
        in_shape: tuple[int, ...],
        patch_size: int,
        kernel_size: int,
        scale_factor: int,
        num_spatial_dims: int,
        with_time: bool,
    ) -> None:
        """Test output shape.

        Args:
            in_shape: input shape
            patch_size: patch size for patch embedding.
            scale_factor: convolution stride for down-sampling/up-sampling.
            kernel_size: convolution kernel size, the value(s) should be odd.
            num_spatial_dims: number of spatial dimensions.
            with_time: with time or not.
        """
        rng = {"params": jax.random.PRNGKey(0)}
        num_channels = (8, 16)

        unet = Unet(
            num_spatial_dims=num_spatial_dims,
            out_channels=self.out_channels,
            num_channels=num_channels,
            num_res_blocks=self.num_res_blocks,
            num_heads=self.num_heads,
            patch_size=patch_size,
            kernel_size=kernel_size,
            scale_factor=scale_factor,
        )

        image = jnp.ones((self.batch_size, *in_shape, self.in_channels))
        if with_time:
            mask = jnp.ones((self.batch_size, *in_shape, self.out_channels))
            t = jnp.ones((self.batch_size,), dtype=jnp.int32)
        else:
            mask = None
            t = None

        out, _ = unet.init_with_output(rng, image, mask, t)
        chex.assert_shape(out, (self.batch_size, *in_shape, self.out_channels))

    @chex.all_variants()
    @parameterized.named_parameters(
        ("Unet without time with remat", False, True),
        ("Unet with time with remat", True, True),
        ("Unet without time without remat", False, False),
        ("Unet with time without remat", True, False),
    )
    def test_output_shape_variants(
        self,
        with_time: bool,
        remat: bool,
    ) -> None:
        """Test Unet output shape under different device variants.

        Args:
            with_time: with time or not.
            remat: remat or not.
        """
        rng = {"params": jax.random.PRNGKey(0)}
        num_spatial_dims = 2
        patch_size = 4
        kernel_size = 3
        scale_factor = 2
        in_shape = (14, 15, 16)
        num_channels = (8, 16)

        unet = Unet(
            num_spatial_dims=num_spatial_dims,
            out_channels=self.out_channels,
            num_channels=num_channels,
            num_res_blocks=self.num_res_blocks,
            num_heads=self.num_heads,
            patch_size=patch_size,
            kernel_size=kernel_size,
            scale_factor=scale_factor,
            remat=remat,
        )

        image = jnp.ones((self.batch_size, *in_shape, self.in_channels))
        if with_time:
            mask = jnp.ones((self.batch_size, *in_shape, self.out_channels))
            t = jnp.ones((self.batch_size,), dtype=jnp.int32)
        else:
            mask = None
            t = None

        out, _ = self.variant(unet.init_with_output)(rng, image, mask, t)
        chex.assert_shape(out, (self.batch_size, *in_shape, self.out_channels))

    @chex.variants(with_jit=True)
    @pytest.mark.slow()
    @parameterized.product(
        num_spatial_dims=[2, 3],
        with_time=[True, False],
    )
    def test_output_real_shape(
        self,
        num_spatial_dims: int,
        with_time: bool,
    ) -> None:
        """Test UNet3D output shape with real setting.

        Args:
            num_spatial_dims: number of spatial dimensions.
            with_time: with time or not.
        """
        rng = {"params": jax.random.PRNGKey(0)}
        patch_size = 4
        kernel_size = 3
        scale_factor = 2
        in_shape = (256, 256, 32)
        num_channels = (4, 4, 8)

        unet = Unet(
            num_spatial_dims=num_spatial_dims,
            out_channels=self.out_channels,
            num_channels=num_channels,
            num_res_blocks=self.num_res_blocks,
            num_heads=self.num_heads,
            patch_size=patch_size,
            kernel_size=kernel_size,
            scale_factor=scale_factor,
        )

        image = jnp.ones((self.batch_size, *in_shape, self.in_channels))
        if with_time:
            mask = jnp.ones((self.batch_size, *in_shape, self.out_channels))
            t = jnp.ones((self.batch_size,), dtype=jnp.int32)
        else:
            mask = None
            t = None

        out, _ = self.variant(unet.init_with_output)(rng, image, mask, t)
        chex.assert_shape(out, (self.batch_size, *in_shape, self.out_channels))

    @parameterized.named_parameters(
        ("Unet without time", False, 34194, 27.610577),
        ("Unet with time", True, 36106, 29.420588),
    )
    def test_params_count(
        self,
        with_time: bool,
        expected_params_count: int,
        expected_params_norm: float,
    ) -> None:
        """Count network parameters.

        Changing layer/model names may change the initial parameters norm.

        Args:
            with_time: with time or not.
            expected_params_count: expected number of parameters.
            expected_params_norm: expected parameters norm.
        """
        rng = {"params": jax.random.PRNGKey(0)}
        num_spatial_dims = 2
        patch_size = 4
        kernel_size = 3
        scale_factor = 2
        in_shape = (14, 15, 16)
        num_channels = (8, 16)
        remat = True

        unet = Unet(
            num_spatial_dims=num_spatial_dims,
            out_channels=self.out_channels,
            num_channels=num_channels,
            num_res_blocks=self.num_res_blocks,
            num_heads=self.num_heads,
            patch_size=patch_size,
            kernel_size=kernel_size,
            scale_factor=scale_factor,
            remat=remat,
        )

        image = jnp.ones((self.batch_size, *in_shape, self.in_channels))
        if with_time:
            mask = jnp.ones((self.batch_size, *in_shape, self.out_channels))
            t = jnp.ones((self.batch_size,), dtype=jnp.int32)
        else:
            mask = None
            t = None

        _, variables = unet.init_with_output(rng, image, mask, t)
        params = variables["params"]

        got_params_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        assert got_params_count == expected_params_count

        got_params_norm = optax.global_norm(params)
        assert got_params_norm == expected_params_norm
