"""Test Unet related classes and functions."""

import chex
import haiku as hk
import jax
import jax.numpy as jnp
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
            scale_factor: convolution stride for down-sampling/up-sampling.
            kernel_size: convolution kernel size, the value(s) should be odd.
            num_spatial_dims: number of spatial dimensions.
            with_time: with time or not.
        """
        channels = (2, 4)

        @hk.testing.transform_and_run()
        def forward(
            image: jnp.ndarray,
            mask: jnp.ndarray,
            t: jnp.ndarray,
        ) -> jnp.ndarray:
            """Forward function of Unet."""
            if not with_time:
                mask = None
                t = None
            return Unet(
                num_spatial_dims=num_spatial_dims,
                out_channels=self.out_channels,
                num_channels=channels,
                num_res_blocks=self.num_res_blocks,
                patch_size=patch_size,
                kernel_size=kernel_size,
                scale_factor=scale_factor,
            )(image, mask, t)

        rng = jax.random.PRNGKey(0)
        rng_image, rng_mask, rng_t = jax.random.split(rng, 3)
        dummy_image = jax.random.uniform(
            rng_image, shape=(self.batch_size, *in_shape, self.in_channels)
        )
        dummy_mask = jax.random.uniform(
            rng_mask, shape=(self.batch_size, *in_shape, self.out_channels)
        )
        dummy_t = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=0, maxval=self.num_timesteps
        )
        outs = forward(dummy_image, dummy_mask, dummy_t)
        chex.assert_shape(outs, (self.batch_size, *in_shape, self.out_channels))

    @chex.all_variants()
    @parameterized.named_parameters(
        ("Unet without time", False),
        ("Unet with time", True),
    )
    def test_output_shape_variants(
        self,
        with_time: bool,
    ) -> None:
        """Test Unet output shape under different device variants.

        Args:
            with_time: with time or not.
        """
        num_spatial_dims = 2
        patch_size = 4
        kernel_size = 3
        scale_factor = 2
        in_shape = (14, 15, 16)
        channels = (2, 4)

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            image: jnp.ndarray,
            mask: jnp.ndarray,
            t: jnp.ndarray,
        ) -> jnp.ndarray:
            """Forward function."""
            if not with_time:
                mask = None
                t = None
            return Unet(
                num_spatial_dims=num_spatial_dims,
                out_channels=self.out_channels,
                num_channels=channels,
                num_res_blocks=self.num_res_blocks,
                patch_size=patch_size,
                kernel_size=kernel_size,
                scale_factor=scale_factor,
            )(image, mask, t)

        rng = jax.random.PRNGKey(0)
        rng_image, rng_mask, rng_t = jax.random.split(rng, 3)
        dummy_image = jax.random.uniform(
            rng_image, shape=(self.batch_size, *in_shape, self.in_channels)
        )
        dummy_mask = jax.random.uniform(
            rng_mask, shape=(self.batch_size, *in_shape, self.out_channels)
        )
        dummy_t = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=0, maxval=self.num_timesteps
        )
        out = forward(dummy_image, dummy_mask, dummy_t)

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
        patch_size = 4
        kernel_size = 3
        scale_factor = 2
        in_shape = (256, 256, 32)
        channels = (1, 2, 4)

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            image: jnp.ndarray,
            mask: jnp.ndarray,
            t: jnp.ndarray,
        ) -> jnp.ndarray:
            """Forward function."""
            if not with_time:
                mask = None
                t = None
            return Unet(
                num_spatial_dims=num_spatial_dims,
                out_channels=self.out_channels,
                num_channels=channels,
                num_res_blocks=self.num_res_blocks,
                patch_size=patch_size,
                kernel_size=kernel_size,
                scale_factor=scale_factor,
            )(image, mask, t)

        rng = jax.random.PRNGKey(0)
        rng_image, rng_mask, rng_t = jax.random.split(rng, 3)
        dummy_image = jax.random.uniform(
            rng_image, shape=(self.batch_size, *in_shape, self.in_channels)
        )
        dummy_mask = jax.random.uniform(
            rng_mask, shape=(self.batch_size, *in_shape, self.out_channels)
        )
        dummy_t = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=0, maxval=self.num_timesteps
        )
        out = forward(dummy_image, dummy_mask, dummy_t)

        chex.assert_shape(out, (self.batch_size, *in_shape, self.out_channels))
