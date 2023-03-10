"""Test Unet related classes and functions."""
from typing import Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from chex._src import fake

from imgx.model import Unet3dSliceTime, Unet3dTime


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestTimeUnet3d(chex.TestCase):
    """Test the class TimeUnet3d and Unet3dSliceTime."""

    batch_size = 2
    in_channels = 1
    out_channels = 2
    num_timesteps = 4

    @parameterized.product(
        (
            {
                "in_shape": (15, 16, 17),
                "kernel_size": 3,
                "scale_factor": 2,
            },
            {
                "in_shape": (13, 14, 15),
                "kernel_size": 5,
                "scale_factor": 1,
            },
            {
                "in_shape": (29, 30, 31),
                "kernel_size": 5,
                "scale_factor": 2,
            },
            {
                "in_shape": (53, 54, 55),
                "kernel_size": 5,
                "scale_factor": 3,
            },
        ),
        model_cls=[Unet3dTime, Unet3dSliceTime],
    )
    def test_output_shape(
        self,
        in_shape: Tuple[int, int, int],
        kernel_size: int,
        scale_factor: int,
        model_cls: hk.Module,
    ) -> None:
        """Test UNet3D output shape.

        Args:
            in_shape: input shape
            scale_factor: convolution stride for down-sampling/up-sampling.
            kernel_size: convolution kernel size, the value(s) should be odd.
            model_cls: model to be tested.
        """
        channels = (2, 4, 2)

        @hk.testing.transform_and_run()
        def forward(
            x: jnp.ndarray,
            t: jnp.ndarray,
        ) -> jnp.ndarray:
            """Forward function of Unet.

            Args:
                x: input.
                t: time.

            Returns:
                Network prediction.
            """
            net = model_cls(
                in_shape=in_shape,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                num_channels=channels,
                kernel_size=kernel_size,
                scale_factor=scale_factor,
                num_timesteps=self.num_timesteps,
            )
            return net(x, t)

        rng = jax.random.PRNGKey(0)
        rng, rng_t = jax.random.split(rng)
        dummy_image = jax.random.uniform(
            rng, shape=(self.batch_size, *in_shape, self.in_channels)
        )
        dummy_t = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=0, maxval=self.num_timesteps
        )
        out = forward(dummy_image, dummy_t)

        chex.assert_shape(out, (self.batch_size, *in_shape, self.out_channels))

    @chex.all_variants
    @parameterized.named_parameters(
        (
            "Unet3dTime",
            Unet3dTime,
        ),
        ("Unet3dSliceTime", Unet3dSliceTime),
    )
    def test_output_shape_variants(
        self,
        model_cls: hk.Module,
    ) -> None:
        """Test UNet3D output shape under different device variants.

        Args:
            model_cls: model to be tested.
        """
        kernel_size = 3
        scale_factor = 2
        in_shape = (14, 15, 16)
        channels = (2, 4)

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            x: jnp.ndarray,
            t: jnp.ndarray,
        ) -> jnp.ndarray:
            """Forward function of Unet.

            Args:
                x: input.
                t: time.

            Returns:
                Network prediction.
            """
            net = model_cls(
                in_shape=in_shape,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                num_channels=channels,
                kernel_size=kernel_size,
                scale_factor=scale_factor,
                num_timesteps=self.num_timesteps,
            )
            return net(x, t)

        rng = jax.random.PRNGKey(0)
        rng, rng_t = jax.random.split(rng)
        dummy_image = jax.random.uniform(
            rng, shape=(self.batch_size, *in_shape, self.in_channels)
        )
        dummy_t = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=0, maxval=self.num_timesteps
        )
        out = forward(dummy_image, dummy_t)

        chex.assert_shape(out, (self.batch_size, *in_shape, self.out_channels))

    @parameterized.named_parameters(
        (
            "Unet3dTime",
            Unet3dTime,
        ),
        ("Unet3dSliceTime", Unet3dSliceTime),
    )
    def test_output_real_shape(
        self,
        model_cls: hk.Module,
    ) -> None:
        """Test UNet3D output shape with real setting.

        Args:
            model_cls: model to be tested.
        """
        kernel_size = 3
        scale_factor = 2
        in_shape = (256, 256, 48)
        channels = (2, 2, 2, 2)

        @hk.testing.transform_and_run()
        def forward(
            x: jnp.ndarray,
            t: jnp.ndarray,
        ) -> jnp.ndarray:
            """Forward function of Unet.

            Args:
                x: input.
                t: time.

            Returns:
                Network prediction.
            """
            net = model_cls(
                in_shape=in_shape,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                num_channels=channels,
                kernel_size=kernel_size,
                scale_factor=scale_factor,
                num_timesteps=self.num_timesteps,
            )
            return net(x, t)

        rng = jax.random.PRNGKey(0)
        rng, rng_t = jax.random.split(rng)
        dummy_image = jax.random.uniform(
            rng, shape=(self.batch_size, *in_shape, self.in_channels)
        )
        dummy_t = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=0, maxval=self.num_timesteps
        )
        out = forward(dummy_image, dummy_t)

        chex.assert_shape(out, (self.batch_size, *in_shape, self.out_channels))
