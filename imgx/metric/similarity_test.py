"""Test similarity functions."""

from __future__ import annotations

import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.metric.similarity import nrmsd, psnr, ssim


def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestSSIM(chex.TestCase):
    """Test SSIM."""

    @parameterized.product(
        (
            {"image_shape": (13,), "in_channels": 1},
            {"image_shape": (11, 13), "in_channels": 1},
            {"image_shape": (11, 13), "in_channels": 3},
            {"image_shape": (11, 13, 15), "in_channels": 2},
        ),
        kernel_type=("gaussian", "uniform"),
    )
    def test_shapes(
        self,
        image_shape: tuple[int, ...],
        in_channels: int,
        kernel_type: str,
    ) -> None:
        """Test return shapes.

        Args:
            image_shape: image shapes.
            in_channels: number of input channels.
            kernel_type: type of kernel, "gaussian" or "uniform".
        """
        batch_size = 2

        image = jnp.ones((batch_size, *image_shape, in_channels))
        got = ssim(image, image, kernel_type=kernel_type)

        chex.assert_shape(got, (batch_size,))
        chex.assert_trees_all_close(got, jnp.ones((batch_size,)))

    @parameterized.named_parameters(
        (
            "1d",
            # x
            # mean(x)=0.6
            # mean(x**2)=0.6
            # var(x)=0.24
            np.array([0.0, 1.0, 0.0, 1.0, 1.0]),
            # mean(y)=0.4
            # mean(y**2)=0.252
            # var(y)=0.092
            # mean(xy)=0.36
            # covar(x,y)=0.12
            np.array([0.2, 0.8, 0, 0.3, 0.7]),
            5,
            "uniform",
            (2 * 0.6 * 0.4 + 0.0001)
            * (2 * 0.12 + 0.0009)
            / ((0.6**2 + 0.4**2 + 0.0001) * (0.24 + 0.092 + 0.0009)),
        ),
    )
    def test_values(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        kernel_size: int,
        kernel_type: str,
        expected: float,
    ) -> None:
        """Test return values.

        Args:
            image1: image 1.
            image2: image 2.
            kernel_size: kernel size.
            kernel_type: type of kernel, "gaussian" or "uniform".
            expected: expected SSIM value.
        """
        image1 = image1[None, ..., None]
        image2 = image2[None, ..., None]
        got = ssim(
            jnp.array(image1), jnp.array(image2), kernel_size=kernel_size, kernel_type=kernel_type
        )
        chex.assert_trees_all_close(got, jnp.array([expected]))


class TestPSNR(chex.TestCase):
    """Test PSNR."""

    @parameterized.product(
        (
            {"image_shape": (13,), "in_channels": 1},
            {"image_shape": (11, 13), "in_channels": 1},
            {"image_shape": (11, 13), "in_channels": 3},
            {"image_shape": (11, 13, 15), "in_channels": 2},
        ),
    )
    def test_shapes(
        self,
        image_shape: tuple[int, ...],
        in_channels: int,
    ) -> None:
        """Test return shapes.

        Args:
            image_shape: image shapes.
            in_channels: number of input channels.
            kernel_type: type of kernel, "gaussian" or "uniform".
        """
        batch_size = 2

        image = jnp.ones((batch_size, *image_shape, in_channels))
        got = psnr(image, image)

        chex.assert_shape(got, (batch_size,))
        chex.assert_tree_all_finite(got)


class TestNRMSD(chex.TestCase):
    """Test NRMSD."""

    @parameterized.product(
        (
            {"image_shape": (13,), "in_channels": 1},
            {"image_shape": (11, 13), "in_channels": 1},
            {"image_shape": (11, 13), "in_channels": 3},
            {"image_shape": (11, 13, 15), "in_channels": 2},
        ),
    )
    def test_shapes(
        self,
        image_shape: tuple[int, ...],
        in_channels: int,
    ) -> None:
        """Test return shapes.

        Args:
            image_shape: image shapes.
            in_channels: number of input channels.
            kernel_type: type of kernel, "gaussian" or "uniform".
        """
        batch_size = 2

        image = jnp.zeros((batch_size, *image_shape, in_channels))
        got = nrmsd(image, image)

        chex.assert_shape(got, (batch_size,))
        chex.assert_tree_all_finite(got)
