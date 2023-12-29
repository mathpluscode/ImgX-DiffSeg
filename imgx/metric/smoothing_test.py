"""Test image/label smoothing related functions."""
import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.metric.smoothing import gaussian_kernel, get_conv, smooth_label


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestGaussianKernel(chex.TestCase):
    """Test gaussian_kernel."""

    @parameterized.named_parameters(
        (
            "1d-sigma1",
            1,
            1.0,
            3,
            np.array(
                [
                    np.exp(-0.5) / (1 + 2 * np.exp(-0.5)),
                    1 / (1 + 2 * np.exp(-0.5)),
                    np.exp(-0.5) / (1 + 2 * np.exp(-0.5)),
                ]
            ),
        ),
        (
            "1d-sigma2",
            1,
            2.0,
            3,
            np.array(
                [
                    np.exp(-0.125) / (1 + 2 * np.exp(-0.125)),
                    1 / (1 + 2 * np.exp(-0.125)),
                    np.exp(-0.125) / (1 + 2 * np.exp(-0.125)),
                ]
            ),
        ),
        (
            "2d-sigma2",
            2,
            2.0,
            3,
            np.array(
                [
                    [
                        np.exp(-0.25) / (1 + 4 * np.exp(-0.125) + 4 * np.exp(-0.25)),
                        np.exp(-0.125) / (1 + 4 * np.exp(-0.125) + 4 * np.exp(-0.25)),
                        np.exp(-0.25) / (1 + 4 * np.exp(-0.125) + 4 * np.exp(-0.25)),
                    ],
                    [
                        np.exp(-0.125) / (1 + 4 * np.exp(-0.125) + 4 * np.exp(-0.25)),
                        1 / (1 + 4 * np.exp(-0.125) + 4 * np.exp(-0.25)),
                        np.exp(-0.125) / (1 + 4 * np.exp(-0.125) + 4 * np.exp(-0.25)),
                    ],
                    [
                        np.exp(-0.25) / (1 + 4 * np.exp(-0.125) + 4 * np.exp(-0.25)),
                        np.exp(-0.125) / (1 + 4 * np.exp(-0.125) + 4 * np.exp(-0.25)),
                        np.exp(-0.25) / (1 + 4 * np.exp(-0.125) + 4 * np.exp(-0.25)),
                    ],
                ]
            ),
        ),
    )
    def test_values(
        self,
        num_spatial_dims: int,
        kernel_sigma: float,
        kernel_size: int,
        expected: np.ndarray,
    ) -> None:
        """Test return values.

        Args:
            num_spatial_dims: number of spatial dimensions.
            kernel_sigma: kernel sigma.
            kernel_size: kernel size.
            expected: expected kernel.
        """
        got = gaussian_kernel(
            num_spatial_dims=num_spatial_dims,
            kernel_sigma=kernel_sigma,
            kernel_size=kernel_size,
        )
        chex.assert_trees_all_close(got, expected)

    @parameterized.named_parameters(
        (
            "1d",
            1,
        ),
        (
            "2d",
            2,
        ),
        ("3d", 3),
        ("4d", 3),
    )
    def test_shapes(
        self,
        num_spatial_dims: int,
    ) -> None:
        """Test return shapes.

        Args:
            num_spatial_dims: number of spatial dimensions.
        """
        kernel_size = 5
        got = gaussian_kernel(
            num_spatial_dims=num_spatial_dims,
            kernel_sigma=1.0,
            kernel_size=kernel_size,
        )
        chex.assert_shape(got, (kernel_size,) * num_spatial_dims)


class TestSmoothLabel(chex.TestCase):
    """Test smooth_label."""

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "no smoothing - exclusive",
            np.array([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            True,
            0.0,
            np.array([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
        ),
        (
            "no smoothing - non exclusive",
            np.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            False,
            0.0,
            np.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
        ),
        (
            "smoothing=1 - exclusive",
            np.array([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            True,
            1.0,
            np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]) / 3.0,
        ),
        (
            "smoothing=1 - non exclusive",
            np.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            False,
            1.0,
            np.array([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]),
        ),
        (
            "smoothing=0.1 - exclusive",
            np.array([[[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]]),
            True,
            0.1,
            np.array([[[0.05, 0.95], [0.05, 0.95], [0.95, 0.05]]]),
        ),
        (
            "smoothing=0.1 - non exclusive",
            np.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            False,
            0.1,
            np.array([[[0.95, 0.05, 0.95], [0.05, 0.95, 0.05], [0.95, 0.05, 0.05]]]),
        ),
    )
    def test_values(
        self,
        mask: np.ndarray,
        classes_are_exclusive: bool,
        label_smoothing: float,
        expected: np.ndarray,
    ) -> None:
        """Test loss values.

        Args:
            mask: unscaled prediction, of shape (..., num_classes).
            classes_are_exclusive: if False, each element can be assigned to multiple classes.
            label_smoothing: label smoothing factor between 0 and 1, 0.0 means no smoothing.
            expected: expected output.
        """
        got = self.variant(smooth_label)(
            mask=mask,
            classes_are_exclusive=classes_are_exclusive,
            label_smoothing=label_smoothing,
        )
        chex.assert_trees_all_close(got, expected)


class TestConv(chex.TestCase):
    """Test get_conv."""

    gaussian_kernel_sigma1 = (
        np.exp(-0.5) / (1 + 2 * np.exp(-0.5)),
        1 / (1 + 2 * np.exp(-0.5)),
        np.exp(-0.5) / (1 + 2 * np.exp(-0.5)),
    )

    @chex.all_variants()
    @parameterized.product(
        in_channels=[1, 2],
        shape=[(3,), (2, 3), (2, 3, 4), (2, 3, 4, 5)],
        kernel_type=["gaussian", "uniform"],
        padding=["SAME", "VALID"],
    )
    def test_shapes(
        self,
        shape: tuple[int, ...],
        in_channels: int,
        kernel_type: str,
        padding: str,
    ) -> None:
        """Test return shapes.

        Args:
            shape: input spatial shape.
            in_channels: number of input channels.
            kernel_type: kernel type.
            padding: SAME or VALID.
        """
        batch = 2
        kernel_size = min(shape)
        conv = get_conv(
            num_spatial_dims=len(shape),
            kernel_sigma=1.0,
            kernel_size=kernel_size,
            kernel_type=kernel_type,
            padding=padding,
        )
        x = jnp.ones((batch, *shape, in_channels))
        got = self.variant(conv)(x)
        out_shape = shape if padding == "SAME" else tuple(x - kernel_size + 1 for x in shape)
        chex.assert_shape(got, (batch, *out_shape, in_channels))

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d - 1 channel - uniform - valid",
            np.array([[0.0], [1.0], [0.0], [0.0], [0.0]]),
            1.0,
            3,
            "uniform",
            "VALID",
            np.array([[1.0 / 3], [1.0 / 3], [0.0]]),
        ),
        (
            "1d - 2 channels - uniform - valid",
            np.array([[0.0, 0.2], [1.0, 0.4], [0.0, 0.5], [0.0, 0.3], [0.0, 0.6]]),
            1.0,
            3,
            "uniform",
            "VALID",
            np.array([[1.0 / 3, 1.1 / 3], [1.0 / 3, 0.4], [0.0, 1.4 / 3]]),
        ),
        (
            "1d -  1 channel - uniform - same",
            np.array([[0.0], [1.0], [0.0], [0.0], [0.0]]),
            1.0,
            3,
            "uniform",
            "SAME",
            np.array([[1.0 / 3], [1.0 / 3], [1.0 / 3], [0.0], [0.0]]),
        ),
        (
            "1d - 2 channels - gaussian - valid",
            np.array([[0.0, 0.2], [1.0, 0.4], [0.0, 0.5], [0.0, 0.3], [0.0, 0.6]]),
            1.0,
            3,
            "gaussian",
            "VALID",
            np.array(
                [
                    [
                        gaussian_kernel_sigma1[1],
                        0.2 * gaussian_kernel_sigma1[0]
                        + 0.4 * gaussian_kernel_sigma1[1]
                        + 0.5 * gaussian_kernel_sigma1[2],
                    ],
                    [
                        gaussian_kernel_sigma1[0],
                        0.4 * gaussian_kernel_sigma1[0]
                        + 0.5 * gaussian_kernel_sigma1[1]
                        + 0.3 * gaussian_kernel_sigma1[2],
                    ],
                    [
                        0.0,
                        0.5 * gaussian_kernel_sigma1[0]
                        + 0.3 * gaussian_kernel_sigma1[1]
                        + 0.6 * gaussian_kernel_sigma1[2],
                    ],
                ]
            ),
        ),
    )
    def test_values(
        self,
        x: np.ndarray,
        kernel_sigma: float,
        kernel_size: int,
        kernel_type: str,
        padding: str,
        expected: np.ndarray,
    ) -> None:
        """Test return shapes.

        Args:
            x: input without batch axis.
            kernel_sigma: sigma for Gaussian kernel.
            kernel_size: size for Gaussian kernel.
            kernel_type: type of kernel, "gaussian" or "uniform".
            padding: padding type, "SAME" or "VALID".
            expected: expected output.
        """
        conv = get_conv(
            num_spatial_dims=x.ndim - 1,
            kernel_sigma=kernel_sigma,
            kernel_size=kernel_size,
            kernel_type=kernel_type,
            padding=padding,
        )
        got = self.variant(conv)(x[None, ...])
        chex.assert_trees_all_close(got, expected[None, ...])
