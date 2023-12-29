"""Test area functions."""

import chex
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.metric.area import class_proportion, class_volume, get_volume


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestClassProportion(chex.TestCase):
    """Test class_proportion."""

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d-1class",
            np.asarray([False, True, True, False])[..., None],
            np.asarray([0.5])[..., None],
        ),
        (
            "1d-1class-empty",
            np.asarray([False, False, False, False])[..., None],
            np.asarray([0.0])[..., None],
        ),
        (
            "1d-2classes",
            np.asarray([[False, True], [True, True], [True, False], [False, False]]),
            np.asarray([[0.5, 0.5]]),
        ),
        (
            "2d-1class",
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            )[..., None],
            np.asarray([3.0 / 20.0])[..., None],
        ),
    )
    def test_values(self, mask: np.ndarray, expected: np.ndarray) -> None:
        """Test exact values.

        Args:
            mask: shape = (batch, d1, ..., dn, num_classes).
            expected: expected coordinates.
        """
        got = self.variant(class_proportion)(
            mask=mask[None, ...],
        )
        chex.assert_trees_all_close(got, expected)


class TestGetVolume(chex.TestCase):
    """Test get_volume."""

    @chex.all_variants()
    @parameterized.product(
        image_shape=[(8, 12, 6), (8, 12)],
        batch_size=[4, 1],
    )
    def test_shapes(self, image_shape: tuple[int, ...], batch_size: int) -> None:
        """Test output shapes.

        Args:
            image_shape: spatial shape of images.
            batch_size: number of samples in batch.
        """
        mask = np.zeros((batch_size, *image_shape), dtype=np.bool_)
        spacing = np.ones(len(image_shape))
        got = self.variant(get_volume)(mask, spacing)
        chex.assert_shape(got, (batch_size,))


class TestClassVolume(chex.TestCase):
    """Test class_volume."""

    @chex.all_variants()
    @parameterized.product(
        image_shape=[(8, 12, 6), (8, 12)],
        num_classes=[1, 2, 3],
        batch_size=[4, 1],
    )
    def test_shapes(self, image_shape: tuple[int, ...], num_classes: int, batch_size: int) -> None:
        """Test output shapes.

        Args:
            image_shape: spatial shape of images.
            num_classes: number of classes.
            batch_size: number of samples in batch.
        """
        mask = np.zeros((batch_size, *image_shape, num_classes), dtype=np.bool_)
        spacing = np.ones(len(image_shape))
        got = self.variant(class_volume)(mask, spacing)
        chex.assert_shape(got, (batch_size, num_classes))
