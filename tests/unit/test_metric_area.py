"""Test area functions."""

import chex
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.metric.area import class_proportion


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestClassProportion(chex.TestCase):
    """Test get_coordinate_grid."""

    @chex.all_variants
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
            np.asarray(
                [[False, True], [True, True], [True, False], [False, False]]
            ),
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
