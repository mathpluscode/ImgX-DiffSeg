"""Tests for image preprocessing."""
from __future__ import annotations

import chex
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx_datasets.preprocess import get_binary_mask_bounding_box


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestBBox(chex.TestCase):
    """Test get_binary_mask_bounding_box."""

    @parameterized.named_parameters(
        (
            "1d-int",
            np.array([0, 1, 0, 1, 0]),
            np.array([1]),
            np.array([4]),
        ),
        (
            "1d-bool",
            np.array([False, True, False, True, False]),
            np.array([1]),
            np.array([4]),
        ),
        (
            "1d-all-true",
            np.array([True, True, True, True, True]),
            np.array([0]),
            np.array([5]),
        ),
        (
            "1d-all-false",
            np.array([False, False, False, False, False]),
            np.array([-1]),
            np.array([-1]),
        ),
        (
            "2d-1x5",
            np.array([[0, 1, 0, 1, 0]]),
            np.array([0, 1]),
            np.array([1, 4]),
        ),
        (
            "2d-2x5",
            np.array([[0, 1, 0, 1, 0], [1, 1, 0, 1, 0]]),
            np.array([0, 0]),
            np.array([2, 4]),
        ),
        (
            "2d-2x5-all-false",
            np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
            np.array([-1, -1]),
            np.array([-1, -1]),
        ),
    )
    def test_values(
        self,
        mask: np.ndarray,
        expected_bbox_min: np.ndarray,
        expected_bbox_max: np.ndarray,
    ) -> None:
        """Test dice loss values.

        Args:
            mask: binary mask with only spatial axes.
            expected_bbox_min: expected bounding box min, inclusive.
            expected_bbox_max: expected bounding box max, exclusive.
        """
        got_bbox_min, got_bbox_max = get_binary_mask_bounding_box(
            mask=mask,
        )
        chex.assert_trees_all_close(got_bbox_min, expected_bbox_min)
        chex.assert_trees_all_close(got_bbox_max, expected_bbox_max)
