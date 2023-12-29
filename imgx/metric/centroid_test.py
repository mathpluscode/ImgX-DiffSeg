"""Test centroid distance functions."""

from __future__ import annotations

import chex
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.data.warp import get_coordinate_grid
from imgx.metric import centroid_distance
from imgx.metric.centroid import get_centroid


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestCentroid(chex.TestCase):
    """Test get_centroid."""

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d-1class",
            1,
            (4,),
        ),
        (
            "1d-2classes",
            2,
            (4,),
        ),
        (
            "2d-1class",
            1,
            (4, 5),
        ),
        (
            "2d-2classes",
            2,
            (4, 3),
        ),
        (
            "3d-3classes",
            3,
            (4, 3, 5),
        ),
    )
    def test_shapes(
        self,
        num_classes: int,
        shape: tuple[int, ...],
    ) -> None:
        """Test exact values.

        Args:
            num_classes: number of classes.
            shape: shape of the grid, (d1, ..., dn).
        """
        batch = 2
        got_centroid, got_nan_mask = self.variant(get_centroid)(
            mask=np.ones((batch, *shape, num_classes)),
            grid=get_coordinate_grid(shape),
        )
        chex.assert_shape(got_centroid, (batch, len(shape), num_classes))
        chex.assert_shape(got_nan_mask, (batch, num_classes))

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d-1class",
            np.asarray([False, True, True, False])[..., None],
            (4,),
            np.asarray([1.5])[..., None],
        ),
        (
            "1d-1class-empty",
            np.asarray([False, False, False, False])[..., None],
            (4,),
            np.asarray([np.nan])[..., None],
        ),
        (
            "1d-2classes",
            np.asarray([[False, True], [True, True], [True, False], [False, False]]),
            (4,),
            np.asarray([[1.5, 0.5]]),
        ),
        (
            "1d-3classes",
            np.asarray(
                [
                    [False, True, True],
                    [True, True, True],
                    [True, False, False],
                    [False, False, False],
                ]
            ),
            (4,),
            np.asarray([[1.5, 0.5, 0.5]]),
        ),
        (
            "1d-3classes-with-nan",
            np.asarray(
                [
                    [False, True, False],
                    [True, True, False],
                    [True, False, False],
                    [False, False, False],
                ]
            ),
            (4,),
            np.asarray([[1.5, 0.5, np.nan]]),
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
            (4, 5),
            np.asarray([2.0 / 3.0, 2.0])[..., None],
        ),
        (
            "2d-nan",
            np.array(
                [
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            )[..., None],
            (4, 5),
            np.asarray([np.nan, np.nan])[..., None],
        ),
    )
    def test_values(self, mask: np.ndarray, shape: tuple[int, ...], expected: np.ndarray) -> None:
        """Test exact values.

        Args:
            mask: boolean mask on the image, (d1, ..., dn, num_classes).
            shape: shape of the grid, (d1, ..., dn).
            expected: expected coordinates.
        """
        got_centroid, got_nan_mask = self.variant(get_centroid)(
            mask=mask[None, ...],
            grid=get_coordinate_grid(shape),
        )
        chex.assert_trees_all_close(got_centroid[0, ...], expected)
        expected_nan_mask = np.sum(np.isnan(expected), axis=0) > 0
        chex.assert_trees_all_close(got_nan_mask[0, ...], expected_nan_mask)


class TestCentroidDistance(chex.TestCase):
    """Test centroid_distance."""

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d-same-1class",
            np.asarray([False, True, True, False])[..., None],
            np.asarray([False, True, True, False])[..., None],
            None,
            np.asarray([0.0]),
        ),
        (
            "1d-same-1class-nan",
            np.asarray([False, False, False, False])[..., None],
            np.asarray([False, True, True, False])[..., None],
            None,
            np.asarray([np.nan]),
        ),
        (
            "1d-diff-1class",
            np.asarray([False, True, True, False])[..., None],
            np.asarray([False, True, False, False])[..., None],
            None,
            np.asarray([0.5]),
        ),
        (
            "1d-diff-2classes",
            np.asarray([[False, True], [True, True], [True, False], [False, False]]),
            np.asarray([[False, True], [True, False], [True, False], [False, False]]),
            None,
            np.asarray([0, 0.5]),
        ),
        (
            "1d-diff-2classes-heterogeneous",
            np.asarray([[False, True], [True, True], [True, False], [False, False]]),
            np.asarray([[False, True], [True, False], [True, False], [False, False]]),
            np.asarray(
                [
                    2,
                ]
            ),
            np.asarray([0, 1.0]),
        ),
        (
            "1d-diff-2classes-heterogeneous-nan",
            np.asarray([[False, True], [True, True], [True, False], [False, False]]),
            np.asarray([[False, True], [False, False], [False, False], [False, False]]),
            np.asarray(
                [
                    2,
                ]
            ),
            np.asarray([np.nan, 1.0]),
        ),
        (
            "2d-same-1class",
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            )[..., None],
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            )[..., None],
            None,
            np.asarray([0.0]),
        ),
        (
            "2d-diff-1class",
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            )[..., None],
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, False],
                    [False, False, False, False, False],
                ]
            )[..., None],
            None,
            # dist between (5/3, 3) and (2, 2.5)
            np.asarray([np.sqrt(1.0 / 9.0 + 0.25)]),
        ),
        (
            "2d-diff-1class-heterogeneous",
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            )[..., None],
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, False],
                    [False, False, False, False, False],
                ]
            )[..., None],
            np.asarray([1.0, 2.0]),
            # dist between (5/3, 3) and (2, 2.5)
            np.asarray([np.sqrt(1.0 / 9.0 + 1.0)]),
        ),
    )
    def test_values(
        self,
        mask_true: np.ndarray,
        mask_pred: np.ndarray,
        spacing: np.ndarray | None,
        expected: float,
    ) -> None:
        """Test exact values.

        Args:
            mask_true: shape = (batch, d1, ..., dn, num_classes)
            mask_pred: shape = (batch, d1, ..., dn, num_classes)
            spacing: spacing of pixel/voxels along each dimension, (n,).
            expected: expected distance.
        """
        got = self.variant(centroid_distance)(
            mask_true=mask_true[None, ...],
            mask_pred=mask_pred[None, ...],
            grid=get_coordinate_grid(mask_true.shape[:-1]),
            spacing=spacing,
        )
        chex.assert_trees_all_close(got[0], expected)
