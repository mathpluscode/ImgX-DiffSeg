"""Test surface distance functions."""

from __future__ import annotations

from functools import partial
from typing import Callable

import chex
import jax
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.metric.surface_distance import (
    OneArgScalarFunc,
    TwoArgsScalarFunc,
    aggregated_surface_distance,
    average_surface_distance,
    get_mask_edges,
    get_surface_distance,
    hausdorff_distance,
    normalized_surface_dice,
    normalized_surface_dice_from_distances,
)


def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


def create_spherical_seg_3d(
    radius: float,
    centre: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Return a binary 3D image with a sphere inside.

    Voxel values will be 1 inside the sphere, and 0 elsewhere.

    https://github.com/Project-MONAI/MONAI/blob/dev/tests/test_surface_distance.py

    Args:
        radius: radius of sphere (in terms of number of voxels, can be partial)
        centre: location of sphere centre.
        shape: shape of image to create.
    """
    image = np.zeros(shape, dtype=np.int32)
    spy, spx, spz = np.ogrid[
        -centre[0] : shape[0] - centre[0],
        -centre[1] : shape[1] - centre[1],
        -centre[2] : shape[2] - centre[2],
    ]
    sphere = (spx * spx + spy * spy + spz * spz) <= radius * radius
    image[sphere] = 1
    image[~sphere] = 0
    return image


def create_circle_seg_2d(
    radius: float,
    centre: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    """Return a binary 2D image with a sphere inside.

    Pixel values will be 1 inside the circle, and 0 elsewhere

    Args:
        radius: radius of sphere (in terms of number of pixels, can be partial)
        centre: location of sphere centre.
        shape: shape of image to create.
    """
    image = np.zeros(shape, dtype=np.int32)
    spy, spx = np.ogrid[
        -centre[0] : shape[0] - centre[0],
        -centre[1] : shape[1] - centre[1],
    ]
    circle = (spx * spx + spy * spy) <= radius * radius
    image[circle] = 1
    image[~circle] = 0
    return image


class TestMaskEdge(chex.TestCase):
    """Test get_mask_edges."""

    @parameterized.named_parameters(
        (
            "2d-same-smaller",
            create_circle_seg_2d(radius=2, centre=(4, 4), shape=(7, 7)),
            create_circle_seg_2d(radius=2, centre=(4, 4), shape=(7, 7)),
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, True],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                ]
            ),
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, True],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                ]
            ),
        ),
        (
            "2d-diff-smaller",
            create_circle_seg_2d(radius=2, centre=(4, 4), shape=(7, 7)),
            create_circle_seg_2d(radius=1, centre=(4, 4), shape=(7, 7)),
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, True],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                ]
            ),
            np.array(
                [
                    [False, False, False, False, False],
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                ]
            ),
        ),
        (
            "2d-shift",
            create_circle_seg_2d(radius=1, centre=(4, 4), shape=(7, 7)),
            create_circle_seg_2d(radius=1, centre=(3, 4), shape=(7, 7)),
            np.array(
                [
                    [False, False, False],
                    [False, True, False],
                    [True, False, True],
                    [False, True, False],
                ]
            ),
            np.array(
                [
                    [False, True, False],
                    [True, False, True],
                    [False, True, False],
                    [False, False, False],
                ]
            ),
        ),
        (
            "2d-zero",
            np.zeros((5, 5)),
            np.zeros((5, 5)),
            np.zeros((5, 5), dtype=np.bool_),
            np.zeros((5, 5), dtype=np.bool_),
        ),
    )
    def test_values(
        self,
        mask_pred: np.ndarray,
        mask_true: np.ndarray,
        expected_edge_pred: np.ndarray,
        expected_edge_true: np.ndarray,
    ) -> None:
        """Test return values.

        Args:
            mask_pred: the predicted binary mask.
            mask_true: the ground truth binary mask.
            expected_edge_pred: the predicted binary edge.
            expected_edge_true: the ground truth binary edge.
        """
        got_edge_pred, got_edge_true = get_mask_edges(
            mask_pred=mask_pred,
            mask_true=mask_true,
        )
        chex.assert_trees_all_close(got_edge_pred, expected_edge_pred)
        chex.assert_trees_all_close(got_edge_true, expected_edge_true)


class TestSurfaceDistance(chex.TestCase):
    """Test surface_distance related functions."""

    @parameterized.product(
        ndims=[2, 3],
        func=[
            partial(average_surface_distance, spacing=None),
            partial(hausdorff_distance, percentile=100, spacing=None),
        ],
    )
    def test_nan_distance(
        self,
        ndims: int,
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        """Test average_surface_distance returns nan given empty inputs.

        Args:
            ndims: numbder of spatial dimentions.
            func: function to test.
        """
        batch = 2
        num_classes = 4
        # build dummy input having non-zero edges
        shape = (batch,) + (num_classes,) * ndims
        mask_true = np.zeros(shape)
        for i in range(num_classes):
            mask_true[:, i, ...] = i
        mask_true = np.array(
            jax.nn.one_hot(
                x=mask_true,
                num_classes=num_classes,
                axis=-1,
            )
        )
        mask_pred = np.zeros_like(mask_true)
        got = func(
            mask_pred=mask_pred,
            mask_true=mask_true,
        )
        assert np.isnan(got).all()

        got = func(
            mask_pred=mask_true,
            mask_true=mask_pred,
        )
        assert np.isnan(got).all()

        got = func(
            mask_pred=mask_pred,
            mask_true=mask_pred,
        )
        assert np.isnan(got).all()

    @parameterized.product(
        ndims=[2, 3],
        func=[
            partial(average_surface_distance, spacing=None),
            partial(hausdorff_distance, percentile=100, spacing=None),
        ],
    )
    def test_zero_distance(
        self,
        ndims: int,
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        """Test average_surface_distance returns zero given same inputs.

        Args:
            ndims: numbder of spatial dimentions.
            func: function to test.
        """
        batch = 2
        num_classes = 4
        # build dummy input having non-zero edges
        shape = (batch,) + (num_classes,) * ndims
        mask_true = np.zeros(shape)
        for i in range(num_classes):
            mask_true[:, i, ...] = i
        mask_true = np.array(
            jax.nn.one_hot(
                x=mask_true,
                num_classes=num_classes,
                axis=-1,
            )
        )
        got = func(mask_pred=mask_true, mask_true=mask_true)
        expected = np.zeros((batch, num_classes))
        assert np.array_equal(got, expected)

    @parameterized.named_parameters(
        (
            "2d-4x3",
            np.array(
                [
                    [False, False, False],
                    [False, True, False],
                    [True, False, True],
                    [False, True, False],
                ]
            ),
            np.array(
                [
                    [False, True, False],
                    [True, False, True],
                    [False, True, False],
                    [False, False, False],
                ]
            ),
            (1.0, 1.0),
            np.array([1.0, 1.0, 1.0, 1.0]),
        ),
        (
            "2d-5x5",
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, True],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                ]
            ),
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            ),
            (1.0, 1.0),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    np.sqrt(2),
                    np.sqrt(2),
                    np.sqrt(2),
                    np.sqrt(2),
                    2.0,
                ]
            ),
        ),
        (
            "2d-5x5-heterogeneous-1",
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, True],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                ]
            ),
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            ),
            (1.0, 2.0),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    np.sqrt(5),
                    np.sqrt(5),
                    2.0,  # via x axis it's shorter
                    2.0,  # via x axis it's shorter
                    2.0,
                ]
            ),
        ),
        (
            "2d-5x5-heterogeneous-2",
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            ),
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, True],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                ]
            ),
            (1.0, 2.0),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    2.0,  # via x axis it's shorter
                ]
            ),
        ),
        (
            "2d-6x5",
            np.array(
                [
                    [False, True, True, True, False],
                    [True, False, False, False, True],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            ),
            np.array(
                [
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                ]
            ),
            (1.0, 1.0),
            np.array(
                [
                    np.sqrt(10),
                    3.0,
                    np.sqrt(10),
                    np.sqrt(8),
                    np.sqrt(8),
                    np.sqrt(2),
                    np.sqrt(2),
                    0.0,
                ]
            ),
        ),
    )
    def test_surface_distance(
        self,
        edge_pred: np.ndarray,
        edge_true: np.ndarray,
        spacing: tuple[float, ...],
        expected: np.ndarray,
    ) -> None:
        """Test get_surface_distance with accurate expected values.

        Args:
            edge_pred: the predicted binary edge.
            edge_true: the ground truth binary edge.
            spacing: spacing of pixel/voxels along each dimension.
            expected: surface distance, 1D array of len = edge size.
        """
        got = get_surface_distance(
            edge_pred=edge_pred,
            edge_true=edge_true,
            spacing=spacing,
        )
        assert np.array_equal(got, expected)

    @parameterized.named_parameters(
        (
            # distances from pred to true
            # 0.0, 0.0, 0.0, np.sqrt(5), np.sqrt(5), 2.0, 2.0, 2.0
            # distances from true to pred
            # 0.0, 0.0, 0.0, 2.0
            "2d-5x5-heterogeneous-asymmetric-1",
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, True],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                ]
            ),
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            ),
            (1.0, 2.0),
            False,
            [
                partial(np.percentile, q=0),
                partial(np.percentile, q=100),
            ],
            [1, 1],
            np.array(
                [
                    0.0,
                    np.sqrt(5),
                ]
            ),
        ),
        (
            # distances from pred to true
            # 0.0, 0.0, 0.0, np.sqrt(5), np.sqrt(5), 2.0, 2.0, 2.0
            # distances from true to pred
            # 0.0, 0.0, 0.0, 2.0
            "2d-5x5-heterogeneous-asymmetric-2",
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            ),
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, True],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                ]
            ),
            (1.0, 2.0),
            False,
            [
                partial(np.percentile, q=0),
                partial(np.percentile, q=100),
            ],
            [1, 1],
            np.array(
                [
                    0.0,
                    2.0,
                ]
            ),
        ),
        (
            # distances from pred to true
            # 0.0, 0.0, 0.0, np.sqrt(5), np.sqrt(5), 2.0, 2.0, 2.0
            # distances from true to pred
            # 0.0, 0.0, 0.0, 2.0
            "2d-5x5-heterogeneous-symmetric",
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            ),
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, True],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                ]
            ),
            (1.0, 2.0),
            True,
            [
                partial(np.percentile, q=0),
                partial(np.percentile, q=100),
                normalized_surface_dice_from_distances,
            ],
            [1, 1, 2],
            np.array(
                [
                    0.0,
                    np.sqrt(5.0),
                    0.5,
                ]
            ),
        ),
    )
    def test_agg_surface_distance(
        self,
        mask_pred: np.ndarray,
        mask_true: np.ndarray,
        spacing: tuple[float, ...],
        symmetric: bool,
        agg_funcs: OneArgScalarFunc
        | TwoArgsScalarFunc
        | list[OneArgScalarFunc | TwoArgsScalarFunc],
        num_args: int | list[int],
        expected: np.ndarray,
    ) -> None:
        """Test get_surface_distance with accurate expected values.

        Args:
            mask_pred: predictions, without batch and class axes.
            mask_true: targets, without batch and class axes.
            spacing: spacing of pixel/voxels along each dimension.
            symmetric: the distance is symmetric to (pred, true) means swapping
                the masks provides the same value.
            agg_funcs: a function or a list of functions
                to aggregate a list of distances.
            num_args: a int or a list of ints, corresponding to number of
                arguments for agg_fn_list.
            expected: surface distance, 1D array of len = edge size.
        """
        got = aggregated_surface_distance(
            mask_pred=mask_pred[None, ..., None],
            mask_true=mask_true[None, ..., None],
            agg_fns=agg_funcs,
            num_args=num_args,
            spacing=spacing,
            symmetric=symmetric,
        )
        assert np.array_equal(got[:, 0, 0], expected)

    @parameterized.named_parameters(
        (
            "monai_3d_example1",
            create_spherical_seg_3d(radius=33, centre=(19, 33, 22), shape=(99, 99, 99)),
            create_spherical_seg_3d(radius=33, centre=(20, 33, 22), shape=(99, 99, 99)),
            (1.0, 1.0, 1.0),
            False,
            0.3483278807706289,
        ),
        (
            "monai_3d_example2",
            create_spherical_seg_3d(radius=20, centre=(20, 33, 22), shape=(99, 99, 99)),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22), shape=(99, 99, 99)),
            (1.0, 1.0, 1.0),
            False,
            12.040033513150455,
        ),
    )
    def test_average_surface_distance(
        self,
        mask_pred: np.ndarray,
        mask_true: np.ndarray,
        spacing: tuple[float, ...],
        symmetric: bool,
        expected: float,
    ) -> None:
        """Test average_surface_distance with accurate expected values.

        https://github.com/Project-MONAI/MONAI/blob/dev/tests/test_surface_distance.py

        Args:
            mask_pred: predictions, without batch and class axes.
            mask_true: targets, without batch and class axes.
            spacing: spacing of pixel/voxels along each dimension.
            symmetric: the distance is symmetric to (pred, true) means swapping
                the masks provides the same value.
            expected: expected value.
        """
        got = average_surface_distance(
            mask_pred=mask_pred[None, ..., None],
            mask_true=mask_true[None, ..., None],
            spacing=spacing,
            symmetric=symmetric,
        )
        assert np.isclose(np.mean(got), expected)

    @parameterized.named_parameters(
        (
            "monai_3d_example1",
            create_spherical_seg_3d(radius=20, centre=(20, 20, 20), shape=(99, 99, 99)),
            create_spherical_seg_3d(radius=20, centre=(10, 20, 20), shape=(99, 99, 99)),
            (1.0, 1.0, 1.0),
            False,
            10,
        ),
        (
            "2d_same_center_diff_radii",
            create_circle_seg_2d(radius=10, centre=(50, 50), shape=(99, 99)),
            create_circle_seg_2d(radius=20, centre=(50, 50), shape=(99, 99)),
            (1.0, 1.0),
            False,
            10,
        ),
        (
            "2d_diff_centers_same_radius",
            create_circle_seg_2d(radius=20, centre=(50, 51), shape=(99, 99)),
            create_circle_seg_2d(radius=20, centre=(50, 50), shape=(99, 99)),
            (1.0, 1.0),
            False,
            1,
        ),
        (
            "2d_diff_centers_diff_radii_asymmetric1",
            create_circle_seg_2d(radius=5, centre=(60, 50), shape=(99, 99)),
            create_circle_seg_2d(radius=10, centre=(50, 50), shape=(99, 99)),
            (1.0, 1.0),
            False,
            5,
        ),
        (
            "2d_diff_centers_diff_radii_asymmetric2",
            create_circle_seg_2d(radius=10, centre=(50, 50), shape=(99, 99)),
            create_circle_seg_2d(radius=5, centre=(60, 50), shape=(99, 99)),
            (1.0, 1.0),
            False,
            15,
        ),
        (
            "2d_diff_centers_diff_radii_symmetric1",
            create_circle_seg_2d(radius=10, centre=(50, 50), shape=(99, 99)),
            create_circle_seg_2d(radius=5, centre=(60, 50), shape=(99, 99)),
            (1.0, 1.0),
            True,
            15,
        ),
        (
            "2d_diff_centers_diff_radii_symmetric2",
            create_circle_seg_2d(radius=5, centre=(60, 50), shape=(99, 99)),
            create_circle_seg_2d(radius=10, centre=(50, 50), shape=(99, 99)),
            (1.0, 1.0),
            True,
            15,
        ),
    )
    def test_max_hausdorff_distance(
        self,
        mask_pred: np.ndarray,
        mask_true: np.ndarray,
        spacing: tuple[float, ...],
        symmetric: bool,
        expected: float,
    ) -> None:
        """Test hausdorff_distance with 100 percentile.

        Some test cases come from
        https://github.com/Project-MONAI/MONAI/blob/dev/tests/test_hausdorff_distance.py

        Args:
            mask_pred: predictions, without batch and class axes.
            mask_true: targets, without batch and class axes.
            spacing: spacing of pixel/voxels along each dimension.
            symmetric: the distance is symmetric to (pred, true) means swapping
                the masks provides the same value.
            expected: expected value.
        """
        got = hausdorff_distance(
            mask_pred=mask_pred[None, ..., None],
            mask_true=mask_true[None, ..., None],
            percentile=100,
            spacing=spacing,
            symmetric=symmetric,
        )
        assert np.isclose(np.mean(got), expected)

    @parameterized.named_parameters(
        (
            # distances from pred to true
            # 0.0, 0.0, 0.0, np.sqrt(5), np.sqrt(5), 2.0, 2.0, 2.0
            # distances from true to pred
            # 0.0, 0.0, 0.0, 2.0
            "2d-5x5-heterogeneous-1mm",
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            ),
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, True],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                ]
            ),
            (1.0, 2.0),
            1.0,
            np.array(
                [
                    0.5,
                ]
            ),
        ),
        (
            # distances from pred to true
            # 0.0, 0.0, 0.0, np.sqrt(5), np.sqrt(5), 2.0, 2.0, 2.0
            # distances from true to pred
            # 0.0, 0.0, 0.0, 2.0
            "2d-5x5-heterogeneous-1.5mm",
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            ),
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, True],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                ]
            ),
            (1.0, 2.0),
            1.5,
            np.array(
                [
                    0.5,
                ]
            ),
        ),
        (
            # distances from pred to true
            # 0.0, 0.0, 0.0, np.sqrt(5), np.sqrt(5), 2.0, 2.0, 2.0
            # distances from true to pred
            # 0.0, 0.0, 0.0, 2.0
            "2d-5x5-heterogeneous-2mm",
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            ),
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, True],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                ]
            ),
            (1.0, 2.0),
            2.0,
            np.array(
                [
                    10.0 / 12.0,
                ]
            ),
        ),
        (
            # distances from pred to true
            # 0.0, 0.0, 0.0, np.sqrt(5), np.sqrt(5), 2.0, 2.0, 2.0
            # distances from true to pred
            # 0.0, 0.0, 0.0, 2.0
            "2d-5x5-heterogeneous-2.24mm",
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            ),
            np.array(
                [
                    [False, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, True],
                    [False, True, False, True, False],
                    [False, False, True, False, False],
                ]
            ),
            (1.0, 2.0),
            2.24,  # sqrt(5) = 2.236...
            np.array(
                [
                    1.0,
                ]
            ),
        ),
    )
    def test_normalized_surface_dice(
        self,
        mask_pred: np.ndarray,
        mask_true: np.ndarray,
        spacing: tuple[float, ...],
        tolerance_mm: float,
        expected: float,
    ) -> None:
        """Test average_surface_distance with accurate expected values.

        https://github.com/Project-MONAI/MONAI/blob/dev/tests/test_surface_distance.py

        Args:
            mask_pred: predictions, without batch and class axes.
            mask_true: targets, without batch and class axes.
            spacing: spacing of pixel/voxels along each dimension.
            tolerance_mm: tolerance value to consider surface being overlapping.
            expected: expected value.
        """
        got = normalized_surface_dice(
            mask_pred=mask_pred[None, ..., None],
            mask_true=mask_true[None, ..., None],
            spacing=spacing,
            tolerance_mm=tolerance_mm,
        )
        assert np.isclose(np.mean(got), expected)
