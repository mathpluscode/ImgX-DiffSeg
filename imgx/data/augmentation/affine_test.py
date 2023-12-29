"""Test function for affine data augmentation."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.data.augmentation.affine import (
    batch_apply_affine_to_grid,
    batch_get_random_affine_matrix,
    batch_random_affine_transform,
    get_affine_matrix,
    get_rotation_matrix,
    get_scaling_matrix,
    get_shear_matrix,
    get_translation_matrix,
)
from imgx.data.warp import get_coordinate_grid
from imgx.datasets.constant import FOREGROUND_RANGE, IMAGE, LABEL


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestDeterministicAffineMatrix(chex.TestCase):
    """Test deterministic affine matrix."""

    sin_30 = 0.5
    cos_30 = np.sqrt(3) / 2
    sqrt2 = np.sqrt(2)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "2d - 30 degrees rotation",
            np.array([np.pi / 6]),
            np.array(
                [
                    [cos_30, -sin_30, 0.0],
                    [sin_30, cos_30, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [sqrt2 * np.cos(75 / 180 * np.pi)],
                    [sqrt2 * np.sin(75 / 180 * np.pi)],
                    [1.0],
                ]
            ),
        ),
        (
            "3d - no rotation",
            np.array([0.0, 0.0, 0.0]),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array([[1.0], [1.0], [1.0], [1.0]]),
        ),
        (
            "3d - x axis - 30 degrees rotation",
            np.array([np.pi / 6, 0.0, 0.0]),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, cos_30, -sin_30, 0.0],
                    [0.0, sin_30, cos_30, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0],
                    [sqrt2 * np.cos(75 / 180 * np.pi)],
                    [sqrt2 * np.sin(75 / 180 * np.pi)],
                    [1.0],
                ]
            ),
        ),
        (
            "3d - y axis - 30 degrees rotation",
            np.array([0.0, np.pi / 6, 0.0]),
            np.array(
                [
                    [cos_30, 0.0, sin_30, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-sin_30, 0.0, cos_30, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [sqrt2 * np.sin(75 / 180 * np.pi)],
                    [1.0],
                    [sqrt2 * np.cos(75 / 180 * np.pi)],
                    [1.0],
                ]
            ),
        ),
        (
            "3d - z axis - 30 degrees rotation",
            np.array([0.0, 0.0, np.pi / 6]),
            np.array(
                [
                    [cos_30, -sin_30, 0.0, 0.0],
                    [sin_30, cos_30, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [sqrt2 * np.cos(75 / 180 * np.pi)],
                    [sqrt2 * np.sin(75 / 180 * np.pi)],
                    [1.0],
                    [1.0],
                ]
            ),
        ),
    )
    def test_rotation(
        self,
        rotations: np.ndarray,
        expected_affine_matrix: np.ndarray,
        expected_rotated_vector: np.ndarray,
    ) -> None:
        """Test affine matrix values and rotated unit vector.

        Args:
            rotations: values correspond to yz, xz, xy planes.
            expected_affine_matrix: expected affine matrix.
            expected_rotated_vector: expected rotated vector.
        """
        if len(rotations) > 1:
            vector = jnp.array([[1.0], [1.0], [1.0], [1.0]])
        else:
            vector = jnp.array([[1.0], [1.0], [1.0]])
        got_affine_matrix = self.variant(get_rotation_matrix)(
            rotations=rotations,
        )
        chex.assert_trees_all_close(got_affine_matrix, expected_affine_matrix)

        got_vector = jnp.matmul(got_affine_matrix, vector)
        chex.assert_trees_all_close(got_vector, expected_rotated_vector)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "2d - scale",
            np.array([2.0, 3.0]),
            np.array(
                [
                    [2.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.array([[2.0], [3.0], [1.0]]),
        ),
        (
            "3d - no scale",
            np.array([1.0, 1.0, 1.0]),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array([[1.0], [1.0], [1.0], [1.0]]),
        ),
        (
            "3d - scale x axis",
            np.array([2.0, 1.0, 1.0]),
            np.array(
                [
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [2.0],
                    [1.0],
                    [1.0],
                    [1.0],
                ]
            ),
        ),
        (
            "3d - scale y axis",
            np.array([1.0, 2.0, 1.0]),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0],
                    [2.0],
                    [1.0],
                    [1.0],
                ]
            ),
        ),
        (
            "3d - scale z axis",
            np.array([1.0, 1.0, 2.0]),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0],
                    [1.0],
                    [2.0],
                    [1.0],
                ]
            ),
        ),
    )
    def test_scaling(
        self,
        scales: np.ndarray,
        expected_affine_matrix: np.ndarray,
        expected_vector: np.ndarray,
    ) -> None:
        """Test affine matrix values and scale unit vector.

        Args:
            scales: correspond to each axis scaling.
            expected_affine_matrix: expected affine matrix.
            expected_vector: expected transformed vector.
        """
        vector = jnp.ones(shape=(len(scales) + 1, 1))
        got_affine_matrix = self.variant(get_scaling_matrix)(
            scales=scales,
        )
        chex.assert_trees_all_close(got_affine_matrix, expected_affine_matrix)

        got_vector = jnp.matmul(got_affine_matrix, vector)
        chex.assert_trees_all_close(got_vector, expected_vector)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "2d - shear x",
            np.array([np.pi / 3, 0.0]),
            np.array(
                [
                    [1.0, np.sqrt(3), 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.array([[1.0 + np.sqrt(3)], [1.0], [1.0]]),
        ),
        (
            "2d - shear y",
            np.array([0.0, -np.pi / 6]),
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0 / np.sqrt(3), 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.array([[1.0], [1.0 - 1.0 / np.sqrt(3)], [1.0]]),
        ),
        (
            "3d - shear xy",
            np.array([np.pi / 3, -np.pi / 6, 0.0, 0.0, 0.0, 0.0]),
            np.array(
                [
                    [1.0, 0.0, np.sqrt(3), 0.0],
                    [0.0, 1.0, -1.0 / np.sqrt(3), 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array([[1.0 + np.sqrt(3)], [1.0 - 1.0 / np.sqrt(3)], [1.0], [1.0]]),
        ),
        (
            "3d - shear xy xz",
            np.array([np.pi / 3, -np.pi / 6, np.pi / 6, -np.pi / 3, 0.0, 0.0]),
            np.array(
                [
                    [1.0, 1.0 / np.sqrt(3), np.sqrt(3) - 1.0 / 3, 0.0],
                    [0.0, 1.0, -1.0 / np.sqrt(3), 0.0],
                    [0.0, -np.sqrt(3), 2.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [[2.0 / 3.0 + 4 / np.sqrt(3)], [1.0 - 1.0 / np.sqrt(3)], [2.0 - np.sqrt(3)], [1.0]]
            ),
        ),
        (
            "3d - shear xy xz yz",
            np.array([np.pi / 3, -np.pi / 6, np.pi / 6, -np.pi / 3, -np.pi / 3, np.pi / 6]),
            np.array(
                [
                    [1.0, 1.0 / np.sqrt(3), np.sqrt(3) - 1.0 / 3, 0.0],
                    [-np.sqrt(3), 0.0, -3.0, 0.0],
                    [1.0 / np.sqrt(3), 1.0 / 3.0 - np.sqrt(3), 3.0 - 1.0 / 3.0 / np.sqrt(3), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [2.0 / 3.0 + 4 / np.sqrt(3)],
                    [-3.0 - np.sqrt(3)],
                    [10.0 / 3.0 - 7.0 / 9.0 * np.sqrt(3)],
                    [1.0],
                ]
            ),
        ),
    )
    def test_shear(
        self,
        shears: np.ndarray,
        expected_affine_matrix: np.ndarray,
        expected_shear_vector: np.ndarray,
    ) -> None:
        """Test affine matrix values and shift unit vector.

        Args:
            shears: correspond to each axis/plane shears.
            expected_affine_matrix: expected affine matrix.
            expected_shear_vector: expected vector.
        """
        if shears.size == 2:
            vector = jnp.ones(shape=(3, 1))
        elif shears.size == 6:
            vector = jnp.ones(shape=(4, 1))
        else:
            raise ValueError("shears must be 2 or 6 values")

        got_shear_matrix = self.variant(get_shear_matrix)(
            shears=shears,
        )
        chex.assert_trees_all_close(got_shear_matrix, expected_affine_matrix, atol=1e-6)

        got_shear_vector = jnp.matmul(got_shear_matrix, vector)
        chex.assert_trees_all_close(got_shear_vector, expected_shear_vector, atol=1e-6)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "2d - shift",
            np.array([-1.0, -2.0]),
            np.array(
                [
                    [1.0, 0.0, -1.0],
                    [0.0, 1.0, -2.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.array([[0.0], [-1.0], [1.0]]),
        ),
        (
            "3d - no shift",
            np.array([0.0, 0.0, 0.0]),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array([[1.0], [1.0], [1.0], [1.0]]),
        ),
        (
            "3d - shift x axis",
            np.array([1.0, 0.0, 0.0]),
            np.array(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [2.0],
                    [1.0],
                    [1.0],
                    [1.0],
                ]
            ),
        ),
        (
            "3d - shift y axis",
            np.array([0.0, 1.0, 0.0]),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0],
                    [2.0],
                    [1.0],
                    [1.0],
                ]
            ),
        ),
        (
            "3d - shift z axis",
            np.array([0.0, 0.0, 1.0]),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0],
                    [1.0],
                    [2.0],
                    [1.0],
                ]
            ),
        ),
    )
    def test_shift(
        self,
        shifts: np.ndarray,
        expected_affine_matrix: np.ndarray,
        expected_vector: np.ndarray,
    ) -> None:
        """Test affine matrix values and shift unit vector.

        Args:
            shifts: correspond to each axis shift.
            expected_affine_matrix: expected affine matrix.
            expected_vector: expected transformed vector.
        """
        vector = jnp.ones(shape=(len(shifts) + 1, 1))
        got_affine_matrix = self.variant(get_translation_matrix)(
            shifts=shifts,
        )
        chex.assert_trees_all_close(got_affine_matrix, expected_affine_matrix)

        got_vector = jnp.matmul(got_affine_matrix, vector)
        chex.assert_trees_all_close(got_vector, expected_vector)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "2d - rotate",
            np.array([1.0, 1.0]),
            np.array([np.pi / 2]),
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array(
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ),
            np.array([[-1.0], [1.0]]),
        ),
        (
            "2d - rotate - scale",
            np.array([1.0, 1.0]),
            np.array([np.pi / 2]),
            np.array([0.8, 1.2]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array(
                [
                    [0.0, -0.8, 0.0],
                    [1.2, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ),
            np.array([[-0.8], [1.2]]),
        ),
        (
            "2d - rotate - scale - shift",
            np.array([1.0, 1.0]),
            np.array([np.pi / 2]),
            np.array([0.8, 1.2]),
            np.array([0.0, 0.0]),
            np.array([-1.0, 1.0]),
            np.array(
                [
                    [0.0, -0.8, -1.0],
                    [1.2, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
            ),
            np.array([[-1.8], [2.2]]),
        ),
        (
            "2d - rotate - scale - shear",
            np.array([1.0, 1.0]),
            np.array([np.pi / 2]),
            np.array([0.8, 1.2]),
            np.array([np.pi / 3, -np.pi / 6]),
            np.array([0.0, 0.0]),
            np.array(
                [
                    [1.2 * np.sqrt(3), -0.8, 0.0],
                    [0.0, 0.8 / np.sqrt(3), 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ),
            np.array([[-0.8 + 1.2 * np.sqrt(3)], [0.8 / np.sqrt(3)]]),
        ),
        (
            "2d with spacing - rotate - scale - shear",
            np.array([1.5, 10.0]),
            np.array([np.pi / 2]),
            np.array([0.8, 1.2]),
            np.array([np.pi / 3, -np.pi / 6]),
            np.array([0.0, 0.0]),
            np.array(
                [
                    [1.2 * np.sqrt(3), -8 / 1.5, 0.0],
                    [0.0, 0.8 / np.sqrt(3), 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ),
            np.array([[-8 / 1.5 + 1.2 * np.sqrt(3)], [0.8 / np.sqrt(3)]]),
        ),
    )
    def test_affine(
        self,
        spacing: np.ndarray,
        rotations: np.ndarray,
        scales: np.ndarray,
        shears: np.ndarray,
        shifts: np.ndarray,
        expected_affine_matrix: np.ndarray,
        expected_vector: np.ndarray,
    ) -> None:
        """Test affine matrix values with multiple transformations.

        Args:
            spacing: spacing for each axis.
            rotations: correspond to rotate around each axis.
            scales: correspond to each axis scaling.
            shears: correspond to each axis/plane shear.
            shifts: correspond to each axis shift.
            expected_affine_matrix: expected affine matrix.
            expected_vector: expected transformed vector.
        """
        vector = jnp.ones(shape=(len(scales) + 1, 1))
        got_affine_matrix = self.variant(get_affine_matrix)(
            spacing=spacing,
            rotations=rotations,
            scales=scales,
            shears=shears,
            shifts=shifts,
        )
        chex.assert_trees_all_close(got_affine_matrix, expected_affine_matrix, atol=1e-6)

        got_vector = jnp.matmul(got_affine_matrix[:-1, :], vector)
        chex.assert_trees_all_close(got_vector, expected_vector)


class TestRandomAffineMatrix(chex.TestCase):
    """Test random affine matrix sampling."""

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "2d - batch size 1",
            1,
            np.array([1.0, 1.0]),
            np.array([0.088]),
            np.array([0.15, 0.15]),
            np.array([0.088, 0.088]),
            np.array([20, 4]),
            (3, 3),
        ),
        (
            "2d - batch size 2",
            2,
            np.array([1.0, 1.0]),
            np.array([0.088]),
            np.array([0.15, 0.15]),
            np.array([0.088, 0.088]),
            np.array([20, 4]),
            (3, 3),
        ),
        (
            "3d - batch size 2",
            2,
            np.array([1.0, 1.0, 1.2]),
            np.array([0.088, 0.088, 0.088]),
            np.array([0.15, 0.15, 0.15]),
            np.array([0.088, 0.088, 0.088, 0.088, 0.088, 0.088]),
            np.array([20, 20, 4]),
            (4, 4),
        ),
    )
    def test_randomness(
        self,
        batch_size: int,
        spacing: np.ndarray,
        max_rotation: np.ndarray,
        max_zoom: np.ndarray,
        max_shear: np.ndarray,
        max_shift: np.ndarray,
        expected_shape: tuple[int, ...],
    ) -> None:
        """Test affine matrix values.

        Test affine matrix shapes, and test random seed impact.

        Args:
            batch_size: number of samples in batch.
            spacing: spacing for each axis.
            max_rotation: maximum rotation in radians.
            max_zoom: maximum zoom in pixel/voxels.
            max_shear: maximum shear in radians.
            max_shift: maximum shift in pixel/voxels.
            expected_shape: expected shape of affine matrix.
        """
        max_rotation = np.tile(max_rotation[None, ...], (batch_size, 1))
        max_zoom = np.tile(max_zoom[None, ...], (batch_size, 1))
        max_shear = np.tile(max_shear[None, ...], (batch_size, 1))
        max_shift = np.tile(max_shift[None, ...], (batch_size, 1))

        # check output shape
        key1 = jax.random.PRNGKey(1)
        got1 = self.variant(batch_get_random_affine_matrix)(
            key=key1,
            spacing=spacing,
            max_rotation=max_rotation,
            max_zoom=max_zoom,
            max_shear=max_shear,
            max_shift=max_shift,
        )
        chex.assert_shape(got1, (batch_size, *expected_shape))

        # if batch size > 1, each affine matrix should be different
        if batch_size > 1:
            diff = jnp.sum(jnp.abs(got1[1, ...] - got1[0, ...])).item()
            chex.assert_scalar_positive(diff)

        # same seed should provide same values
        got2 = self.variant(batch_get_random_affine_matrix)(
            key=key1,
            spacing=spacing,
            max_rotation=max_rotation,
            max_zoom=max_zoom,
            max_shear=max_shear,
            max_shift=max_shift,
        )
        chex.assert_trees_all_equal(got1, got2)

        # different seeds should provide different values
        key3 = jax.random.PRNGKey(3)
        got3 = self.variant(batch_get_random_affine_matrix)(
            key=key3,
            spacing=spacing,
            max_rotation=max_rotation,
            max_zoom=max_zoom,
            max_shear=max_shear,
            max_shift=max_shift,
        )
        diff = jnp.sum(jnp.abs(got1 - got3)).item()
        chex.assert_scalar_positive(diff)


class TestApplyAffineMatrix(chex.TestCase):
    """Test apply_affine_to_grid."""

    @chex.all_variants()
    @parameterized.parameters(4, 1)
    def test_values(
        self,
        batch_size: int,
    ) -> None:
        """Test transformed grid values.

        Args:
            batch_size: number of samples in batch.
        """
        grid = np.array(
            [
                # x
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                ],
                # y
                [
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
            ],
        )
        affine_matrix = np.array(
            [
                [2.0, 1.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        expected = np.array(
            [
                # shift -> 2x+y -> shift back
                [
                    [-1.5, -0.5],
                    [0.5, 1.5],
                    [2.5, 3.5],
                ],
                # shift -> 3y -> shift back
                [
                    [-1.0, 2.0],
                    [-1.0, 2.0],
                    [-1.0, 2.0],
                ],
            ],
        )
        batch_affine_matrix = np.tile(affine_matrix[None, ...], (batch_size, 1, 1))
        batch_expected = np.tile(expected[None, ...], (batch_size,) + (1,) * len(expected.shape))
        got = self.variant(batch_apply_affine_to_grid)(
            grid=grid,
            affine_matrix=batch_affine_matrix,
        )
        chex.assert_trees_all_equal(got, batch_expected)


class TestRandomAffineTransformation(chex.TestCase):
    """Test batch_random_affine_transform."""

    @chex.all_variants()
    @parameterized.product(
        (
            {
                "spacing": np.array([1.0, 1.0, 1.0]),
                "max_rotation": np.array([0.088, 0.088, 0.088]),
                "max_zoom": np.array([0.05, 0.05, 0.05]),
                "max_shear": np.array([0.088, 0.088, 0.088, 0.088, 0.088, 0.088]),
                "max_shift": np.array([2, 3, 1]),
                "image_shape": (8, 12, 6),
            },
            {
                "spacing": np.array([1.2, 1.0]),
                "max_rotation": np.array([0.088]),
                "max_zoom": np.array([0.05, 0.05]),
                "max_shear": np.array([0.088, 0.088]),
                "max_shift": np.array([2, 3]),
                "image_shape": (8, 12),
            },
        ),
        batch_size=[4, 1],
    )
    def test_shapes(
        self,
        batch_size: int,
        spacing: np.ndarray,
        max_rotation: np.ndarray,
        max_zoom: np.ndarray,
        max_shear: np.ndarray,
        max_shift: np.ndarray,
        image_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes.

        Args:
            batch_size: number of samples in batch.
            spacing: spacing for each axis.
            max_rotation: maximum rotation in radians.
            max_shear: maximum shear in radians.
            max_zoom: maximum scaling difference in pixel/voxels.
            max_shift: maximum shift in pixel/voxels.
            image_shape: image spatial shape.
        """
        key = jax.random.PRNGKey(0)
        grid = get_coordinate_grid(shape=image_shape)
        image = jax.random.uniform(key=key, shape=(batch_size, *image_shape), minval=0, maxval=1)
        label = jax.random.uniform(key=key, shape=(batch_size, *image_shape), minval=0, maxval=1)
        label = jnp.array(label > jnp.mean(label), dtype=np.float32)
        batch = {
            IMAGE: image,
            LABEL: label,
            FOREGROUND_RANGE: jnp.zeros((len(image_shape), 2)),
        }
        got = self.variant(batch_random_affine_transform)(
            key=key,
            batch=batch,
            grid=grid,
            spacing=spacing,
            max_rotation=max_rotation,
            max_zoom=max_zoom,
            max_shear=max_shear,
            max_shift=max_shift,
        )

        # check shapes
        assert len(got) == 2
        chex.assert_shape(got[IMAGE], (batch_size, *image_shape))
        chex.assert_shape(got[LABEL], (batch_size, *image_shape))

        # check label remains boolean
        assert jnp.unique(got[LABEL]).size == jnp.unique(label).size
