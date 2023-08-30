"""Test function for data augmentation."""


import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.data.augmentation import (
    batch_apply_affine_to_grid,
    batch_get_random_affine_matrix,
    batch_random_affine_transform,
    batch_resample_image_label,
    get_affine_matrix,
    get_rotation_matrix,
    get_scaling_matrix,
    get_translation_matrix,
)
from imgx.metric.centroid import get_coordinate_grid
from imgx_datasets.constant import FOREGROUND_RANGE, IMAGE, LABEL


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
            np.asarray([np.pi / 6]),
            np.asarray(
                [
                    [cos_30, -sin_30, 0.0],
                    [sin_30, cos_30, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.asarray(
                [
                    [sqrt2 * np.cos(75 / 180 * np.pi)],
                    [sqrt2 * np.sin(75 / 180 * np.pi)],
                    [1.0],
                ]
            ),
        ),
        (
            "3d - no rotation",
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.asarray([[1.0], [1.0], [1.0], [1.0]]),
        ),
        (
            "3d - x axis - 30 degrees rotation",
            np.asarray([np.pi / 6, 0.0, 0.0]),
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, cos_30, -sin_30, 0.0],
                    [0.0, sin_30, cos_30, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.asarray(
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
            np.asarray([0.0, np.pi / 6, 0.0]),
            np.asarray(
                [
                    [cos_30, 0.0, sin_30, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-sin_30, 0.0, cos_30, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.asarray(
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
            np.asarray([0.0, 0.0, np.pi / 6]),
            np.asarray(
                [
                    [cos_30, -sin_30, 0.0, 0.0],
                    [sin_30, cos_30, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.asarray(
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
        radians: np.ndarray,
        expected_affine_matrix: np.ndarray,
        expected_rotated_vector: np.ndarray,
    ) -> None:
        """Test affine matrix values and rotated unit vector.

        Args:
            radians: values correspond to yz, xz, xy planes.
            expected_affine_matrix: expected affine matrix.
            expected_rotated_vector: expected rotated vector.
        """
        if len(radians) > 1:
            vector = jnp.array([[1.0], [1.0], [1.0], [1.0]])
        else:
            vector = jnp.array([[1.0], [1.0], [1.0]])
        got_affine_matrix = self.variant(get_rotation_matrix)(
            radians=radians,
        )
        chex.assert_trees_all_close(got_affine_matrix, expected_affine_matrix)

        got_rotated_vector = jnp.matmul(got_affine_matrix, vector)
        chex.assert_trees_all_close(got_rotated_vector, expected_rotated_vector)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "2d - shift",
            np.asarray([-1.0, -2.0]),
            np.asarray(
                [
                    [1.0, 0.0, -1.0],
                    [0.0, 1.0, -2.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.asarray([[0.0], [-1.0], [1.0]]),
        ),
        (
            "3d - no shift",
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.asarray([[1.0], [1.0], [1.0], [1.0]]),
        ),
        (
            "3d - shift x axis",
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.asarray(
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
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.asarray(
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
            np.asarray([0.0, 0.0, 1.0]),
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.asarray(
                [
                    [1.0],
                    [1.0],
                    [2.0],
                    [1.0],
                ]
            ),
        ),
    )
    def test_translation(
        self,
        shifts: np.ndarray,
        expected_affine_matrix: np.ndarray,
        expected_rotated_vector: np.ndarray,
    ) -> None:
        """Test affine matrix values and rotated unit vector.

        Args:
            shifts: correspond to each axis shift.
            expected_affine_matrix: expected affine matrix.
            expected_rotated_vector: expected rotated vector.
        """
        vector = jnp.ones(shape=(len(shifts) + 1, 1))
        got_affine_matrix = self.variant(get_translation_matrix)(
            shifts=shifts,
        )
        chex.assert_trees_all_close(got_affine_matrix, expected_affine_matrix)

        got_rotated_vector = jnp.matmul(got_affine_matrix, vector)
        chex.assert_trees_all_close(got_rotated_vector, expected_rotated_vector)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "2d - scale",
            np.asarray([2.0, 3.0]),
            np.asarray(
                [
                    [2.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.asarray([[2.0], [3.0], [1.0]]),
        ),
        (
            "3d - no scale",
            np.asarray([1.0, 1.0, 1.0]),
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.asarray([[1.0], [1.0], [1.0], [1.0]]),
        ),
        (
            "3d - scale x axis",
            np.asarray([2.0, 1.0, 1.0]),
            np.asarray(
                [
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.asarray(
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
            np.asarray([1.0, 2.0, 1.0]),
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.asarray(
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
            np.asarray([1.0, 1.0, 2.0]),
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.asarray(
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
        expected_rotated_vector: np.ndarray,
    ) -> None:
        """Test affine matrix values and rotated unit vector.

        Args:
            scales: correspond to each axis scaling.
            expected_affine_matrix: expected affine matrix.
            expected_rotated_vector: expected rotated vector.
        """
        vector = jnp.ones(shape=(len(scales) + 1, 1))
        got_affine_matrix = self.variant(get_scaling_matrix)(
            scales=scales,
        )
        chex.assert_trees_all_close(got_affine_matrix, expected_affine_matrix)

        got_rotated_vector = jnp.matmul(got_affine_matrix, vector)
        chex.assert_trees_all_close(got_rotated_vector, expected_rotated_vector)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "2d - rotate - scale - shift",
            np.asarray([np.pi / 2]),
            np.asarray([-1.0, 1.0]),
            np.asarray([0.8, 1.2]),
            np.asarray(
                [
                    [0.0, -0.8, -1.0],
                    [1.2, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
            ),
            np.asarray([[-1.8], [2.2]]),
        ),
    )
    def test_affine(
        self,
        radians: np.ndarray,
        shifts: np.ndarray,
        scales: np.ndarray,
        expected_affine_matrix: np.ndarray,
        expected_rotated_vector: np.ndarray,
    ) -> None:
        """Test affine matrix values and rotated unit vector.

        Args:
            radians: correspond to rotate around each axis.
            shifts: correspond to each axis shift.
            scales: correspond to each axis scaling.
            expected_affine_matrix: expected affine matrix.
            expected_rotated_vector: expected rotated vector.
        """
        vector = jnp.ones(shape=(len(scales) + 1, 1))
        got_affine_matrix = self.variant(get_affine_matrix)(
            radians=radians,
            shifts=shifts,
            scales=scales,
        )
        chex.assert_trees_all_close(
            got_affine_matrix, expected_affine_matrix, atol=1e-6
        )

        got_rotated_vector = jnp.matmul(got_affine_matrix[:-1, :], vector)
        chex.assert_trees_all_close(got_rotated_vector, expected_rotated_vector)


class TestRandomAffineMatrix(chex.TestCase):
    """Test random affine matrix sampling."""

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "2d - batch size 1",
            1,
            np.asarray(
                [
                    0.088,
                ]
            ),
            np.asarray([20, 4]),
            np.asarray([0.15, 0.15]),
            (3, 3),
        ),
        (
            "2d - batch size 2",
            2,
            np.asarray(
                [
                    0.088,
                ]
            ),
            np.asarray([20, 4]),
            np.asarray([0.15, 0.15]),
            (3, 3),
        ),
        (
            "3d - batch size 2",
            2,
            np.asarray([0.088, 0.088, 0.088]),
            np.asarray([20, 20, 4]),
            np.asarray([0.15, 0.15, 0.15]),
            (4, 4),
        ),
    )
    def test_values(
        self,
        batch_size: int,
        max_rotation: np.ndarray,
        max_translation: np.ndarray,
        max_scaling: np.ndarray,
        expected_shape: tuple,
    ) -> None:
        """Test affine matrix values.

        Test affine matrix shapes, and test random seed impact.

        Args:
            batch_size: number of samples in batch.
            max_rotation: maximum rotation in radians.
            max_translation: maximum translation in pixel/voxels.
            max_scaling: maximum scaling difference in pixel/voxels.
            expected_shape: expected shape of affine matrix.
        """
        max_rotation = np.tile(max_rotation[None, ...], (batch_size, 1))
        max_translation = np.tile(max_translation[None, ...], (batch_size, 1))
        max_scaling = np.tile(max_scaling[None, ...], (batch_size, 1))

        # check output shape
        key1 = jax.random.PRNGKey(1)
        got1 = self.variant(batch_get_random_affine_matrix)(
            key=key1,
            max_rotation=max_rotation,
            min_translation=-max_translation,
            max_translation=max_translation,
            max_scaling=max_scaling,
        )
        chex.assert_shape(got1, (batch_size, *expected_shape))

        # if batch size > 1, each affine matrix should be different
        if batch_size > 1:
            diff = jnp.sum(jnp.abs(got1[1, ...] - got1[0, ...])).item()
            chex.assert_scalar_positive(diff)

        # same seed should provide same values
        got2 = self.variant(batch_get_random_affine_matrix)(
            key=key1,
            max_rotation=max_rotation,
            min_translation=-max_translation,
            max_translation=max_translation,
            max_scaling=max_scaling,
        )
        chex.assert_trees_all_equal(got1, got2)

        # different seeds should provide different values
        key3 = jax.random.PRNGKey(3)
        got3 = self.variant(batch_get_random_affine_matrix)(
            key=key3,
            max_rotation=max_rotation,
            min_translation=-max_translation,
            max_translation=max_translation,
            max_scaling=max_scaling,
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
        grid = np.asarray(
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
        affine_matrix = np.asarray(
            [
                [2.0, 1.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        expected = np.asarray(
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
        batch_affine_matrix = np.tile(
            affine_matrix[None, ...], (batch_size, 1, 1)
        )
        batch_expected = np.tile(
            expected[None, ...], (batch_size,) + (1,) * len(expected.shape)
        )
        got = self.variant(batch_apply_affine_to_grid)(
            grid=grid,
            affine_matrix=batch_affine_matrix,
        )
        chex.assert_trees_all_equal(got, batch_expected)


class TestResample(chex.TestCase):
    """Test apply_affine_to_grid."""

    @chex.all_variants()
    @parameterized.product(
        (
            {
                "image": np.asarray(
                    [
                        [
                            [2.0, 1.0, 0.0],
                            [0.0, 3.0, 4.0],
                        ],
                        [
                            [2.0, 1.0, 0.0],
                            [0.0, 3.0, 4.0],
                        ],
                    ],
                ),
                "label": np.asarray(
                    [
                        [
                            [2.0, 1.0, 0.0],
                            [0.0, 3.0, 4.0],
                        ],
                        [
                            [2.0, 1.0, 0.0],
                            [0.0, 3.0, 4.0],
                        ],
                    ],
                ),
                "grid": np.asarray(
                    [
                        # first image, un changed
                        [
                            # x axis
                            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                            # y axis
                            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                        ],
                        # second image, changed
                        # (0.4, 0) x-axis linear interpolation
                        # (0, 0.6) y-axis linear interpolation
                        # (0.4, 1.6) x/y-axis linear interpolation
                        # (1.0, 3.0) out of boundary
                        [
                            # x axis
                            [[0.4, 0.0, 0.4], [1.0, 1.0, 1.0]],
                            # y axis
                            [[0.0, 0.6, 1.6], [0.0, 3.0, 2.0]],
                        ],
                    ]
                ),  # (batch=2, n=2, d1=2, d2=3)
                "expected_image": np.asarray(
                    [
                        [
                            [2.0, 1.0, 0.0],
                            [0.0, 3.0, 4.0],
                        ],
                        [
                            [1.2, 1.4, 1.68],
                            [0.0, 0.0, 4.0],
                        ],
                    ],
                ),
                "expected_label": np.asarray(
                    [
                        [
                            [2.0, 1.0, 0.0],
                            [0.0, 3.0, 4.0],
                        ],
                        [
                            [2.0, 1.0, 0.0],
                            [0.0, 0.0, 4.0],
                        ],
                    ],
                ),
            },
        ),
        num_channels=[0, 1, 2],
    )
    def test_shapes(
        self,
        image: np.ndarray,
        label: np.ndarray,
        grid: np.ndarray,
        expected_image: np.ndarray,
        expected_label: np.ndarray,
        num_channels: int,
    ) -> None:
        """Test affine matrix values.

        Test affine matrix shapes, and test random seed impact.

        Args:
            image: input image batch.
            label: input label batch.
            grid: batch of grid with affine applied.
            expected_image: expected image.
            expected_label: expected label.
            num_channels: number of channels to add to image.
        """
        if num_channels == 1:
            image = image[..., None]
            expected_image = expected_image[..., None]
        elif num_channels > 1:
            reps = (1,) * (len(image.shape) - 1) + (num_channels,)
            image = np.tile(image[..., None], reps)
            expected_image = np.tile(expected_image[..., None], reps)

        input_dict = {IMAGE: image, LABEL: label}
        got = self.variant(batch_resample_image_label)(
            input_dict=input_dict,
            grid=grid,
        )
        expected = {IMAGE: expected_image, LABEL: expected_label}
        chex.assert_trees_all_close(got, expected)


class TestRandomAffineTransformation(chex.TestCase):
    """Test batch_random_affine_transform."""

    @chex.all_variants()
    @parameterized.product(
        (
            {
                "max_rotation": np.asarray([0.088, 0.088, 0.088]),
                "max_translation": np.asarray([2, 3, 1]),
                "max_scaling": np.asarray([0.05, 0.05, 0.05]),
                "image_shape": (8, 12, 6),
            },
            {
                "max_rotation": np.asarray([0.088]),
                "max_translation": np.asarray([2, 3]),
                "max_scaling": np.asarray([0.05, 0.05]),
                "image_shape": (8, 12),
            },
        ),
        batch_size=[4, 1],
    )
    def test_shapes(
        self,
        batch_size: int,
        max_rotation: np.ndarray,
        max_translation: np.ndarray,
        max_scaling: np.ndarray,
        image_shape: tuple,
    ) -> None:
        """Test affine matrix values.

        Test affine matrix shapes, and test random seed impact.

        Args:
            batch_size: number of samples in batch.
            max_rotation: maximum rotation in radians.
            max_translation: maximum translation in pixel/voxels.
            max_scaling: maximum scaling difference in pixel/voxels.
            image_shape: image spatial shape.
        """
        key = jax.random.PRNGKey(0)
        grid = get_coordinate_grid(shape=image_shape)
        image = jax.random.uniform(
            key=key, shape=(batch_size, *image_shape), minval=0, maxval=1
        )
        label = jax.random.uniform(
            key=key, shape=(batch_size, *image_shape), minval=0, maxval=1
        )
        label = jnp.asarray(label > jnp.mean(label), dtype=np.float32)
        input_dict = {
            IMAGE: image,
            LABEL: label,
            FOREGROUND_RANGE: jnp.zeros((len(image_shape), 2)),
        }
        got = self.variant(batch_random_affine_transform)(
            key=key,
            input_dict=input_dict,
            grid=grid,
            max_rotation=max_rotation,
            max_translation=max_translation,
            max_scaling=max_scaling,
        )

        # check shapes
        assert len(got) == 2
        chex.assert_shape(got[IMAGE], (batch_size, *image_shape))
        chex.assert_shape(got[LABEL], (batch_size, *image_shape))

        # check label remains boolean
        assert jnp.unique(got[LABEL]).size == jnp.unique(label).size
