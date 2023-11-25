"""Test warp and grid functions."""

from __future__ import annotations

from functools import partial

import chex
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.data.warp import batch_grid_sample, get_coordinate_grid


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestGrid(chex.TestCase):
    """Test get_coordinate_grid."""

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        (
            "1d",
            (2,),
            np.asarray([[0.0, 1.0]]),
        ),
        (
            "2d",
            (3, 2),
            np.asarray(
                [
                    [
                        [0.0, 0.0],
                        [1.0, 1.0],
                        [2.0, 2.0],
                    ],
                    [
                        [0.0, 1.0],
                        [0.0, 1.0],
                        [0.0, 1.0],
                    ],
                ],
            ),
        ),
    )
    def test_values(self, shape: tuple[int, ...], expected: np.ndarray) -> None:
        """Test exact values.

        Args:
            shape: shape of the grid, (d1, ..., dn).
            expected: expected coordinates.
        """
        got = self.variant(get_coordinate_grid)(
            shape=shape,
        )
        chex.assert_trees_all_equal(got, expected)


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

        got_image = self.variant(partial(batch_grid_sample, order=1))(
            x=image,
            grid=grid,
        )
        chex.assert_trees_all_close(got_image, expected_image)
        got_label = self.variant(partial(batch_grid_sample, order=0))(
            x=label,
            grid=grid,
        )
        chex.assert_trees_all_close(got_label, expected_label)
