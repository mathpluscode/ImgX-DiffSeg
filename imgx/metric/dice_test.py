"""Test dice score metric related functions."""

from functools import partial

import chex
import jax
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.metric import dice_score, iou, stability


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestDiceScore(chex.TestCase):
    """Test dice_score."""

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d",
            np.array([[[0.2, 0.3, 0.5], [0.0, 1.0, 0.0]]]),
            np.array([[2, 1]]),
            np.array(
                [
                    [
                        0.0,  # no target on background
                        2.0 / 2.3,
                        1.0 / 1.5,
                    ]
                ],
            ),
        ),
        (
            "2d",
            np.array(
                [
                    [
                        [[0.2, 0.3, 0.5], [0.0, 1.0, 0.0]],
                        [[0.9, 0.0, 0.1], [0.5, 0.1, 0.4]],
                    ],
                ]
            ),
            np.array([[[2, 1], [0, 1]]]),
            np.array(
                [
                    [
                        1.8 / 2.6,
                        2.2 / 3.4,
                        1.0 / 2.0,
                    ]
                ],
            ),
        ),
    )
    def test_values(
        self,
        mask_pred: np.ndarray,
        targets: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        """Test dice loss values.

        Args:
            mask_pred: unscaled prediction, of shape (..., num_classes).
            targets: values are integers, of shape (...).
            expected: expected output.
        """
        num_classes = mask_pred.shape[-1]
        # (batch, ..., num_classes)
        mask_true = jax.nn.one_hot(targets, num_classes=num_classes, axis=-1)
        got = self.variant(dice_score)(
            mask_pred=mask_pred,
            mask_true=mask_true,
        )
        chex.assert_trees_all_close(got, expected)


class TestIOU(chex.TestCase):
    """Test iou."""

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d",
            np.array([[[0.2, 0.3, 0.5], [0.0, 1.0, 0.0]]]),
            np.array([[2, 1]]),
            np.array(
                [
                    [
                        0.0,  # no target on background
                        1.0 / 1.3,
                        0.5 / 1.0,
                    ]
                ],
            ),
        ),
        (
            "2d",
            np.array(
                [
                    [
                        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    ],
                ]
            ),
            np.array([[[2, 1], [0, 1]]]),
            np.array(
                [
                    [
                        0.5,
                        0.5,
                        1.0,
                    ]
                ],
            ),
        ),
        (
            "2d-nan",
            np.array(
                [
                    [
                        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    ],
                ]
            ),
            np.array([[[0, 1], [0, 1]]]),
            np.array(
                [
                    [
                        1.0 / 3.0,
                        1.0 / 3.0,
                        np.nan,
                    ]
                ],
            ),
        ),
    )
    def test_values(
        self,
        mask_pred: np.ndarray,
        targets: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        """Test dice loss values.

        Args:
            mask_pred: soft mask, of shape (..., num_classes).
            targets: values are integers, of shape (...).
            expected: expected output.
        """
        num_classes = mask_pred.shape[-1]
        # (batch, ..., num_classes)
        mask_true = jax.nn.one_hot(targets, num_classes=num_classes, axis=-1)
        got = self.variant(iou)(
            mask_pred=mask_pred,
            mask_true=mask_true,
        )
        chex.assert_trees_all_close(got, expected)


class TestStability(chex.TestCase):
    """Test stability."""

    batch_size = 3

    @chex.all_variants()
    @parameterized.product(
        spatial_shape=[(4, 5), (4, 5, 6)],
        num_classes=[1, 4],
    )
    def test_shapes(
        self,
        spatial_shape: tuple[int, ...],
        num_classes: int,
    ) -> None:
        """Test dice loss values.

        Args:
            spatial_shape: spatial shape of the mask.
            num_classes: number of classes.
        """
        logits = jax.random.uniform(
            jax.random.PRNGKey(0),
            shape=(self.batch_size, *spatial_shape, num_classes),
        )
        got = self.variant(stability)(logits)
        chex.assert_shape(got, (self.batch_size, num_classes))

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "2d - 0 1",
            np.array(
                [
                    [
                        [[0.8, -1.8, 1.0], [2.3, -2.0, -0.3]],
                        [[-1.0, 1.5, -0.5], [1.1, -1.3, 0.2]],
                    ],
                ]
            ),
            0.0,
            1.0,
            np.array([[0.5, 1.0, 0.25]]),
        ),
        (
            "2d - 0 1 - unnormalized",
            np.array(
                [
                    [
                        [[1.8, -0.8, 2.0], [1.3, -3.0, -1.3]],
                        [[-1.0, 1.5, -0.5], [1.1, -1.3, 0.2]],
                    ],
                ]
            ),
            0.0,
            1.0,
            np.array([[0.5, 1.0, 0.25]]),
        ),
        (
            "2d - 1 1",
            np.array(
                [
                    [
                        [[0.8, -1.8, 1.0], [2.3, -2.0, -0.3]],
                        [[-1.0, 1.5, -0.5], [1.1, -1.3, 0.2]],
                    ],
                ]
            ),
            1.0,
            1.0,
            np.array([[1.0 / 3.0, 0.0, 0.0]]),
        ),
        (
            "2d - 2 0",
            np.array(
                [
                    [
                        [[0.8, -1.8, 1.0], [2.3, -2.0, -0.3]],
                        [[-1.0, 1.5, -0.5], [1.1, -1.3, 0.2]],
                    ],
                ]
            ),
            2.0,
            0.0,
            np.array([[1.0, np.nan, np.nan]]),
        ),
    )
    def test_values(
        self,
        logits: np.ndarray,
        threshold: float,
        threshold_offset: float,
        expected: np.ndarray,
    ) -> None:
        """Test dice loss values.

        Args:
            logits: shape (batch, ..., num_classes).
            threshold: threshold for prediction.
            threshold_offset: offset for threshold.
            expected: expected output.
        """
        got = self.variant(
            partial(
                stability,
                threshold=threshold,
                threshold_offset=threshold_offset,
            )
        )(logits)
        chex.assert_trees_all_close(got, expected)
