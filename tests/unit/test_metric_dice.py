"""Test dice score metric related functions."""

import chex
import jax
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.metric import dice_score, iou


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestDiceScore(chex.TestCase):
    """Test dice_score."""

    @chex.all_variants
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

    @chex.all_variants
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
