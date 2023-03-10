"""Test dice loss functions."""

import chex
import jax
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.loss import mean_cross_entropy, mean_focal_loss


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestCrossEntropyLoss(chex.TestCase):
    """Test mean_cross_entropy."""

    prob_0 = 1 / (1 + np.exp(-1) + np.exp(-2))
    prob_1 = np.exp(-1) / (1 + np.exp(-1) + np.exp(-2))
    prob_2 = np.exp(-2) / (1 + np.exp(-1) + np.exp(-2))

    @chex.all_variants
    @parameterized.named_parameters(
        (
            "1d",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[2, 1, 0]]),
            np.mean(
                -np.log(
                    [prob_2, prob_1, prob_0],
                )
            ),
        ),
        (
            "2d",
            np.array(
                [
                    [
                        [[2.0, 1.0, 0.0], [0.0, -1.0, -2.0]],
                        [[1.0, 2.0, 0.0], [1.0, -1.0, 0.0]],
                    ],
                ]
            ),
            np.array([[[2, 1], [0, 1]]]),
            np.mean(
                -np.log(
                    [prob_2, prob_1, prob_1, prob_2],
                )
            ),
        ),
    )
    def test_values(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        """Test dice loss values.

        Args:
            logits: unscaled prediction, of shape (..., num_classes).
            targets: values are integers, of shape (...).
            expected: expected output.
        """
        num_classes = logits.shape[-1]
        # (batch, ..., num_classes)
        mask_true = jax.nn.one_hot(targets, num_classes=num_classes, axis=-1)
        got = self.variant(mean_cross_entropy)(
            logits=logits,
            mask_true=mask_true,
        )
        chex.assert_trees_all_close(got, expected)


class TestFocalLoss(chex.TestCase):
    """Test mean_focal_loss."""

    prob_0 = 1 / (1 + np.exp(-1) + np.exp(-2))
    prob_1 = np.exp(-1) / (1 + np.exp(-1) + np.exp(-2))
    prob_2 = np.exp(-2) / (1 + np.exp(-1) + np.exp(-2))

    @chex.all_variants
    @parameterized.named_parameters(
        (
            "1d-gamma=0.0",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[2, 1, 0]]),
            0.0,
            np.mean(
                -np.log(
                    [prob_2, prob_1, prob_0],
                )
            ),
        ),
        (
            "2d-gamma=0.0",
            np.array(
                [
                    [
                        [[2.0, 1.0, 0.0], [0.0, -1.0, -2.0]],
                        [[1.0, 2.0, 0.0], [1.0, -1.0, 0.0]],
                    ],
                ]
            ),
            np.array([[[2, 1], [0, 1]]]),
            0.0,
            np.mean(
                -np.log(
                    [prob_2, prob_1, prob_1, prob_2],
                )
            ),
        ),
        (
            "1d-gamma=1.2",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[2, 1, 0]]),
            1.2,
            np.mean(
                np.array(
                    [
                        -((1 - p) ** 1.2) * np.log(p)
                        for p in [prob_2, prob_1, prob_0]
                    ],
                )
            ),
        ),
        (
            "2d-gamma=1.2",
            np.array(
                [
                    [
                        [[2.0, 1.0, 0.0], [0.0, -1.0, -2.0]],
                        [[1.0, 2.0, 0.0], [1.0, -1.0, 0.0]],
                    ],
                ]
            ),
            np.array([[[2, 1], [0, 1]]]),
            1.2,
            np.mean(
                np.array(
                    [
                        -((1 - p) ** 1.2) * np.log(p)
                        for p in [prob_2, prob_1, prob_1, prob_2]
                    ],
                )
            ),
        ),
    )
    def test_values(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        gamma: float,
        expected: np.ndarray,
    ) -> None:
        """Test dice loss values.

        Args:
            logits: unscaled prediction, of shape (..., num_classes).
            targets: values are integers, of shape (...).
            gamma: adjust class imbalance, 0 is equivalent to cross entropy.
            expected: expected output.
        """
        num_classes = logits.shape[-1]
        # (batch, ..., num_classes)
        mask_true = jax.nn.one_hot(targets, num_classes=num_classes, axis=-1)
        got = self.variant(mean_focal_loss)(
            logits=logits,
            mask_true=mask_true,
            gamma=gamma,
        )
        chex.assert_trees_all_close(got, expected)
