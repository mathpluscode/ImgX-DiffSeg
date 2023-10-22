"""Test dice loss functions."""

import chex
import jax
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.loss import dice_loss


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestSoftmaxDiceLoss(chex.TestCase):
    """Test dice_loss with mutual exclusive labels."""

    prob_0 = 1 / (1 + np.exp(-1) + np.exp(-2))
    prob_1 = np.exp(-1) / (1 + np.exp(-1) + np.exp(-2))
    prob_2 = np.exp(-2) / (1 + np.exp(-1) + np.exp(-2))

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d-with-background",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[2, 1, 0]]),
            np.array(
                [
                    1 - 2 * prob_0 / (3 * prob_0 + 1),
                    1 - 2 * prob_1 / (3 * prob_1 + 1),
                    1 - 2 * prob_2 / (3 * prob_2 + 1),
                ]
            ),
        ),
        (
            "2d-with-background",
            np.array(
                [
                    [
                        [[2.0, 1.0, 0.0], [0.0, -1.0, -2.0]],
                        [[1.0, 2.0, 0.0], [1.0, -1.0, 0.0]],
                    ],
                ]
            ),
            np.array([[[2, 1], [0, 1]]]),
            np.array(
                [
                    1 - 2 * prob_1 / (3 * prob_0 + prob_1 + 1),
                    1 - 2 * (prob_1 + prob_2) / (prob_0 + 2 * prob_1 + prob_2 + 2),
                    1 - 2 * prob_2 / (prob_1 + 3 * prob_2 + 1),
                ]
            ),
        ),
        (
            "2d-with-empty-class-and-background",
            np.array(
                [
                    [
                        [[2.0, 1.0, 0.0], [0.0, -1.0, -2.0]],
                        [[1.0, 2.0, 0.0], [1.0, -1.0, 0.0]],
                    ],
                ]
            ),
            np.array([[[0, 1], [0, 1]]]),
            np.array(
                [
                    1 - 2 * (prob_0 + prob_1) / (3 * prob_0 + prob_1 + 2),
                    1 - 2 * (prob_1 + prob_2) / (prob_0 + 2 * prob_1 + prob_2 + 2),
                    np.nan,
                ]
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
        # (batch, num_classes)
        got = self.variant(dice_loss)(
            logits=logits,
            mask_true=mask_true,
            classes_are_exclusive=True,
        )
        chex.assert_trees_all_close(got, expected[None, :])


class TestSigmoidDiceLoss(chex.TestCase):
    """Test mean_dice_loss with multi-hot labels."""

    sigmoid_0 = 0.5
    sigmoid_1 = 1 / (1 + np.exp(-1))
    sigmoid_2 = 1 / (1 + np.exp(-2))
    sigmoid_n1 = 1 / (1 + np.exp(1))
    sigmoid_n2 = 1 / (1 + np.exp(2))

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d - one hot - with background",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            np.array(
                [
                    1 - 2 * sigmoid_0 / (2 * sigmoid_0 + sigmoid_2 + 1),
                    1 - 2 * sigmoid_n1 / (2 * sigmoid_n1 + sigmoid_1 + 1),
                    1 - 2 * sigmoid_0 / (2 * sigmoid_n2 + sigmoid_0 + 1),
                ]
            ),
        ),
        (
            "1d - multi hot - with background",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]]),
            np.array(
                [
                    1 - 2 * (sigmoid_0 + sigmoid_2) / (2 * sigmoid_0 + sigmoid_2 + 2),
                    1 - 4 * sigmoid_n1 / (2 * sigmoid_n1 + sigmoid_1 + 2),
                    1 - 2 * (sigmoid_0 + sigmoid_n2) / (2 * sigmoid_n2 + sigmoid_0 + 2),
                ]
            ),
        ),
        (
            "2d - one hot - with background",
            np.array(
                [
                    [
                        [[2.0, 1.0, 0.0], [0.0, -1.0, -2.0]],
                        [[1.0, 2.0, 0.0], [1.0, -1.0, 0.0]],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    ],
                ]
            ),
            np.array(
                [
                    1 - 2 * sigmoid_1 / (2 * sigmoid_1 + sigmoid_2 + sigmoid_0 + 1),
                    1 - 4 * sigmoid_n1 / (2 * sigmoid_n1 + sigmoid_1 + sigmoid_2 + 2),
                    1 - 2 * sigmoid_0 / (sigmoid_n2 + 3 * sigmoid_0 + 1),
                ]
            ),
        ),
        (
            "2d - multi hot - with background",
            np.array(
                [
                    [
                        [[2.0, 1.0, 0.0], [0.0, -1.0, -2.0]],
                        [[1.0, 2.0, 0.0], [1.0, -1.0, 0.0]],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                        [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
                    ],
                ]
            ),
            np.array(
                [
                    1 - 2 * sigmoid_1 / (2 * sigmoid_1 + sigmoid_2 + sigmoid_0 + 1),
                    1
                    - 2
                    * (2 * sigmoid_n1 + sigmoid_2)
                    / (2 * sigmoid_n1 + sigmoid_1 + sigmoid_2 + 3),
                    1 - 4 * sigmoid_0 / (sigmoid_n2 + 3 * sigmoid_0 + 2),
                ]
            ),
        ),
    )
    def test_values(
        self,
        logits: np.ndarray,
        mask_true: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        """Test dice loss values.

        Args:
            logits: unscaled prediction, of shape (..., num_classes).
            mask_true: binary masks, of shape (..., num_classes).
            expected: expected output.
        """
        got = self.variant(dice_loss)(
            logits=logits,
            mask_true=mask_true,
            classes_are_exclusive=False,
        )
        chex.assert_trees_all_close(got, expected[None, :])
