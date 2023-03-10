"""Test dice loss functions."""

import chex
import jax
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.loss import mean_dice_loss


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestDiceLoss(chex.TestCase):
    """Test dice_loss."""

    prob_0 = 1 / (1 + np.exp(-1) + np.exp(-2))
    prob_1 = np.exp(-1) / (1 + np.exp(-1) + np.exp(-2))
    prob_2 = np.exp(-2) / (1 + np.exp(-1) + np.exp(-2))

    @chex.all_variants
    @parameterized.named_parameters(
        (
            "1d-with-background",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[2, 1, 0]]),
            True,
            np.mean(
                np.array(
                    [
                        [
                            1 - 2 * prob_0 / (3 * prob_0 + 1),
                            1 - 2 * prob_1 / (3 * prob_1 + 1),
                            1 - 2 * prob_2 / (3 * prob_2 + 1),
                        ]
                    ],
                )
            ),
        ),
        (
            "1d-without-background",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[2, 1, 0]]),
            False,
            np.mean(
                np.array(
                    [
                        [
                            1 - 2 * prob_1 / (3 * prob_1 + 1),
                            1 - 2 * prob_2 / (3 * prob_2 + 1),
                        ]
                    ],
                )
            ),
        ),
        (
            "1d-without-and-miss-background",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0]]]),
            np.array([[2, 1]]),
            False,
            np.mean(
                np.array(
                    [
                        [
                            1 - 2 * prob_1 / (2 * prob_1 + 1),
                            1 - 2 * prob_2 / (2 * prob_2 + 1),
                        ]
                    ],
                )
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
            True,
            np.mean(
                np.array(
                    [
                        [
                            1 - 2 * prob_1 / (3 * prob_0 + prob_1 + 1),
                            1
                            - 2
                            * (prob_1 + prob_2)
                            / (prob_0 + 2 * prob_1 + prob_2 + 2),
                            1 - 2 * prob_2 / (prob_1 + 3 * prob_2 + 1),
                        ]
                    ],
                )
            ),
        ),
        (
            "2d-without-background",
            np.array(
                [
                    [
                        [[2.0, 1.0, 0.0], [0.0, -1.0, -2.0]],
                        [[1.0, 2.0, 0.0], [1.0, -1.0, 0.0]],
                    ],
                ]
            ),
            np.array([[[2, 1], [0, 1]]]),
            False,
            np.mean(
                np.array(
                    [
                        [
                            1
                            - 2
                            * (prob_1 + prob_2)
                            / (prob_0 + 2 * prob_1 + prob_2 + 2),
                            1 - 2 * prob_2 / (prob_1 + 3 * prob_2 + 1),
                        ]
                    ],
                )
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
            True,
            np.mean(
                np.array(
                    [
                        [
                            1
                            - 2 * (prob_0 + prob_1) / (3 * prob_0 + prob_1 + 2),
                            1
                            - 2
                            * (prob_1 + prob_2)
                            / (prob_0 + 2 * prob_1 + prob_2 + 2),
                        ]
                    ],
                )
            ),
        ),
        (
            "2d-with-empty-class-without-background",
            np.array(
                [
                    [
                        [[2.0, 1.0, 0.0], [0.0, -1.0, -2.0]],
                        [[1.0, 2.0, 0.0], [1.0, -1.0, 0.0]],
                    ],
                ]
            ),
            np.array([[[0, 1], [0, 1]]]),
            False,
            np.array(
                1 - 2 * (prob_1 + prob_2) / (prob_0 + 2 * prob_1 + prob_2 + 2),
            ),
        ),
    )
    def test_values(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        include_background: bool,
        expected: np.ndarray,
    ) -> None:
        """Test dice loss values.

        Args:
            logits: unscaled prediction, of shape (..., num_classes).
            targets: values are integers, of shape (...).
            include_background: include background as a separate class.
            expected: expected output.
        """
        num_classes = logits.shape[-1]
        # (batch, ..., num_classes)
        mask_true = jax.nn.one_hot(targets, num_classes=num_classes, axis=-1)
        got = self.variant(mean_dice_loss)(
            logits=logits,
            mask_true=mask_true,
            include_background=include_background,
        )
        chex.assert_trees_all_close(got, expected)
