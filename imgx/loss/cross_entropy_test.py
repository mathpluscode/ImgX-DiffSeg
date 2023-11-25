"""Test cross entropy loss functions."""

import chex
import jax
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.loss.cross_entropy import (
    cross_entropy,
    focal_loss,
    sigmoid_binary_cross_entropy,
    sigmoid_focal_loss,
    softmax_cross_entropy,
    softmax_focal_loss,
)


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestSoftmaxCrossEntropyLoss(chex.TestCase):
    """Test mean_softmax_cross_entropy, mean_cross_entropy."""

    prob_0 = 1 / (1 + np.exp(-1) + np.exp(-2))
    prob_1 = np.exp(-1) / (1 + np.exp(-1) + np.exp(-2))
    prob_2 = np.exp(-2) / (1 + np.exp(-1) + np.exp(-2))

    @chex.all_variants()
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
        """Test loss values.

        Args:
            logits: unscaled prediction, of shape (..., num_classes).
            targets: values are integers, of shape (...).
            expected: expected output.
        """
        num_classes = logits.shape[-1]
        # (batch, ..., num_classes)
        mask_true = jax.nn.one_hot(targets, num_classes=num_classes, axis=-1)
        got = self.variant(softmax_cross_entropy)(
            logits=logits,
            mask_true=mask_true,
        )
        chex.assert_trees_all_close(got, expected)

        got = self.variant(cross_entropy)(
            logits=logits,
            mask_true=mask_true,
            classes_are_exclusive=True,
        )
        chex.assert_trees_all_close(got, expected)


class TestSigmoidCrossEntropyLoss(chex.TestCase):
    """Test mean_sigmoid_binary_cross_entropy, mean_cross_entropy."""

    sigmoid_0 = 0.5
    sigmoid_1 = 1 / (1 + np.exp(-1))
    sigmoid_2 = 1 / (1 + np.exp(-2))
    sigmoid_n1 = 1 / (1 + np.exp(1))
    sigmoid_n2 = 1 / (1 + np.exp(2))

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d - one hot",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            np.mean(
                -np.log(
                    [
                        [
                            [sigmoid_n2, sigmoid_n1, sigmoid_0],
                            [sigmoid_0, sigmoid_n1, sigmoid_2],
                            [sigmoid_0, sigmoid_1, sigmoid_2],
                        ]
                    ],
                )
            ),
        ),
        (
            "1d - multi hot",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]]),
            np.mean(
                -np.log(
                    [
                        [
                            [sigmoid_2, sigmoid_n1, sigmoid_0],
                            [sigmoid_0, sigmoid_n1, sigmoid_n2],
                            [sigmoid_0, sigmoid_n1, sigmoid_2],
                        ]
                    ],
                )
            ),
        ),
        (
            "2d - one hot",
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
            np.mean(
                -np.log(
                    [
                        [
                            [
                                [sigmoid_n2, sigmoid_n1, sigmoid_0],
                                [sigmoid_0, sigmoid_n1, sigmoid_2],
                            ],
                            [
                                [sigmoid_1, sigmoid_n2, sigmoid_0],
                                [sigmoid_n1, sigmoid_n1, sigmoid_0],
                            ],
                        ],
                    ]
                )
            ),
        ),
        (
            "2d - multi hot",
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
                        [[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
                        [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
                    ],
                ]
            ),
            np.mean(
                -np.log(
                    [
                        [
                            [
                                [sigmoid_n2, sigmoid_1, sigmoid_0],
                                [sigmoid_0, sigmoid_n1, sigmoid_2],
                            ],
                            [
                                [sigmoid_1, sigmoid_2, sigmoid_0],
                                [sigmoid_n1, sigmoid_n1, sigmoid_0],
                            ],
                        ],
                    ]
                )
            ),
        ),
    )
    def test_values(
        self,
        logits: np.ndarray,
        mask_true: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        """Test loss values.

        Args:
            logits: unscaled prediction, of shape (..., num_classes).
            mask_true: binary masks, of shape (..., num_classes).
            expected: expected output.
        """
        got = self.variant(sigmoid_binary_cross_entropy)(
            logits=logits,
            mask_true=mask_true,
        )
        chex.assert_trees_all_close(got, expected)

        got = self.variant(cross_entropy)(
            logits=logits,
            mask_true=mask_true,
            classes_are_exclusive=False,
        )
        chex.assert_trees_all_close(got, expected)


class TestSoftmaxFocalLoss(chex.TestCase):
    """Test mean_softmax_focal_loss, mean_focal_loss."""

    prob_0 = 1 / (1 + np.exp(-1) + np.exp(-2))
    prob_1 = np.exp(-1) / (1 + np.exp(-1) + np.exp(-2))
    prob_2 = np.exp(-2) / (1 + np.exp(-1) + np.exp(-2))

    @chex.all_variants()
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
                    [-((1 - p) ** 1.2) * np.log(p) for p in [prob_2, prob_1, prob_0]],
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
                    [-((1 - p) ** 1.2) * np.log(p) for p in [prob_2, prob_1, prob_1, prob_2]],
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
        mask_true = jax.nn.one_hot(
            x=targets,
            num_classes=num_classes,
            axis=-1,
        )
        got = self.variant(softmax_focal_loss)(
            logits=logits,
            mask_true=mask_true,
            gamma=gamma,
        )
        chex.assert_trees_all_close(got, expected)

        got = self.variant(focal_loss)(
            logits=logits,
            mask_true=mask_true,
            gamma=gamma,
            classes_are_exclusive=True,
        )
        chex.assert_trees_all_close(got, expected)


def prob_to_sigmoid_focal_loss(prob: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Convert probability to sigmoid focal loss."""
    return -((1 - prob) ** gamma) * np.log(prob)


class TestSigmoidFocalLoss(chex.TestCase):
    """Test mean_sigmoid_focal_loss, mean_focal_loss."""

    sigmoid_0 = 0.5
    sigmoid_1 = 1 / (1 + np.exp(-1))
    sigmoid_2 = 1 / (1 + np.exp(-2))
    sigmoid_n1 = 1 / (1 + np.exp(1))
    sigmoid_n2 = 1 / (1 + np.exp(2))

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d - gamma=0.0 - one hot",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            0.0,
            np.mean(
                -np.log(
                    [
                        [
                            [sigmoid_n2, sigmoid_n1, sigmoid_0],
                            [sigmoid_0, sigmoid_n1, sigmoid_2],
                            [sigmoid_0, sigmoid_1, sigmoid_2],
                        ]
                    ],
                )
            ),
        ),
        (
            "1d - gamma=0.0 - multi hot",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]]),
            0.0,
            np.mean(
                -np.log(
                    [
                        [
                            [sigmoid_2, sigmoid_n1, sigmoid_0],
                            [sigmoid_0, sigmoid_n1, sigmoid_n2],
                            [sigmoid_0, sigmoid_n1, sigmoid_2],
                        ]
                    ],
                )
            ),
        ),
        (
            "2d - gamma=0.0 - one hot",
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
            0.0,
            np.mean(
                -np.log(
                    [
                        [
                            [
                                [sigmoid_n2, sigmoid_n1, sigmoid_0],
                                [sigmoid_0, sigmoid_n1, sigmoid_2],
                            ],
                            [
                                [sigmoid_1, sigmoid_n2, sigmoid_0],
                                [sigmoid_n1, sigmoid_n1, sigmoid_0],
                            ],
                        ],
                    ]
                )
            ),
        ),
        (
            "2d - gamma=0.0 - multi hot",
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
                        [[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
                        [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
                    ],
                ]
            ),
            0,
            np.mean(
                -np.log(
                    [
                        [
                            [
                                [sigmoid_n2, sigmoid_1, sigmoid_0],
                                [sigmoid_0, sigmoid_n1, sigmoid_2],
                            ],
                            [
                                [sigmoid_1, sigmoid_2, sigmoid_0],
                                [sigmoid_n1, sigmoid_n1, sigmoid_0],
                            ],
                        ],
                    ]
                )
            ),
        ),
        (
            "1d - gamma=1.2 - one hot",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            1.2,
            np.mean(
                prob_to_sigmoid_focal_loss(
                    np.array(
                        [
                            [
                                [sigmoid_n2, sigmoid_n1, sigmoid_0],
                                [sigmoid_0, sigmoid_n1, sigmoid_2],
                                [sigmoid_0, sigmoid_1, sigmoid_2],
                            ]
                        ],
                    )
                )
            ),
        ),
        (
            "2d - gamma=1.2 - one hot",
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
            1.2,
            np.mean(
                prob_to_sigmoid_focal_loss(
                    np.array(
                        [
                            [
                                [
                                    [sigmoid_n2, sigmoid_n1, sigmoid_0],
                                    [sigmoid_0, sigmoid_n1, sigmoid_2],
                                ],
                                [
                                    [sigmoid_1, sigmoid_n2, sigmoid_0],
                                    [sigmoid_n1, sigmoid_n1, sigmoid_0],
                                ],
                            ],
                        ]
                    )
                )
            ),
        ),
        (
            "2d - gamma=1.2 - multi hot",
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
                        [[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
                        [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
                    ],
                ]
            ),
            1.2,
            np.mean(
                prob_to_sigmoid_focal_loss(
                    np.array(
                        [
                            [
                                [
                                    [sigmoid_n2, sigmoid_1, sigmoid_0],
                                    [sigmoid_0, sigmoid_n1, sigmoid_2],
                                ],
                                [
                                    [sigmoid_1, sigmoid_2, sigmoid_0],
                                    [sigmoid_n1, sigmoid_n1, sigmoid_0],
                                ],
                            ],
                        ]
                    )
                )
            ),
        ),
    )
    def test_values(
        self,
        logits: np.ndarray,
        mask_true: np.ndarray,
        gamma: float,
        expected: np.ndarray,
    ) -> None:
        """Test dice loss values.

        Args:
            logits: unscaled prediction, of shape (..., num_classes).
            mask_true: binary masks, of shape (..., num_classes).
            gamma: adjust class imbalance, 0 is equivalent to cross entropy.
            expected: expected output.
        """
        got = self.variant(sigmoid_focal_loss)(
            logits=logits,
            mask_true=mask_true,
            gamma=gamma,
        )
        chex.assert_trees_all_close(got, expected)

        got = self.variant(focal_loss)(
            logits=logits,
            mask_true=mask_true,
            gamma=gamma,
            classes_are_exclusive=False,
        )
        chex.assert_trees_all_close(got, expected)
