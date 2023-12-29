"""Test cross entropy loss functions."""

import chex
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.loss.cross_entropy import (
    cross_entropy,
    focal_loss,
    sigmoid_focal_loss,
    softmax_focal_loss,
)


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestSoftmaxCrossEntropyLoss(chex.TestCase):
    """Test cross_entropy for exclusive classes."""

    prob_0 = 1 / (1 + np.exp(-1) + np.exp(-2))
    prob_1 = np.exp(-1) / (1 + np.exp(-1) + np.exp(-2))
    prob_2 = np.exp(-2) / (1 + np.exp(-1) + np.exp(-2))

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            np.mean(
                -np.log(
                    [prob_2, prob_1, prob_0],
                )
            ),
        ),
        (
            "1d - soft label 1",
            np.array([[[2.0, 1.0, 0.0], [0.0, -2.0, -1.0], [0.0, -1.0, -2.0]]]),
            np.array([[[0.0, 0.1, 0.9], [0.2, 0.8, 0.0], [0.7, 0.3, 0.0]]]),
            np.mean(
                -np.log(
                    [prob_2, prob_1, prob_0],
                )
                * np.array([1.7, 0.4, 0.9])
            ),
        ),
        (
            "1d - soft label 2",
            np.array([[[2.0, 1.0, 0.0], [0.0, -2.0, -1.0], [-1.0, -0.0, -2.0]]]),
            np.array([[[0.15, 0.1, 0.75], [0.2, 0.8, 0.0], [0.7, 0.3, 0.0]]]),
            np.mean(
                -np.log(
                    [prob_2, prob_1, prob_0],
                )
                * np.array([1.55, 0.8, 0.65])
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
            np.array([[[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]]),
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
        mask_true: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        """Test loss values.

        Args:
            logits: unscaled prediction, of shape (..., num_classes).
            mask_true: shape (..., num_classes), values at the last axis sums to one.
            expected: expected output.
        """
        got = self.variant(cross_entropy)(
            logits=logits,
            mask_true=mask_true,
            classes_are_exclusive=True,
        )
        chex.assert_trees_all_close(got, expected)


class TestSigmoidCrossEntropyLoss(chex.TestCase):
    """Test cross_entropy for non-exclusive classes."""

    log_sigmoid_0 = np.log(0.5)
    log_sigmoid_1 = np.log(1 / (1 + np.exp(-1)))
    log_sigmoid_2 = np.log(1 / (1 + np.exp(-2)))
    log_sigmoid_n1 = np.log(1 / (1 + np.exp(1)))
    log_sigmoid_n2 = np.log(1 / (1 + np.exp(2)))

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d - one hot",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            np.mean(
                -np.sum(
                    np.array(
                        [
                            [
                                [log_sigmoid_n2, log_sigmoid_n1, log_sigmoid_0],
                                [log_sigmoid_0, log_sigmoid_n1, log_sigmoid_2],
                                [log_sigmoid_0, log_sigmoid_1, log_sigmoid_2],
                            ]
                        ]
                    ),
                    axis=-1,
                )
            ),
        ),
        (
            "1d - soft label",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[0.1, 0.2, 1.0], [0.0, 0.9, 0.0], [1.0, 0.0, 0.0]]]),
            np.mean(
                -np.sum(
                    np.array(
                        [
                            [
                                0.1 * log_sigmoid_2 + 0.9 * log_sigmoid_n2,
                                0.2 * log_sigmoid_1 + 0.8 * log_sigmoid_n1,
                                log_sigmoid_0,
                            ],
                            [
                                log_sigmoid_0,
                                0.1 * log_sigmoid_1 + 0.9 * log_sigmoid_n1,
                                log_sigmoid_2,
                            ],
                            [log_sigmoid_0, log_sigmoid_1, log_sigmoid_2],
                        ]
                    ),
                    axis=-1,
                )
            ),
        ),
        (
            "1d - multi hot",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]]),
            np.mean(
                -np.sum(
                    np.array(
                        [
                            [
                                [log_sigmoid_2, log_sigmoid_n1, log_sigmoid_0],
                                [log_sigmoid_0, log_sigmoid_n1, log_sigmoid_n2],
                                [log_sigmoid_0, log_sigmoid_n1, log_sigmoid_2],
                            ]
                        ]
                    ),
                    axis=-1,
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
                -np.sum(
                    np.array(
                        [
                            [
                                [log_sigmoid_n2, log_sigmoid_n1, log_sigmoid_0],
                                [log_sigmoid_0, log_sigmoid_n1, log_sigmoid_2],
                            ],
                            [
                                [log_sigmoid_1, log_sigmoid_n2, log_sigmoid_0],
                                [log_sigmoid_n1, log_sigmoid_n1, log_sigmoid_0],
                            ],
                        ],
                    ),
                    axis=-1,
                ),
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
                -np.sum(
                    np.array(
                        [
                            [
                                [log_sigmoid_n2, log_sigmoid_1, log_sigmoid_0],
                                [log_sigmoid_0, log_sigmoid_n1, log_sigmoid_2],
                            ],
                            [
                                [log_sigmoid_1, log_sigmoid_2, log_sigmoid_0],
                                [log_sigmoid_n1, log_sigmoid_n1, log_sigmoid_0],
                            ],
                        ]
                    ),
                    axis=-1,
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
            mask_true: shape (..., num_classes), values at the last axis sums to one.
            expected: expected output.
        """
        got = self.variant(cross_entropy)(
            logits=logits,
            mask_true=mask_true,
            classes_are_exclusive=False,
        )
        chex.assert_trees_all_close(got, expected)


class TestSoftmaxFocalLoss(chex.TestCase):
    """Test focal_loss for exclusive classes."""

    prob_0 = 1 / (1 + np.exp(-1) + np.exp(-2))
    prob_1 = np.exp(-1) / (1 + np.exp(-1) + np.exp(-2))
    prob_2 = np.exp(-2) / (1 + np.exp(-1) + np.exp(-2))

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d-gamma=0.0",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
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
                    [prob_2, prob_1, prob_1, prob_2],
                )
            ),
        ),
        (
            "1d-gamma=2.0",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            2.0,
            np.mean(
                np.array(
                    [-((1 - p) ** 2.0) * np.log(p) for p in [prob_2, prob_1, prob_0]],
                )
            ),
        ),
        (
            "1d-gamma=2.0-soft label",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[0.1, 0.0, 0.8], [0.0, 0.7, 0.0], [0.6, 0.0, 0.0]]]),
            2.0,
            np.mean(
                -np.array(
                    [
                        (1 - prob_0) ** 2 * np.log(prob_0) * 0.1
                        + (1 - prob_2) ** 2 * np.log(prob_2) * 0.8,
                        (1 - prob_1) ** 2 * np.log(prob_1) * 0.7,
                        (1 - prob_0) ** 2 * np.log(prob_0) * 0.6,
                    ],
                )
            ),
        ),
        (
            "2d-gamma=2.0",
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
            2.0,
            np.mean(
                np.array(
                    [-((1 - p) ** 2.0) * np.log(p) for p in [prob_2, prob_1, prob_1, prob_2]],
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
        """Test loss values.

        Args:
            logits: unscaled prediction, of shape (..., num_classes).
            mask_true: masks, of shape (..., num_classes).
            gamma: adjust class imbalance, 0 is equivalent to cross entropy.
            expected: expected output.
        """
        got = self.variant(focal_loss)(
            logits=logits,
            mask_true=mask_true,
            gamma=gamma,
            classes_are_exclusive=True,
        )
        chex.assert_trees_all_close(got, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d",
            (3,),
        ),
        (
            "2d",
            (3, 4),
        ),
        (
            "3d",
            (3, 4, 5),
        ),
    )
    def test_shapes(
        self,
        spatial_shape: tuple[int, ...],
    ) -> None:
        """Test loss shapes.

        Args:
            spatial_shape: spatial shape.
        """
        batch = 2
        num_classes = 3
        logits = np.ones((batch, *spatial_shape, num_classes), dtype=np.float32)
        mask_true = np.ones((batch, *spatial_shape, num_classes), dtype=np.float32)
        got = self.variant(softmax_focal_loss)(
            logits=logits,
            mask_true=mask_true,
        )
        chex.assert_shape(got, (batch, *spatial_shape))


class TestSigmoidFocalLoss(chex.TestCase):
    """Test focal_loss for non-exclusive classes."""

    sigmoid_0 = 0.5
    sigmoid_1 = 1 / (1 + np.exp(-1))
    sigmoid_2 = 1 / (1 + np.exp(-2))
    sigmoid_n1 = 1 / (1 + np.exp(1))
    sigmoid_n2 = 1 / (1 + np.exp(2))
    log_sigmoid_0 = np.log(sigmoid_0)
    log_sigmoid_1 = np.log(sigmoid_1)
    log_sigmoid_2 = np.log(sigmoid_2)
    log_sigmoid_n1 = np.log(sigmoid_n1)
    log_sigmoid_n2 = np.log(sigmoid_n2)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d - gamma=0.0 - one hot",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            0.0,
            np.mean(
                -np.sum(
                    np.array(
                        [
                            [
                                [log_sigmoid_n2, log_sigmoid_n1, log_sigmoid_0],
                                [log_sigmoid_0, log_sigmoid_n1, log_sigmoid_2],
                                [log_sigmoid_0, log_sigmoid_1, log_sigmoid_2],
                            ]
                        ]
                    ),
                    axis=-1,
                )
            ),
        ),
        (
            "1d - gamma=0.0 - multi hot",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]]),
            0.0,
            np.mean(
                -np.sum(
                    np.array(
                        [
                            [
                                [log_sigmoid_2, log_sigmoid_n1, log_sigmoid_0],
                                [log_sigmoid_0, log_sigmoid_n1, log_sigmoid_n2],
                                [log_sigmoid_0, log_sigmoid_n1, log_sigmoid_2],
                            ]
                        ]
                    ),
                    axis=-1,
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
                -np.sum(
                    np.array(
                        [
                            [
                                [log_sigmoid_n2, log_sigmoid_n1, log_sigmoid_0],
                                [log_sigmoid_0, log_sigmoid_n1, log_sigmoid_2],
                            ],
                            [
                                [log_sigmoid_1, log_sigmoid_n2, log_sigmoid_0],
                                [log_sigmoid_n1, log_sigmoid_n1, log_sigmoid_0],
                            ],
                        ]
                    ),
                    axis=-1,
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
                -np.sum(
                    np.array(
                        [
                            [
                                [log_sigmoid_n2, log_sigmoid_1, log_sigmoid_0],
                                [log_sigmoid_0, log_sigmoid_n1, log_sigmoid_2],
                            ],
                            [
                                [log_sigmoid_1, log_sigmoid_2, log_sigmoid_0],
                                [log_sigmoid_n1, log_sigmoid_n1, log_sigmoid_0],
                            ],
                        ],
                    ),
                    axis=-1,
                )
            ),
        ),
        (
            "1d - gamma=2.0 - one hot",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            2.0,
            np.mean(
                -np.sum(
                    np.array(
                        [
                            [
                                [
                                    (sigmoid_2**2) * log_sigmoid_n2,
                                    (sigmoid_1**2) * log_sigmoid_n1,
                                    (sigmoid_0**2) * log_sigmoid_0,
                                ],
                                [
                                    (sigmoid_0**2) * log_sigmoid_0,
                                    (sigmoid_1**2) * log_sigmoid_n1,
                                    (sigmoid_n2**2) * log_sigmoid_2,
                                ],
                                [
                                    (sigmoid_0**2) * log_sigmoid_0,
                                    (sigmoid_n1**2) * log_sigmoid_1,
                                    (sigmoid_n2**2) * log_sigmoid_2,
                                ],
                            ]
                        ],
                    ),
                    axis=-1,
                ),
            ),
        ),
        (
            "1d-gamma=2.0-soft label",
            np.array([[[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]),
            np.array([[[0.1, 0.0, 0.8], [0.0, 0.7, 0.0], [0.6, 0.0, 0.0]]]),
            2.0,
            np.mean(
                -np.sum(
                    np.array(
                        [
                            [
                                [
                                    (sigmoid_n2**2) * log_sigmoid_2 * 0.1
                                    + (sigmoid_2**2) * log_sigmoid_n2 * 0.9,
                                    (sigmoid_1**2) * log_sigmoid_n1,
                                    (sigmoid_0**2) * log_sigmoid_0,
                                ],
                                [
                                    (sigmoid_0**2) * log_sigmoid_0,
                                    (sigmoid_n1**2) * log_sigmoid_1 * 0.3
                                    + (sigmoid_1**2) * log_sigmoid_n1 * 0.7,
                                    (sigmoid_n2**2) * log_sigmoid_2,
                                ],
                                [
                                    (sigmoid_0**2) * log_sigmoid_0,
                                    (sigmoid_n1**2) * log_sigmoid_1,
                                    (sigmoid_n2**2) * log_sigmoid_2,
                                ],
                            ]
                        ],
                    ),
                    axis=-1,
                ),
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
        """Test loss values.

        Args:
            logits: unscaled prediction, of shape (..., num_classes).
            mask_true: masks, of shape (..., num_classes).
            gamma: adjust class imbalance, 0 is equivalent to cross entropy.
            expected: expected output.
        """
        got = self.variant(focal_loss)(
            logits=logits,
            mask_true=mask_true,
            gamma=gamma,
            classes_are_exclusive=False,
        )
        chex.assert_trees_all_close(got, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d",
            (3,),
        ),
        (
            "2d",
            (3, 4),
        ),
        (
            "3d",
            (3, 4, 5),
        ),
    )
    def test_shapes(
        self,
        spatial_shape: tuple[int, ...],
    ) -> None:
        """Test loss shapes.

        Args:
            spatial_shape: spatial shape.
        """
        batch = 2
        num_classes = 3
        logits = np.ones((batch, *spatial_shape, num_classes), dtype=np.float32)
        mask_true = np.ones((batch, *spatial_shape, num_classes), dtype=np.float32)
        got = self.variant(sigmoid_focal_loss)(
            logits=logits,
            mask_true=mask_true,
        )
        chex.assert_shape(got, (batch, *spatial_shape, num_classes))
