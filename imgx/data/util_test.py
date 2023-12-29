"""Tests for image utils of datasets."""


import chex
import numpy as np
import pytest
from chex._src import fake

from imgx.data.util import get_foreground_range


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (
            np.array([0, 1, 2, 3]),
            np.array([[1, 3]]),
        ),
        (
            np.array([1, 2, 3, 0]),
            np.array([[0, 2]]),
        ),
        (
            np.array([1, 2, 3, 4]),
            np.array([[0, 3]]),
        ),
        (
            np.array([0, 1, 2, 3, 4, 0, 0]),
            np.array([[1, 4]]),
        ),
        (
            np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 0, 0, 0]]),
            np.array([[0, 1], [1, 3]]),
        ),
    ],
    ids=[
        "1d-left",
        "1d-right",
        "1d-none",
        "1d-both",
        "2d",
    ],
)
def test_get_foreground_range(
    label: np.ndarray,
    expected: np.ndarray,
) -> None:
    """Test get_translation_range return values.

    Args:
        label: label with int values, not one-hot.
        expected: expected range.
    """
    got = get_foreground_range(
        label=label,
    )
    chex.assert_trees_all_equal(got, expected)
