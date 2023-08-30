"""Test time sampler class and related functions."""


import chex
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.diffusion.time_sampler import scatter_add, scatter_set


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestScatter(chex.TestCase):
    """Test scatter_add and scatter_set."""

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "add to zeros",
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0, 3, 0]),
            np.array([-1.0, 2.1, 1.2]),
            np.array([0.2, 0.0, 0.0, 2.1, 0.0]),
        ),
        (
            "add to non zeros",
            np.array([0.0, 1.0, 0.0, 2.0, 0.0]),
            np.array([0, 1, 2]),
            np.array([-1.0, 2.1, 1.2]),
            np.array([-1.0, 3.1, 1.2, 2.0, 0.0]),
        ),
    )
    def test_scatter_add(
        self,
        x: np.ndarray,
        indices: np.ndarray,
        updates: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        got = self.variant(scatter_add)(
            x,
            indices,
            updates,
        )
        chex.assert_trees_all_close(got, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "set to zeros",
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0, 3, 1]),
            np.array([-1.0, 2.1, 1.2]),
            np.array([-1.0, 1.2, 0.0, 2.1, 0.0]),
        ),
        (
            "set to non zeros",
            np.array([0.0, 1.0, 0.0, 2.0, 4.0]),
            np.array([0, 3, 1]),
            np.array([-1.0, 2.1, 1.2]),
            np.array([-1.0, 1.2, 0.0, 2.1, 4.0]),
        ),
    )
    def test_scatter_set(
        self,
        x: np.ndarray,
        indices: np.ndarray,
        updates: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        got = self.variant(scatter_set)(
            x,
            indices,
            updates,
        )
        chex.assert_trees_all_close(got, expected)
