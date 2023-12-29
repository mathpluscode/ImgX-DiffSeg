"""Test deformation loss functions."""
from functools import partial

import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.loss.deformation import bending_energy_loss, gradient_norm_loss, jacobian_loss


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestGradientNormLoss(chex.TestCase):
    """Test gradient_norm_loss."""

    batch = 2

    @chex.all_variants()
    @parameterized.product(
        shape=[(2, 3), (2, 3, 4), (2, 3, 4, 5)],
        norm_ord=[1, 2],
    )
    def test_shapes(
        self,
        shape: tuple[int, ...],
        norm_ord: int,
    ) -> None:
        """Test return shapes.

        Args:
            shape: input shape.
            norm_ord: 1 for L1 or 2 for L2.
        """
        x = jnp.ones((self.batch, *shape))
        spacing = jnp.ones((len(shape),))
        got = self.variant(partial(gradient_norm_loss, norm_ord=norm_ord, spacing=spacing))(x)
        chex.assert_shape(got, (self.batch,))

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d - L1 norm",
            np.array([[[-2.3], [1.1], [-0.5]]]),
            1,
            # norm [1.7, 0.9, -0.8]
            np.array([3.4 / 3]),
        ),
        (
            "1d - L2 norm",
            np.array([[[-2.3], [1.1], [-0.5]]]),
            2,
            # norm [1.7, 0.9, -0.8]
            np.array([(1.7 * 1.7 + 0.9 * 0.9 + 0.8 * 0.8) / 3]),
        ),
        (
            "batch 1d - L1 norm",
            np.array([[[2.0], [1.0], [0.1]], [[0.0], [-1.0], [-2.0]], [[0.2], [-1.0], [-2.0]]]),
            1,
            # norm  (singleton axis is removed)
            # [[-0.5, -0.95, -0.45],
            #  [-0.5, -1.0, -0.5],
            #  [-0.6, -1.1, -0.5]],
            np.array([1.9 / 3, 2.0 / 3, 2.2 / 3]),
        ),
        (
            "2d - L1 norm",
            # (1,3,1,3)
            np.array([[[[2.0, 1.0, 0.1]], [[0.0, -1.0, -2.0]], [[0.2, -1.0, -2.0]]]]),
            1,
            # dx norm (singleton axis is removed)
            # [[-1.0, -1.0, -1.05],
            #  [-0.9, -1.0, -1.05],
            #  [0.1, 0.0, 0.0]]
            # dy norm are all zeros
            np.array([6.1 / 18]),
        ),
    )
    def test_values(
        self,
        x: jnp.ndarray,
        norm_ord: int,
        expected: jnp.ndarray,
    ) -> None:
        """Test return values.

        Args:
            x: input.
            norm_ord: norm order.
            expected: expected output.
        """
        spacing = jnp.ones((x.ndim - 2,))
        got = self.variant(partial(gradient_norm_loss, norm_ord=norm_ord, spacing=spacing))(x)
        chex.assert_trees_all_close(got, expected)


class TestBendingEnergyLoss(chex.TestCase):
    """Test bending_energy_loss."""

    batch = 2

    @chex.all_variants()
    @parameterized.product(
        shape=[(2, 3), (2, 3, 4), (2, 3, 4, 5)],
    )
    def test_shapes(
        self,
        shape: tuple[int, ...],
    ) -> None:
        """Test return shapes.

        Args:
            shape: input shape.
        """
        x = jnp.ones((self.batch, *shape))
        spacing = jnp.ones((x.ndim - 2,))
        got = self.variant(bending_energy_loss)(x, spacing)
        chex.assert_shape(got, (self.batch,))

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d",
            # (1, 3, 1)
            np.array([[[-2.3], [1.1], [-0.5]]]),
            # dx [1.7, 0.9, -0.8]
            # dxx [-0.4, -1.25, -0.85]
            np.array([(0.4 * 0.4 + 1.25 * 1.25 + 0.85 * 0.85) / 3]),
        ),
    )
    def test_values(
        self,
        x: jnp.ndarray,
        expected: jnp.ndarray,
    ) -> None:
        """Test return values.

        Args:
            x: input.
            expected: expected output.
        """
        spacing = jnp.ones((x.ndim - 2,))
        got = self.variant(bending_energy_loss)(x, spacing)
        chex.assert_trees_all_close(got, expected)


class TestJacobianLoss(chex.TestCase):
    """Test jacobian_loss."""

    batch = 2

    @chex.all_variants()
    @parameterized.product(
        shape=[(2, 3), (2, 3, 4), (2, 3, 4, 5)],
    )
    def test_shapes(
        self,
        shape: tuple[int, ...],
    ) -> None:
        """Test return shapes.

        Args:
            shape: input shape.
        """
        x = jnp.ones((self.batch, *shape, len(shape)))
        spacing = jnp.ones((x.ndim - 2,))
        got = self.variant(jacobian_loss)(x, spacing)
        chex.assert_shape(got, (self.batch,))
