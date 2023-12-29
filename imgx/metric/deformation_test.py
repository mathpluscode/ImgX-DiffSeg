"""Test deformation functions."""
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import SimpleITK as sitk  # noqa: N813
from absl.testing import parameterized
from chex._src import fake

from imgx.metric.deformation import gradient, gradient_along_axis, jacobian_det


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestGradientAlongAxis(chex.TestCase):
    """Test gradient_along_axis."""

    @chex.all_variants()
    @parameterized.product(
        shape=[(3,), (2, 3), (2, 3, 4), (2, 3, 4, 5)],
    )
    def test_shapes(
        self,
        shape: tuple[int, ...],
    ) -> None:
        """Test return shapes.

        Args:
            shape: input shape.
        """
        x = jnp.ones(shape)
        for axis in range(x.ndim):
            got = self.variant(partial(gradient_along_axis, axis=axis, spacing=1.0))(x)
            chex.assert_shape(got, shape)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d - case 1",
            np.array([2.0, 1.0, 0.0]),
            0,
            1.0,
            np.array([-0.5, -1.0, -0.5]),
        ),
        (
            "1d - case 2",
            np.array([-2.3, 1.1, -0.5]),
            0,
            1.0,
            np.array([1.7, 0.9, -0.8]),
        ),
        (
            "2d - axis 0",
            np.array([[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]),
            0,
            1.0,
            np.array([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]]),
        ),
        (
            "2d - axis 1",
            np.array([[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]),
            1,
            1.0,
            np.array([[-0.5, -1.0, -0.5], [-0.5, -1.0, -0.5], [-0.5, -1.0, -0.5]]),
        ),
        (
            "2d - axis 1",
            np.array([[2.0, 1.0, 0.0], [0.0, -1.0, -2.0], [0.0, -1.0, -2.0]]),
            1,
            2.0,
            np.array([[-0.25, -0.5, -0.25], [-0.25, -0.5, -0.25], [-0.25, -0.5, -0.25]]),
        ),
    )
    def test_values(
        self,
        x: jnp.ndarray,
        axis: int,
        spacing: float,
        expected: jnp.ndarray,
    ) -> None:
        """Test return values.

        Args:
            x: input.
            axis: axis to take gradient.
            spacing: spacing between each pixel/voxel.
            expected: expected output.
        """
        got = self.variant(partial(gradient_along_axis, axis=axis, spacing=spacing))(x)
        chex.assert_trees_all_close(got, expected)


class TestGradient(chex.TestCase):
    """Test gradient."""

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
        spacing = jnp.ones((len(shape),))
        got = self.variant(gradient)(x, spacing)
        chex.assert_shape(got, (self.batch, *shape, len(shape) - 1))

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "1d",
            np.array([[[-2.3], [1.1], [-0.5]]]),
            np.array([[1.7, 0.9, -0.8]])[..., None, None],
        ),
        (
            "batch 1d",
            np.array([[[2.0], [1.0], [0.1]], [[0.0], [-1.0], [-2.0]], [[0.2], [-1.0], [-2.0]]]),
            np.array([[-0.5, -0.95, -0.45], [-0.5, -1.0, -0.5], [-0.6, -1.1, -0.5]])[
                ..., None, None
            ],
        ),
        (
            "2d",
            # (1,3,1,3)
            np.array([[[[2.0, 1.0, 0.1]], [[0.0, -1.0, -2.0]], [[0.2, -1.0, -2.0]]]]),
            np.stack(
                [
                    np.array([[[[-1.0, -1.0, -1.05]], [[-0.9, -1.0, -1.05]], [[0.1, 0.0, 0.0]]]]),
                    np.zeros((1, 3, 1, 3)),
                ],
                axis=-1,
            ),
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
        got = self.variant(gradient)(x, spacing)
        chex.assert_trees_all_close(got, expected)


class TestJacobian(chex.TestCase):
    """Test Jacobian related functions."""

    batch = 2

    @chex.all_variants()
    @parameterized.product(
        shape=[(2, 3), (2, 3, 4), (2, 3, 4, 5)],
    )
    def test_jacobian_det_shape(
        self,
        shape: tuple[int, ...],
    ) -> None:
        """Test return shapes.

        Args:
            shape: input shape.
        """
        x = jnp.ones((self.batch, *shape, len(shape)))
        spacing = jnp.ones((len(shape),))
        got = self.variant(jacobian_det)(x, spacing)
        chex.assert_shape(got, (self.batch, *shape))

    @chex.all_variants()
    @parameterized.product(
        shape=[(2, 3), (2, 3, 4)],
    )
    def test_jacobian_det_values_with_zero_ddf(
        self,
        shape: tuple[int, ...],
    ) -> None:
        """Test return values.

        Args:
            shape: input shape.
        """
        ddf = jnp.zeros((1, *shape, len(shape)))
        spacing = jnp.ones((len(shape),))
        got = self.variant(jacobian_det)(ddf, spacing)[0]
        ddf_volume = sitk.GetImageFromArray(np.array(ddf[0]), isVector=True)
        jacobian_det_volume = sitk.DisplacementFieldJacobianDeterminant(ddf_volume)
        expected = sitk.GetArrayViewFromImage(jacobian_det_volume)
        chex.assert_trees_all_close(got, expected)
        chex.assert_trees_all_close(got, jnp.ones_like(got))

    @chex.all_variants()
    @parameterized.product(
        shape=[(2, 3), (2, 3, 4)],
    )
    def test_jacobian_det_values_with_rand_ddf(
        self,
        shape: tuple[int, ...],
    ) -> None:
        """Test return values.

        Args:
            shape: input shape.
        """
        key = jax.random.PRNGKey(0)
        ddf = jax.random.uniform(key, (1, *shape, len(shape)))
        spacing = jnp.ones((len(shape),))
        got = self.variant(jacobian_det)(ddf, spacing)[0]

        # sitk expects (dn, ..., d1, n)
        n = len(ddf.shape) - 2
        # (dn, ..., d1, n)
        ddf_sitk = np.transpose(ddf[0], axes=[*list(range(n - 1, -1, -1)), n])
        # GetSize() = (d1, ..., dn)
        ddf_volume = sitk.GetImageFromArray(np.array(ddf_sitk), isVector=True)
        jacobian_det_volume = sitk.DisplacementFieldJacobianDeterminant(ddf_volume)
        # (dn, ..., d1)
        expected = sitk.GetArrayViewFromImage(jacobian_det_volume)
        # (d1, ..., dn)
        expected = np.transpose(expected, axes=list(range(n - 1, -1, -1)))
        chex.assert_trees_all_close(got, expected)
