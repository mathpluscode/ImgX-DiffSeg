"""Test the flip functions."""


from functools import partial

import chex
import jax
import numpy as np
from absl.testing import parameterized
from chex._src import fake

from imgx.data.augmentation.flip import batch_random_flip, random_flip
from imgx.datasets.constant import IMAGE


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestRandomFlip(chex.TestCase):
    """Test random_flip."""

    @chex.all_variants()
    @parameterized.named_parameters(
        ("1d - flip", np.array([1, 2, 3, 4]), np.array([True]), np.array([4, 3, 2, 1])),
        ("1d - no flip", np.array([1, 2, 3, 4]), np.array([False]), np.array([1, 2, 3, 4])),
        (
            "2d - no flip",
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            np.array([False, False]),
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        ),
        (
            "2d - flip the first axis",
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            np.array([True, False]),
            np.array([[5, 6, 7, 8], [1, 2, 3, 4]]),
        ),
        (
            "2d - flip the second axis",
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            np.array([False, True]),
            np.array([[4, 3, 2, 1], [8, 7, 6, 5]]),
        ),
        (
            "2d - flip both axes",
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            np.array([True, True]),
            np.array([[8, 7, 6, 5], [4, 3, 2, 1]]),
        ),
    )
    def test_values(
        self,
        x: np.ndarray,
        to_flip: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        """Test output shapes.

        Args:
            x: input array.
            to_flip: (n, ), with boolean values, True means to flip along that axis.
            expected: expected output array.
        """
        got = self.variant(random_flip)(
            x=x,
            to_flip=to_flip,
        )

        chex.assert_trees_all_equal(got, expected)

    @chex.all_variants()
    @parameterized.product(
        image_shape=[(3, 4, 5, 6), (3, 4, 5), (3, 4), (3,)],
    )
    def test_shapes(
        self,
        image_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes.

        Args:
            image_shape: image spatial shape.
        """
        key = jax.random.PRNGKey(0)
        key_flip, key_image = jax.random.split(key, 2)
        image = jax.random.uniform(key=key_image, shape=image_shape, minval=0, maxval=1)
        to_flip = jax.random.uniform(key=key_flip, shape=(len(image_shape),)) < 1 / 2
        got = self.variant(random_flip)(
            x=image,
            to_flip=to_flip,
        )

        chex.assert_shape(got, image_shape)


class TestBatchRandomFlip(chex.TestCase):
    """Test batch_random_flip."""

    batch_size = 2

    @chex.all_variants()
    @parameterized.product(
        image_shape=[(3, 4, 5, 6), (3, 4, 5), (3, 4), (3,)],
        p=[0.0, 0.5, 1.0],
    )
    def test_shapes(
        self,
        image_shape: tuple[int, ...],
        p: float,
    ) -> None:
        """Test output shapes.

        Args:
            image_shape: image spatial shape.
            p: probability to flip for each axis.
        """
        key = jax.random.PRNGKey(0)
        key_image, key = jax.random.split(key)
        num_spatial_dims = len(image_shape)
        image = jax.random.uniform(
            key=key_image, shape=(self.batch_size, *image_shape), minval=0, maxval=1
        )
        batch = {IMAGE: image}
        got = self.variant(partial(batch_random_flip, num_spatial_dims=num_spatial_dims, p=p))(
            key, batch
        )

        chex.assert_shape(got[IMAGE], (self.batch_size, *image_shape))
        if p == 0:
            chex.assert_trees_all_equal(got, batch)
