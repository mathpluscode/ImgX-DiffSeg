"""Test slicing functions."""

from functools import partial

import chex
import jax.numpy as jnp
from absl.testing import parameterized
from chex._src import fake

from imgx.model.slice import merge_spatial_dim_into_batch, split_spatial_dim_from_batch


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestMergeSplitSpatialDims(chex.TestCase):
    """Test merge_spatial_dim_into_batch and split_spatial_dim_from_batch."""

    @chex.all_variants()
    @parameterized.named_parameters(
        ("2d-1", (2, 3, 4, 5), 1, (8, 3, 5)),
        ("2d-2", (2, 3, 4, 5), 2, (2, 3, 4, 5)),
        ("3d-1", (2, 3, 4, 5, 6), 1, (40, 3, 6)),
        ("3d-2", (2, 3, 4, 5, 6), 2, (10, 3, 4, 6)),
        ("3d-3", (2, 3, 4, 5, 6), 3, (2, 3, 4, 5, 6)),
    )
    def test_merge_spatial_dim_into_batch(
        self,
        in_shape: tuple[int, ...],
        num_spatial_dims: int,
        expected_shape: tuple[int, ...],
    ) -> None:
        """Test merge_spatial_dim_into_batch.

        Args:
            in_shape: input shape.
            num_spatial_dims: number of spatial dimensions.
            expected_shape: expected output shape.
        """
        x = jnp.ones(in_shape)
        x = self.variant(partial(merge_spatial_dim_into_batch, num_spatial_dims=num_spatial_dims))(
            x
        )
        chex.assert_shape(x, expected_shape)

    @chex.all_variants()
    @parameterized.named_parameters(
        ("2d-1", (8, 3, 5), 1, 2, (3, 4), (2, 3, 4, 5)),
        ("2d-2", (2, 3, 4, 5), 2, 2, (3, 4), (2, 3, 4, 5)),
        ("3d-1", (40, 3, 6), 1, 2, (3, 4, 5), (2, 3, 4, 5, 6)),
        ("3d-2", (10, 3, 4, 6), 2, 2, (3, 4, 5), (2, 3, 4, 5, 6)),
        ("3d-3", (2, 3, 4, 5, 6), 3, 2, (3, 4, 5), (2, 3, 4, 5, 6)),
    )
    def test_split_spatial_dim_from_batch(
        self,
        in_shape: tuple[int, ...],
        num_spatial_dims: int,
        batch_size: int,
        spatial_shape: tuple[int, ...],
        expected_shape: tuple[int, ...],
    ) -> None:
        """Test split_spatial_dim_from_batch.

        Args:
            in_shape: input shape, with certain spatial axes merged.
            num_spatial_dims: number of spatial dimensions.
            batch_size: batch size.
            spatial_shape: spatial shape.
            expected_shape: expected output shape.
        """
        x = jnp.ones(in_shape)
        x = self.variant(
            partial(
                split_spatial_dim_from_batch,
                num_spatial_dims=num_spatial_dims,
                spatial_shape=spatial_shape,
                batch_size=batch_size,
            )
        )(x)
        chex.assert_shape(x, expected_shape)
