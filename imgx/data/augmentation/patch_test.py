"""Script to test dataset patching."""

from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from chex._src import fake
from jax import lax

from imgx.data.augmentation.patch import (
    add_patch_with_channel,
    batch_patch_grid_mean_aggregate,
    batch_patch_grid_sample,
    batch_patch_random_sample,
    get_patch_grid,
)
from imgx.datasets.constant import IMAGE, LABEL


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestBatchPatchRandomSample(chex.TestCase):
    """Test batch_patch_random_sample."""

    batch_size = 2

    @chex.all_variants()
    @parameterized.product(
        (
            # 2d
            {
                "patch_shape": (8, 8),
                "image_shape": (10, 12),
            },
            # 3d
            {
                "patch_shape": (4, 5, 6),
                "image_shape": (8, 10, 6),
            },
        ),
        num_channels=[0, 1, 2],
    )
    def test_shapes(
        self,
        patch_shape: tuple[int, ...],
        image_shape: tuple[int, ...],
        num_channels: int,
    ) -> None:
        """Test random cropped patch shapes.

        Args:
            patch_shape: patch size.
            image_shape: image spatial shape.
            num_channels: number of channels.
        """
        key = jax.random.PRNGKey(0)
        image_key, label_key = jax.random.split(key)
        if num_channels > 0:
            image_in_shape = (self.batch_size, *image_shape, num_channels)
            image_out_shape = (self.batch_size, *patch_shape, num_channels)
        else:
            image_in_shape = (self.batch_size, *image_shape)
            image_out_shape = (self.batch_size, *patch_shape)
        image = jax.random.uniform(
            key=image_key,
            shape=image_in_shape,
            minval=0,
            maxval=1,
        )
        label = jax.random.uniform(
            key=label_key,
            shape=(self.batch_size, *image_shape),
            minval=0,
            maxval=1,
        )
        label = jnp.asarray(label > jnp.mean(label), dtype=np.float32)
        batch = {
            IMAGE: image,
            LABEL: label,
        }
        got = self.variant(partial(batch_patch_random_sample, patch_shape=patch_shape))(
            key=key,
            batch=batch,
            image_shape=image_shape,
        )

        # check shapes
        assert len(got) == 2
        chex.assert_shape(got[IMAGE], image_out_shape)
        chex.assert_shape(got[LABEL], (self.batch_size, *patch_shape))

        # check label remains boolean
        assert jnp.unique(got[LABEL]).size == jnp.unique(label).size


class TestGetPatchGrid(chex.TestCase):
    """Test get_patch_grid."""

    @parameterized.named_parameters(
        (
            "2d - with overlap",
            (3, 5),
            (6, 11),
            (1, 3),
            np.array(
                [
                    [0, 0],
                    [0, 2],
                    [0, 4],
                    [0, 6],
                    [2, 0],
                    [2, 2],
                    [2, 4],
                    [2, 6],
                    [3, 0],
                    [3, 2],
                    [3, 4],
                    [3, 6],
                ]
            ),
        ),
        (
            "2d - patch is image",
            (3, 5),
            (3, 5),
            (0, 0),
            np.array(
                [
                    [0, 0],
                ]
            ),
        ),
        (
            "2d - patch < image",
            (3, 5),
            (3, 7),
            (0, 0),
            np.array(
                [
                    [0, 0],
                    [0, 2],
                ]
            ),
        ),
        (
            "3d",
            (4, 5, 6),
            (8, 10, 6),
            (2, 1, 4),
            np.array(
                [
                    [0, 0, 0],
                    [0, 4, 0],
                    [0, 5, 0],
                    [2, 0, 0],
                    [2, 4, 0],
                    [2, 5, 0],
                    [4, 0, 0],
                    [4, 4, 0],
                    [4, 5, 0],
                ]
            ),
        ),
        (
            "3d - amos",
            (128, 128, 128),
            (192, 128, 128),
            (64, 0, 0),
            np.array(
                [
                    [0, 0, 0],
                    [64, 0, 0],
                ]
            ),
        ),
        (
            "3d - male pelvic",
            (256, 256, 32),
            (256, 256, 48),
            (0, 0, 16),
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 16],
                ]
            ),
        ),
    )
    def test_values(
        self,
        patch_shape: tuple[int, ...],
        image_shape: tuple[int, ...],
        patch_overlap: tuple[int, ...],
        expected: jnp.ndarray,
    ) -> None:
        """Test get_patch_grid return values.

        Args:
            patch_shape: patch size.
            image_shape: image spatial shape.
            patch_overlap: overlap between patches.
            expected: expected output.
        """
        got = get_patch_grid(
            image_shape=image_shape,
            patch_shape=patch_shape,
            patch_overlap=patch_overlap,
        )
        chex.assert_trees_all_equal(got, expected)


class TestBatchPatchGridSample(chex.TestCase):
    """Test batch_patch_grid_sample."""

    batch_size = 3

    @chex.all_variants()
    @parameterized.product(
        (
            # 2d
            {
                "patch_shape": (3, 5),
                "image_shape": (6, 11),
                "patch_overlap": (1, 3),
                "num_patches": 12,
            },
            # 3d
            {
                "patch_shape": (4, 5, 6),
                "image_shape": (8, 10, 6),
                "patch_overlap": (2, 1, 4),
                "num_patches": 9,
            },
        ),
        num_channels=[0, 1, 2],
    )
    def test_shapes(
        self,
        patch_shape: tuple[int, ...],
        image_shape: tuple[int, ...],
        patch_overlap: tuple[int, ...],
        num_patches: int,
        num_channels: int,
    ) -> None:
        """Test batch_grid_patch shapes.

        Args:
            patch_shape: patch size.
            image_shape: image spatial shape.
            patch_overlap: overlap between patches.
            num_patches: number of patches.
            num_channels: number of channels for images.
        """
        key = jax.random.PRNGKey(0)
        if num_channels > 0:
            image_in_shape = (self.batch_size, *image_shape, num_channels)
            out_shape = (
                self.batch_size,
                num_patches,
                *patch_shape,
                num_channels,
            )
        else:
            image_in_shape = (self.batch_size, *image_shape)
            out_shape = (self.batch_size, num_patches, *patch_shape)
        x = jax.random.uniform(
            key=key,
            shape=image_in_shape,
            minval=0,
            maxval=1,
        )
        start_indices = get_patch_grid(
            image_shape=image_shape,
            patch_shape=patch_shape,
            patch_overlap=patch_overlap,
        )
        got = self.variant(
            partial(
                batch_patch_grid_sample,
                patch_shape=patch_shape,
                start_indices=start_indices,
            )
        )(x=x)

        # check shapes
        chex.assert_shape(got, out_shape)
        agg_x = self.variant(partial(batch_patch_grid_mean_aggregate, image_shape=image_shape))(
            x_patch=got,
            start_indices=start_indices,
        )
        chex.assert_trees_all_close(agg_x, x)


class TestAddPatchWithChannel(chex.TestCase):
    """Test add_patch_with_channel."""

    batch_size = 2

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "2d - no channel",
            (3, 5),
            (6, 11),
            (1, 3),
            0,
        ),
        (
            "2d - one channel",
            (3, 5),
            (6, 11),
            (1, 3),
            1,
        ),
        (
            "2d - two channels",
            (3, 5),
            (6, 11),
            (1, 3),
            2,
        ),
        (
            "3d - one channel",
            (4, 5, 6),
            (8, 10, 6),
            (2, 1, 4),
            1,
        ),
    )
    def test_values(
        self,
        patch_shape: tuple[int, ...],
        image_shape: tuple[int, ...],
        start_indices: tuple[int, ...],
        num_channels: int,
    ) -> None:
        """Test add_patch_with_channel shapes.

        Args:
            patch_shape: patch size.
            image_shape: image spatial shape.
            start_indices: shape = (n,).
            num_channels: number of channels after spatial channels.
        """
        channel_dims = (2,) * num_channels
        aux_indices = (0,) * num_channels
        key = jax.random.PRNGKey(0)
        patch = jax.random.uniform(
            key=key,
            shape=(self.batch_size, *patch_shape, *channel_dims),
            minval=0,
            maxval=1,
        )
        x = jnp.zeros((self.batch_size, *image_shape, *channel_dims))
        count = jnp.zeros(image_shape)
        x, count = self.variant(add_patch_with_channel)(
            x=x,
            count=count,
            patch=patch,
            start_indices=jnp.array(start_indices, dtype=jnp.int32),
        )

        # check values
        x_patch = lax.dynamic_slice(
            x,
            start_indices=jnp.array((0, *start_indices, *aux_indices)),
            slice_sizes=(self.batch_size, *patch_shape, *channel_dims),
        )
        count_patch = lax.dynamic_slice(
            count,
            start_indices=jnp.array(start_indices),
            slice_sizes=patch_shape,
        )
        chex.assert_trees_all_equal(x_patch, patch)
        chex.assert_trees_all_equal(
            count_patch,
            jnp.ones(patch_shape),
        )


class TestBatchPatchGridMeanAggregate(chex.TestCase):
    """Test batch_patch_grid_mean_aggregate."""

    batch_size = 2

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "2d - one channel",
            (3, 5),
            (6, 11),
            (1, 3),
            1,
        ),
        (
            "3d - no channel",
            (4, 5, 6),
            (8, 10, 6),
            (2, 1, 4),
            0,
        ),
        (
            "3d - one channel",
            (4, 5, 6),
            (8, 10, 6),
            (2, 1, 4),
            1,
        ),
        (
            "3d - two channels",
            (4, 5, 6),
            (8, 10, 6),
            (2, 1, 4),
            2,
        ),
    )
    def test_shapes(
        self,
        patch_shape: tuple[int, ...],
        image_shape: tuple[int, ...],
        patch_overlap: tuple[int, ...],
        num_channels: int,
    ) -> None:
        """Test batch_patch_grid_mean_aggregate shapes.

        Args:
            patch_shape: patch size.
            image_shape: image spatial shape.
            patch_overlap: overlap between patches.
            num_channels: number of channels after spatial channels.
        """
        # (num_patches, n)
        start_indices = get_patch_grid(
            image_shape=image_shape,
            patch_shape=patch_shape,
            patch_overlap=patch_overlap,
        )
        num_patches = start_indices.shape[0]

        channel_dims = (2,) * num_channels
        key = jax.random.PRNGKey(0)
        x_patch = jax.random.uniform(
            key=key,
            shape=(self.batch_size, num_patches, *patch_shape, *channel_dims),
            minval=0,
            maxval=1,
        )
        x = self.variant(partial(batch_patch_grid_mean_aggregate, image_shape=image_shape))(
            x_patch=x_patch,
            start_indices=start_indices,
        )

        # check shapes
        chex.assert_shape(x, (self.batch_size, *image_shape, *channel_dims))
