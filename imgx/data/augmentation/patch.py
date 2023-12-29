"""Script for image patching."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from omegaconf import DictConfig

from imgx.data.augmentation import AugmentationFn
from imgx.data.util import get_batch_size
from imgx.datasets import INFO_MAP
from imgx.datasets.constant import IMAGE, LABEL


def batch_patch_random_sample(
    key: jax.Array,
    batch: dict[str, jnp.ndarray],
    image_shape: tuple[int, ...],
    patch_shape: tuple[int, ...],
) -> dict[str, jnp.ndarray]:
    """Randomly crop patch from image and label.

    The crop per sample in the batch is different.
    Image and label have no channel dimension.

    The crop in each sample is the same.

    Args:
        key: jax random key.
        batch: dict having images or labels, and foreground_range.
            images have shape (batch, d1, ..., dn) or (batch, d1, ..., dn, c)
            labels have shape (batch, d1, ..., dn)
            batch should not have other keys such as UID.
        image_shape: image spatial shape, (d1, ..., dn).
        patch_shape: patch size shape, (p1, ..., pn),
            patch_shape should <= image_shape for all dimensions.

    Returns:
        Augmented dict having image and label.
        image and label all have shapes (batch, p1, ..., pn).
    """
    batch_size = get_batch_size(batch)

    # define sample range
    indice_range = jnp.array(image_shape) - jnp.array(patch_shape)

    # sample a corner for each sample in the batch
    # (batch, n)
    start_indices = jax.random.randint(
        key,
        shape=(batch_size, len(image_shape)),
        minval=0,  # inclusive
        maxval=indice_range,  # exclusive
    )

    # slice per sample in the batch
    def slice_per_sample(x: jnp.ndarray, start_indices_i: jnp.ndarray) -> jnp.ndarray:
        """Slice per sample in the batch.

        Args:
            x: image or label, shape = (d1, ..., dn).
            start_indices_i: start indices, shape = (n,).

        Returns:
            Patch: shape = (p1, ..., pn).
        """
        return lax.dynamic_slice(x, start_indices_i, patch_shape)

    # crop patch
    cropped_batch = {}
    for k, v in batch.items():
        if LABEL in k:
            # assume label related keys have label in name
            # vmap on batch axis
            # (batch, p1, ..., pn)
            if v.ndim != len(image_shape) + 1:
                raise ValueError(f"Label {k} has wrong ndim {v.ndim}.")
            cropped_batch[k] = jax.vmap(
                slice_per_sample,
                in_axes=(0, 0),
            )(v, start_indices)
        elif IMAGE in k:
            # assume image related keys have image in name
            if v.ndim not in [len(image_shape) + 1, len(image_shape) + 2]:
                raise ValueError(f"Image {k} has wrong ndim {v.ndim}.")
            # vmap on batch axis
            slice_image_vmap = jax.vmap(
                slice_per_sample,
                in_axes=(0, 0),
            )
            if v.ndim == len(image_shape) + 2:
                # vmap on channel axis
                ch_axis = len(image_shape) + 1
                slice_image_vmap = jax.vmap(
                    slice_image_vmap,
                    in_axes=(ch_axis, None),
                    out_axes=ch_axis,
                )
            # (batch, p1, ..., pn) or (batch, p1, ..., pn, c)
            cropped_batch[k] = slice_image_vmap(v, start_indices)
    return cropped_batch


def get_patch_grid(
    image_shape: tuple[int, ...],
    patch_shape: tuple[int, ...],
    patch_overlap: tuple[int, ...],
) -> np.ndarray:
    """Get start_indices per patch following a grid.

    Use numpy due to for and if loops.

    https://github.com/fepegar/torchio/blob/main/src/torchio/data/sampler/grid.py

    Args:
        image_shape: image size, (d1, ..., dn).
        patch_shape: patch size, (p1, ..., pn),
            patch_shape should <= image_shape for all dimensions.
        patch_overlap: overlap between patches, (o1, ..., on),
            patch_overlap should <= patch_shape for all dimensions.

    Returns:
        Indices grid of shape (num_patches, n).
    """
    indices = []
    for img_size_dim, patch_size_dim, ovlp_size_dim in zip(image_shape, patch_shape, patch_overlap):
        # Example with image_size 10, patch_size 5, overlap 2:
        # [0 1 2 3 4 5 6 7 8 9]
        # [0 0 0 0 0]
        #       [1 1 1 1 1]
        #           [2 2 2 2 2]
        # indices_dim = [0, 3, 5]
        end = img_size_dim - patch_size_dim + 1
        step = patch_size_dim - ovlp_size_dim
        indices_dim = np.arange(0, end, step)
        if indices_dim[-1] != end - 1:
            indices_dim = np.append(indices_dim, img_size_dim - patch_size_dim)
        indices.append(indices_dim)
    return np.stack(np.meshgrid(*indices, indexing="ij"), axis=-1).reshape(-1, len(image_shape))


def batch_patch_grid_sample(
    x: jnp.ndarray,
    start_indices: np.ndarray,
    patch_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Extract patch following a grid.

    Args:
        x: has shape (batch, d1, ..., dn) or (batch, d1, ..., dn, c).
        start_indices: indices grid of shape (num_patches, n).
        patch_shape: patch size, shape = (p1, ..., pn),
            patch_shape should <= image_shape for all dimensions.

    Returns:
        Patched, has shapes (batch, num_patches, p1, ..., pn)
            or (batch, num_patches, p1, ..., pn, c).
    """
    if x.ndim not in [len(patch_shape) + 1, len(patch_shape) + 2]:
        raise ValueError(f"Image has wrong ndim {x.ndim}.")

    def slice_per_sample(
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Slice per sample in the batch.

        Args:
            x: shape = (d1, ..., dn).

        Returns:
            Patch: shape = (num_patches, p1, ..., pn).
        """
        return jax.vmap(
            lax.dynamic_slice,
            in_axes=(None, 0, None),
        )(x, start_indices, patch_shape)

    if x.ndim == len(patch_shape) + 2:
        # start_indices (num_patches, n+1).
        # patch_shape: patch size, shape = (p1, ..., pn, c),
        num_patches = start_indices.shape[0]
        num_channels = x.shape[-1]
        start_indices = np.concatenate(
            [
                start_indices,
                np.zeros((num_patches, 1), dtype=start_indices.dtype),
            ],
            axis=-1,
        )
        patch_shape = (*patch_shape, num_channels)
    # (batch, num_patches, p1, ..., pn) or (batch, num_patches, p1, ..., pn, c)
    return jax.vmap(
        slice_per_sample,
        in_axes=(0,),
    )(x)


def add_patch_with_channel(
    x: jnp.ndarray,
    count: jnp.ndarray,
    patch: jnp.ndarray,
    start_indices: np.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Add patch and update the count.

    Args:
        x: shape = (batch, d1, ..., dn, ...).
        count: shape = (d1, ..., dn).
            record the number of patches added to each position.
        patch: shape = (batch, p1, ..., pn, ...).
        start_indices: shape = (n,).

    Returns:
        Updated x and count.
    """
    image_shape = x.shape[1 : 1 + len(start_indices)]
    patch_shape = patch.shape[1 : 1 + len(start_indices)]
    num_channels = x.ndim - len(start_indices) - 1

    # update count with spatial only start_indices
    count += lax.dynamic_update_slice(
        jnp.zeros(image_shape),
        jnp.ones(patch_shape),
        start_indices,
    )

    # update start_indices with additional channels
    # then update x
    if num_channels > 0:
        aux_indices = jnp.array((0,) * num_channels)
        start_indices = jnp.append(start_indices, aux_indices)
    x += jax.vmap(
        lax.dynamic_update_slice,
        in_axes=(0, 0, None),
    )(
        jnp.zeros_like(x),
        patch,
        start_indices,
    )

    return x, count


def batch_patch_grid_mean_aggregate(
    x_patch: jnp.ndarray,
    start_indices: np.ndarray,
    image_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Aggregate patches by average on overlapping area following a grid.

    Args:
        x_patch: array of shape (batch, num_patches, p1, ..., pn, ...).
        start_indices: indices grid of shape (num_patches, n).
        image_shape: image size, (d1, ..., dn).

    Returns:
        Aggregated array of shape (batch, d1, ..., dn, ...).
    """
    num_spatial_dims = start_indices.shape[1]
    patch_shape = x_patch.shape[2 : 2 + num_spatial_dims]
    channel_dims = x_patch.shape[len(patch_shape) + 2 :]
    batch_size = x_patch.shape[0]

    # (num_patches, batch, p1, ..., pn, ...)
    x_patch = jnp.moveaxis(x_patch, 1, 0)

    def add_patch_with_channel_i(
        x_and_count: tuple[jnp.ndarray, jnp.ndarray],
        patch_and_start_indices: tuple[jnp.ndarray, np.ndarray],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
        """Add patch and update the count."""
        return (
            add_patch_with_channel(
                x=x_and_count[0],
                count=x_and_count[1],
                patch=patch_and_start_indices[0],
                start_indices=patch_and_start_indices[1],
            ),
            None,
        )

    # x is the summed image
    # count stores the number of patches that have been summed per voxel
    x = jnp.zeros((batch_size, *image_shape, *channel_dims))
    count = jnp.zeros(image_shape)

    (x, count), _ = lax.scan(
        f=add_patch_with_channel_i,
        init=(x, count),
        xs=(x_patch, start_indices),
    )
    broadcast_dimensions = tuple(range(1, len(image_shape) + 1))
    count = lax.broadcast_in_dim(
        count,
        shape=x.shape,
        broadcast_dimensions=broadcast_dimensions,
    )

    return x / count


def get_random_patch_fn(
    config: DictConfig,
) -> AugmentationFn:
    """Return a data augmentation function for patching.

    Args:
        config: entire config.

    Returns:
        A data augmentation function.
    """
    dataset_info = INFO_MAP[config.data.name]
    image_shape = dataset_info.image_spatial_shape
    patch_shape = config.data.loader.patch_shape
    if all(i <= j for i, j in zip(image_shape, patch_shape)):
        # no need to patch
        return lambda _, batch: batch
    return partial(
        batch_patch_random_sample,
        image_shape=image_shape,
        patch_shape=patch_shape,
    )
