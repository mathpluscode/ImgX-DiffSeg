"""Image augmentation functions."""
from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates
from omegaconf import DictConfig

from imgx.data.patch import batch_patch_random_sample
from imgx.metric.centroid import get_coordinate_grid
from imgx_datasets import INFO_MAP
from imgx_datasets.constant import FOREGROUND_RANGE, IMAGE, LABEL


def get_2d_rotation_matrix(
    radians: jnp.ndarray,
) -> jnp.ndarray:
    """Return 2d rotation matrix given radians.

    The affine transformation applies as following:
        [x, = [[* * 0]  * [x,
         y,    [* * 0]     y,
         1]    [0 0 1]]    1]

    Args:
        radians: tuple of one values, correspond to xy planes.

    Returns:
        Rotation matrix of shape (3, 3).
    """
    sin, cos = jnp.sin(radians[0]), jnp.cos(radians[0])
    return jnp.array(
        [
            [cos, -sin, 0.0],
            [sin, cos, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def get_3d_rotation_matrix(
    radians: jnp.ndarray,
) -> jnp.ndarray:
    """Return 3d rotation matrix given radians.

    The affine transformation applies as following:
        [x, = [[* * * 0]  * [x,
         y,    [* * * 0]     y,
         z,    [* * * 0]     z,
         1]    [0 0 0 1]]    1]

    Args:
        radians: tuple of three values, correspond to yz, xz, xy planes.

    Returns:
        Rotation matrix of shape (4, 4).
    """
    affine = jnp.eye(4)

    # rotation of yz around x-axis
    sin, cos = jnp.sin(radians[0]), jnp.cos(radians[0])
    affine_ax = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos, -sin, 0.0],
            [0.0, sin, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    affine = jnp.matmul(affine_ax, affine)

    # rotation of zx around y-axis
    sin, cos = jnp.sin(radians[1]), jnp.cos(radians[1])
    affine_ax = jnp.array(
        [
            [cos, 0.0, sin, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin, 0.0, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    affine = jnp.matmul(affine_ax, affine)

    # rotation of xy around z-axis
    sin, cos = jnp.sin(radians[2]), jnp.cos(radians[2])
    affine_ax = jnp.array(
        [
            [cos, -sin, 0.0, 0.0],
            [sin, cos, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    affine = jnp.matmul(affine_ax, affine)

    return affine


def get_rotation_matrix(
    radians: jnp.ndarray,
) -> jnp.ndarray:
    """Return rotation matrix given radians.

    Args:
        radians: correspond to rotate around each axis.

    Returns:
        Rotation matrix of shape (n+1, n+1).

    Raises:
        ValueError: if not 2D or 3D.
    """
    if radians.size == 1:
        return get_2d_rotation_matrix(radians)
    if radians.size == 3:
        return get_3d_rotation_matrix(radians)
    raise ValueError("Only support 2D/3D rotations.")


def get_translation_matrix(
    shifts: jnp.ndarray,
) -> jnp.ndarray:
    """Return 3d translation matrix given shifts.

    For example, the 3D affine transformation applies as following:
        [x, = [[1 0 0 *]  * [x,
         y,    [0 1 0 *]     y,
         z,    [0 0 1 *]     z,
         1]    [0 0 0 1]]    1]

    Args:
        shifts: correspond to each axis shift.

    Returns:
        Translation matrix of shape (n+1, n+1).
    """
    ndims = shifts.size
    shifts = jnp.concatenate([shifts, jnp.array([1.0])])
    return jnp.concatenate(
        [
            jnp.eye(ndims + 1, ndims),
            shifts[:, None],
        ],
        axis=1,
    )


def get_scaling_matrix(
    scales: jnp.ndarray,
) -> jnp.ndarray:
    """Return scaling matrix given scales.

    For example, the 3D affine transformation applies as following:
        [x, = [[* 0 0 0]  * [x,
         y,    [0 * 0 0]     y,
         z,    [0 0 * 0]     z,
         1]    [0 0 0 1]]    1]

    Args:
        scales: correspond to each axis scaling.

    Returns:
        Affine matrix of shape (n+1, n+1).
    """
    scales = jnp.concatenate([scales, jnp.array([1.0])])
    return jnp.diag(scales)


def get_affine_matrix(
    radians: jnp.ndarray,
    shifts: jnp.ndarray,
    scales: jnp.ndarray,
) -> jnp.ndarray:
    """Return an affine matrix from parameters.

    The matrix is not squared, as the last row is not needed. For rotation,
    translation, and scaling matrix, they are kept for composition purpose.
    For example, the 3D affine transformation applies as following:
        [x, = [[* * * *]  * [x,
         y,    [* * * *]     y,
         z,    [* * * *]     z,
         1]    [0 0 0 1]]    1]

    Args:
        radians: correspond to rotate around each axis.
        shifts: correspond to each axis shift.
        scales: correspond to each axis scaling.

    Returns:
        Affine matrix of shape (n+1, n+1).
    """
    affine_rot = get_rotation_matrix(radians)
    affine_shift = get_translation_matrix(shifts)
    affine_scale = get_scaling_matrix(scales)
    return jnp.matmul(affine_shift, jnp.matmul(affine_scale, affine_rot))


def batch_get_random_affine_matrix(
    key: jax.random.PRNGKey,
    max_rotation: jnp.ndarray,
    min_translation: jnp.ndarray,
    max_translation: jnp.ndarray,
    max_scaling: jnp.ndarray,
) -> jnp.ndarray:
    """Get a batch of random affine matrices.

    Args:
        key: jax random key.
        max_rotation: maximum rotation in radians.
        min_translation: minimum translation in pixel/voxels.
        max_translation: maximum translation in pixel/voxels.
        max_scaling: maximum scaling difference in pixel/voxels.

    Returns:
        Affine matrix of shape (batch, n+1, n+1).
    """
    key_radian, key_shift, key_scale = jax.random.split(key, num=3)
    radians = jax.random.uniform(
        key=key_radian,
        shape=max_rotation.shape,
        minval=-max_rotation,
        maxval=max_rotation,
    )
    shifts = jax.random.uniform(
        key=key_shift,
        shape=max_translation.shape,
        minval=min_translation,
        maxval=max_translation,
    )
    scales = jax.random.uniform(
        key=key_scale,
        shape=max_scaling.shape,
        minval=1.0 - max_scaling,
        maxval=1.0 + max_scaling,
    )
    # vmap on first axis, which is a batch
    return jax.vmap(get_affine_matrix)(radians, shifts, scales)


def apply_affine_to_grid(
    grid: jnp.ndarray, affine_matrix: jnp.ndarray
) -> jnp.ndarray:
    """Apply affine matrix to grid.

    The grid has non-negative coordinates, means the origin is at a corner.
    Need to shift the grid such that the origin is at center,
    then apply affine, then shift the origin back.

    Args:
        grid: grid coordinates, of shape  (n, d1, ..., dn).
            grid[:, i1, ..., in] = [i1, ..., in]
        affine_matrix: shape (n+1, n+1)

    Returns:
        Grid with updated coordinates.
    """
    # (n+1, d1, ..., dn)
    extended_grid = jnp.concatenate(
        [grid, jnp.ones((1,) + grid.shape[1:])], axis=0
    )

    # shift to center
    shift = (jnp.array(grid.shape[1:]) - 1) / 2
    shift_matrix = get_translation_matrix(-shift)  # (n+1, n+1)
    # (n+1, n+1) * (n+1, d1, ..., dn) = (n+1, d1, ..., dn)
    extended_grid = jnp.einsum("ji,i...->j...", shift_matrix, extended_grid)

    # affine
    # (n+1, n+1) * (n+1, d1, ..., dn) = (n+1, d1, ..., dn)
    extended_grid = jnp.einsum("ji,i...->j...", affine_matrix, extended_grid)

    # shift to corner
    shift_matrix = get_translation_matrix(shift)[:-1, :]  # (n, n+1)
    # (n, n+1) * (n+1, d1, ..., dn) = (n, d1, ..., dn)
    extended_grid = jnp.einsum("ji,i...->j...", shift_matrix, extended_grid)

    return extended_grid


def batch_apply_affine_to_grid(
    grid: jnp.ndarray, affine_matrix: jnp.ndarray
) -> jnp.ndarray:
    """Apply batch of affine matrix to grid.

    Args:
        grid: grid coordinates, of shape  (n, d1, ..., dn).
            grid[:, i1, ..., in] = [i1, ..., in]
        affine_matrix: shape (batch, n+1, n+1).

    Returns:
        Grid with updated coordinates, shape (batch, n, d1, ..., dn).
    """
    return jax.vmap(apply_affine_to_grid, in_axes=(None, 0))(
        grid, affine_matrix
    )


def batch_resample_image_label(
    input_dict: dict[str, jnp.ndarray],
    grid: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Apply batch of affine matrix to image and label.

    Args:
        input_dict: dict having image and label.
            image may have shape (batch, d1, ..., dn) or (batch, d1, ..., dn, c)
            label has shape (batch, d1, ..., dn)
        grid: grid coordinates, of shape (batch, n, d1, ..., dn).

    Returns:
        Updated image and label, of same shape.
    """
    image = input_dict[IMAGE]
    label = input_dict[LABEL]

    # check shapes
    if image.ndim not in [label.ndim, label.ndim + 1]:
        raise ValueError(
            f"image and label must have same ndim or ndim+1, "
            f"got {image.ndim} and {label.ndim} "
            f"for image and label, correspondingly."
        )

    # vmap on batch axis
    resample_image_vmap = jax.vmap(
        partial(
            map_coordinates,
            order=1,
            mode="constant",
            cval=0.0,
        ),
        in_axes=(0, 0),
    )
    if image.ndim == label.ndim + 1:
        # vmap on channel axis
        ch_axis = image.ndim - 1
        resample_image_vmap = jax.vmap(
            resample_image_vmap,
            in_axes=(ch_axis, None),
            out_axes=ch_axis,
        )
    # vmap on batch axis
    resample_label_vmap = jax.vmap(
        partial(
            map_coordinates,
            order=0,
            mode="constant",
            cval=0.0,
        ),
        in_axes=(0, 0),
    )
    image = resample_image_vmap(image, grid)
    label = resample_label_vmap(label, grid)
    return {IMAGE: image, LABEL: label}


def batch_random_affine_transform(
    key: jax.random.PRNGKey,
    input_dict: dict[str, jnp.ndarray],
    grid: jnp.ndarray,
    max_rotation: jnp.ndarray,
    max_translation: jnp.ndarray,
    max_scaling: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Keep image and label only.

    Args:
        key: jax random key.
        input_dict: dict having image and label.
            image may have shape (batch, d1, ..., dn) or (batch, d1, ..., dn, c)
            label has shape (batch, d1, ..., dn)
        grid: grid coordinates, of shape  (n, d1, ..., dn).
            grid[:, i1, ..., in] = [i1, ..., in]
        max_rotation: maximum rotation in radians, shape = (batch, ...).
        max_translation: maximum translation in pixel/voxels,
            shape = (batch, d1, ..., dn).
        max_scaling: maximum scaling difference in pixel/voxels,
            shape = (batch, d1, ..., dn).

    Returns:
        Augmented dict having image and label, shapes are not changed.
    """
    # (batch, ...)
    batch_size = input_dict[IMAGE].shape[0]
    max_rotation = jnp.tile(max_rotation[None, ...], (batch_size, 1))
    max_translation = jnp.tile(max_translation[None, ...], (batch_size, 1))
    min_translation = -max_translation
    max_scaling = jnp.tile(max_scaling[None, ...], (batch_size, 1))

    # refine translation to avoid remove classes
    shape = jnp.array(input_dict[LABEL].shape[1:])
    shape = jnp.tile(shape[None, ...], (batch_size, 1))
    max_translation = jnp.minimum(
        max_translation, shape - 1 - input_dict[FOREGROUND_RANGE][..., -1]
    )
    min_translation = jnp.maximum(
        min_translation, -input_dict[FOREGROUND_RANGE][..., 0]
    )
    # (batch, n+1, n+1)
    affine_matrix = batch_get_random_affine_matrix(
        key=key,
        max_rotation=max_rotation,
        min_translation=min_translation,
        max_translation=max_translation,
        max_scaling=max_scaling,
    )

    # (batch, n, d1, ..., dn)
    grid = batch_apply_affine_to_grid(grid=grid, affine_matrix=affine_matrix)

    return batch_resample_image_label(
        input_dict=input_dict,
        grid=grid,
    )


def build_aug_fn_from_fns(
    fns: Sequence[Callable],
) -> Callable:
    """Combine a list of data augmentation functions.

    Args:
        fns: entire config.

    Returns:
        A data augmentation function.
    """

    def aug_fn(
        key: jax.random.PRNGKey, input_dict: dict[str, jnp.ndarray]
    ) -> dict[str, jnp.ndarray]:
        keys = jax.random.split(key, num=len(fns))
        for k, fn in zip(keys, fns):
            input_dict = fn(k, input_dict)
        return input_dict

    return aug_fn


def build_aug_fn_from_config(
    config: DictConfig,
) -> Callable:
    """Return a data augmentation function.

    Args:
        config: entire config.

    Returns:
        A data augmentation function.
    """
    data_config = config.data
    dataset_name = data_config["name"]
    dataset_info = INFO_MAP[dataset_name]
    image_shape = dataset_info.image_spatial_shape
    da_config = data_config.loader.data_augmentation
    patch_shape = data_config.loader.patch_shape

    grid = get_coordinate_grid(shape=image_shape)
    max_rotation = np.array(da_config["max_rotation"])
    max_translation = np.array(da_config["max_translation"])
    max_scaling = np.array(da_config["max_scaling"])

    aug_fns = [
        partial(
            batch_random_affine_transform,
            grid=grid,
            max_rotation=max_rotation,
            max_translation=max_translation,
            max_scaling=max_scaling,
        ),
        partial(
            batch_patch_random_sample,
            image_shape=image_shape,
            patch_shape=patch_shape,
        ),
    ]

    if len(aug_fns) == 1:
        return aug_fns[0]
    return build_aug_fn_from_fns(aug_fns)
