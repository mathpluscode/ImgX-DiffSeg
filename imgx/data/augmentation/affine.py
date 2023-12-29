"""Affine augmentation for image and label."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from imgx.data.augmentation import AugmentationFn
from imgx.data.util import get_batch_size
from imgx.data.warp import batch_grid_sample, get_coordinate_grid
from imgx.datasets import INFO_MAP
from imgx.datasets.constant import FOREGROUND_RANGE, IMAGE, LABEL


def get_2d_rotation_matrix(
    rotations: jnp.ndarray,
) -> jnp.ndarray:
    """Return 2d rotation matrix given radians.

    The affine transformation applies as following:
        [x, = [[* * 0]  * [x,
         y,    [* * 0]     y,
         1]    [0 0 1]]    1]

    Args:
        rotations: tuple of one values, correspond to xy planes.

    Returns:
        Rotation matrix of shape (3, 3).
    """
    sin, cos = jnp.sin(rotations[0]), jnp.cos(rotations[0])
    return jnp.array(
        [
            [cos, -sin, 0.0],
            [sin, cos, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def get_3d_rotation_matrix(
    rotations: jnp.ndarray,
) -> jnp.ndarray:
    """Return 3d rotation matrix given radians.

    The affine transformation applies as following:
        [x, = [[* * * 0]  * [x,
         y,    [* * * 0]     y,
         z,    [* * * 0]     z,
         1]    [0 0 0 1]]    1]

    Args:
        rotations: tuple of three values, correspond to yz, xz, xy planes.

    Returns:
        Rotation matrix of shape (4, 4).
    """
    affine = jnp.eye(4)

    # rotation of yz around x-axis
    sin, cos = jnp.sin(rotations[0]), jnp.cos(rotations[0])
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
    sin, cos = jnp.sin(rotations[1]), jnp.cos(rotations[1])
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
    sin, cos = jnp.sin(rotations[2]), jnp.cos(rotations[2])
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
    rotations: jnp.ndarray,
) -> jnp.ndarray:
    """Return rotation matrix given radians.

    Rotation is anti-clockwise.

    Args:
        rotations: correspond to rotate around each axis.

    Returns:
        Rotation matrix of shape (n+1, n+1).

    Raises:
        ValueError: if not 2D or 3D.
    """
    if rotations.size == 1:
        return get_2d_rotation_matrix(rotations)
    if rotations.size == 3:
        return get_3d_rotation_matrix(rotations)
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


def get_2d_shear_matrix(
    shears: jnp.ndarray,
) -> jnp.ndarray:
    """Return 2D shear matrix.

    For example, the 2D shear matrix for x-axis applies as following
        [x, = [[1 s 0]  * [x,
         y,    [0 1 0]     y,
         1]    [0 0 1]]    1]
    where s is shear in x direction
    such that x = x + sy

    https://www.mauriciopoppe.com/notes/computer-graphics/transformation-matrices/shearing/

    Args:
        shears: radians, correspond to each plane shear, (n*(n-1),).

    Returns:
        Affine matrix of shape (n+1, n+1).
    """
    tans = jnp.tan(shears)
    shear_x = jnp.array(
        [
            [1.0, tans[0], 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    shear_y = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [tans[1], 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return jnp.matmul(shear_y, shear_x)


def get_3d_shear_matrix(
    shears: jnp.ndarray,
) -> jnp.ndarray:
    """Return 3D shear matrix.

    For example, the 3D shear matrix for xy plane applies as following
        [x, = [[1 0 s 0]  * [x,
         y,    [0 1 t 0]     y,
         z,    [0 0 1 0]     z,
         1]    [0 0 0 1]]    1]
    where s is shear in x direction, t is shear in y direction,
    such that x = x + sz, y = y + tz.

    https://www.mauriciopoppe.com/notes/computer-graphics/transformation-matrices/shearing/

    Args:
        shears: radians, correspond to each plane shear, (n*(n-1),).

    Returns:
        Affine matrix of shape (n+1, n+1).
    """
    tans = jnp.tan(shears)
    shear_xy = jnp.array(
        [
            [1.0, 0.0, tans[0], 0.0],
            [0.0, 1.0, tans[1], 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    shear_xz = jnp.array(
        [
            [1.0, tans[2], 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, tans[3], 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    shear_yz = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [tans[4], 1.0, 0.0, 0.0],
            [tans[5], 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return jnp.matmul(shear_yz, jnp.matmul(shear_xz, shear_xy))


def get_shear_matrix(
    shears: jnp.ndarray,
) -> jnp.ndarray:
    """Return rotation matrix given radians.

    Args:
        shears: correspond to shearing per axis/plane.

    Returns:
        Shear matrix of shape (n+1, n+1).

    Raises:
        ValueError: if not 2D or 3D.
    """
    if shears.size == 2:
        return get_2d_shear_matrix(shears)
    if shears.size == 6:
        return get_3d_shear_matrix(shears)
    raise ValueError("Only support 2D/3D rotations.")


def get_affine_matrix(
    spacing: jnp.ndarray,
    rotations: jnp.ndarray,
    scales: jnp.ndarray,
    shears: jnp.ndarray,
    shifts: jnp.ndarray,
) -> jnp.ndarray:
    """Return an affine matrix from parameters.

    The matrix is not squared, as the last row is not needed. For rotation,
    translation, and scaling matrix, they are kept for composition purpose.
    For example, the 3D affine transformation applies as following:
        [x, = [[* * * *]  * [x,
         y,    [* * * *]     y,
         z,    [* * * *]     z,
         1]    [0 0 0 1]]    1]

    As the spacing may not be uniform, we first scale to isotropic spacing,
    then apply rotation, scale, shear, and shift, at last scale back to original spacing.

    Args:
        spacing: correspond to each axis spacing, (n,).
        rotations: correspond to rotate around each axis, (1,) for 2d and (3,) for 3d
        scales: correspond to each axis scaling, (n,).
        shears: correspond to each plane shear, (n*(n-1),).
        shifts: correspond to each axis shift, (n,).

    Returns:
        Affine matrix of shape (n+1, n+1).
    """
    affine_to_iso = get_scaling_matrix(spacing)
    affine_rot = get_rotation_matrix(rotations)
    affine_scale = get_scaling_matrix(scales)
    affine_shear = get_shear_matrix(shears)
    affine_shift = get_translation_matrix(shifts)
    affine_from_iso = get_scaling_matrix(1.0 / spacing)

    affine = jnp.matmul(affine_rot, affine_to_iso)
    affine = jnp.matmul(affine_scale, affine)
    affine = jnp.matmul(affine_shear, affine)
    affine = jnp.matmul(affine_shift, affine)
    affine = jnp.matmul(affine_from_iso, affine)
    return affine


def batch_get_random_affine_matrix(
    key: jax.Array,
    spacing: jnp.ndarray,
    max_rotation: jnp.ndarray,
    max_zoom: jnp.ndarray,
    max_shear: jnp.ndarray,
    max_shift: jnp.ndarray,
    p: float = 0.5,
) -> jnp.ndarray:
    """Get a batch of random affine matrices.

    Args:
        key: jax random key.
        spacing: correspond to each axis spacing, (n,).
        max_rotation: maximum rotation in radians, (1,) for 2d and (3,) for 3d.
        max_zoom: maximum zoom in pixel/voxels, (num_spatial_dims,).
        max_shear: maximum shear in radians, (num_spatial_dims*(num_spatial_dims-1),).
        max_shift: maximum shift in pixel/voxels, (num_spatial_dims,).
        p: probability to activate each transformation independently.

    Returns:
        Affine matrix of shape (batch, n+1, n+1), n is num_spatial_dims.
    """
    ket_transform, key_act = jax.random.split(key, num=2)
    key_rot, key_zoom, key_shear, key_shift = jax.random.split(ket_transform, num=4)
    key_rot_act, key_zoom_act, key_shear_act, key_shift_act = jax.random.split(key_act, num=4)

    rotations = jax.random.uniform(
        key=key_rot,
        shape=max_rotation.shape,
        minval=-max_rotation,
        maxval=max_rotation,
    )
    rotations = jnp.where(
        jax.random.uniform(key=key_rot_act, shape=rotations.shape) < p,
        rotations,
        jnp.zeros_like(rotations),
    )

    scales = jax.random.uniform(
        key=key_zoom,
        shape=max_zoom.shape,
        minval=1.0 - max_zoom,
        maxval=1.0 + max_zoom,
    )
    scales = jnp.where(
        jax.random.uniform(key=key_zoom_act, shape=scales.shape) < p,
        scales,
        jnp.ones_like(scales),
    )

    shears = jax.random.uniform(
        key=key_shear,
        shape=max_shear.shape,
        minval=-max_shear,
        maxval=max_shear,
    )
    shears = jnp.where(
        jax.random.uniform(key=key_shear_act, shape=shears.shape) < p,
        shears,
        jnp.zeros_like(shears),
    )

    shifts = jax.random.uniform(
        key=key_shift,
        shape=max_shift.shape,
        minval=-max_shift,
        maxval=max_shift,
    )
    shifts = jnp.where(
        jax.random.uniform(key=key_shift_act, shape=shifts.shape) < p,
        shifts,
        jnp.zeros_like(shifts),
    )

    # vmap on first axis, which is a batch
    return jax.vmap(get_affine_matrix, in_axes=(None, 0, 0, 0, 0))(
        spacing, rotations, scales, shears, shifts
    )


def apply_affine_to_grid(grid: jnp.ndarray, affine_matrix: jnp.ndarray) -> jnp.ndarray:
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
    extended_grid = jnp.concatenate([grid, jnp.ones((1,) + grid.shape[1:])], axis=0)

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


def batch_apply_affine_to_grid(grid: jnp.ndarray, affine_matrix: jnp.ndarray) -> jnp.ndarray:
    """Apply batch of affine matrix to grid.

    Args:
        grid: grid coordinates, of shape  (n, d1, ..., dn).
            grid[:, i1, ..., in] = [i1, ..., in]
        affine_matrix: shape (batch, n+1, n+1).

    Returns:
        Grid with updated coordinates, shape (batch, n, d1, ..., dn).
    """
    return jax.vmap(apply_affine_to_grid, in_axes=(None, 0))(grid, affine_matrix)


def batch_random_affine_transform(
    key: jax.Array,
    batch: dict[str, jnp.ndarray],
    grid: jnp.ndarray,
    spacing: jnp.ndarray,
    max_rotation: jnp.ndarray,
    max_zoom: jnp.ndarray,
    max_shear: jnp.ndarray,
    max_shift: jnp.ndarray,
    p: float = 0.5,
) -> dict[str, jnp.ndarray]:
    """Perform random affine transformation on image and label.

    Args:
        key: jax random key.
        batch: dict having images or labels, or foreground_range.
            images have shape (batch, d1, ..., dn) or (batch, d1, ..., dn, c)
            labels have shape (batch, d1, ..., dn)
            batch should not have other keys such as UID.
            if foreground_range exists, it's pre-calculated based on label, it's
            pre-calculated because nonzero function is not jittable.
        grid: grid coordinates, of shape  (n, d1, ..., dn).
            grid[:, i1, ..., in] = [i1, ..., in]
        spacing: correspond to each axis spacing, (n,).
        max_rotation: maximum rotation in radians, (1,) for 2d and (3,) for 3d.
        max_zoom: maximum scaling difference in pixel/voxels, (n,).
        max_shear: maximum shear in radians, (n*(n-1),).
        max_shift: maximum translation in pixel/voxels, (n,).
        p: probability to activate each transformation independently.

    Returns:
        Augmented dict having image and label, shapes are not changed.
    """
    batch_size = get_batch_size(batch)

    # (batch, ...)
    max_rotation = jnp.tile(max_rotation[None, ...], (batch_size, 1))
    max_zoom = jnp.tile(max_zoom[None, ...], (batch_size, 1))
    max_shear = jnp.tile(max_shear[None, ...], (batch_size, 1))
    max_shift = jnp.tile(max_shift[None, ...], (batch_size, 1))

    # (batch, n+1, n+1)
    affine_matrix = batch_get_random_affine_matrix(
        spacing=spacing,
        key=key,
        max_rotation=max_rotation,
        max_zoom=max_zoom,
        max_shear=max_shear,
        max_shift=max_shift,
        p=p,
    )

    # (batch, n, d1, ..., dn)
    grid = batch_apply_affine_to_grid(grid=grid, affine_matrix=affine_matrix)

    resampled_batch = {}
    for k, v in batch.items():
        if LABEL in k:
            # assume label related keys have label in name
            resampled_batch[k] = batch_grid_sample(x=v, grid=grid, order=0)
        elif IMAGE in k:
            # assume image related keys have image in name
            resampled_batch[k] = batch_grid_sample(x=v, grid=grid, order=1)
        elif k == FOREGROUND_RANGE:
            # not needed anymore
            continue
        else:
            raise ValueError(f"Unknown key {k} in batch.")
    return resampled_batch


def get_random_affine_augmentation_fn(config: DictConfig) -> AugmentationFn:
    """Return a data augmentation function for random affine transformation.

    Args:
        config: entire config.

    Returns:
        A data augmentation function.
    """
    dataset_info = INFO_MAP[config.data.name]
    image_shape = dataset_info.image_spatial_shape
    n = len(image_shape)
    grid = get_coordinate_grid(shape=image_shape)
    da_config = config.data.loader.data_augmentation

    max_rotation = np.deg2rad([da_config.max_rotation] * (n * (n - 1) // 2))
    max_zoom = jnp.array([da_config.max_zoom] * n)
    max_shear = jnp.array([np.deg2rad(da_config.max_shear)] * n * (n - 1))
    max_shift = jnp.array([da_config.max_shift * x for x in image_shape])

    return partial(
        batch_random_affine_transform,
        grid=grid,
        spacing=jnp.array(dataset_info.image_spacing),
        max_rotation=max_rotation,
        max_zoom=max_zoom,
        max_shear=max_shear,
        max_shift=max_shift,
        p=da_config.p,
    )
