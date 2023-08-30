"""Metric centroid distance."""
from __future__ import annotations

import jax.numpy as jnp


def get_coordinate_grid(shape: tuple[int, ...]) -> jnp.ndarray:
    """Generate a grid with given shape.

    This function is not jittable as the output depends on the value of shapes.

    Args:
        shape: shape of the grid, (d1, ..., dn).

    Returns:
        grid: grid coordinates, of shape  (n, d1, ..., dn).
            grid[:, i1, ..., in] = [i1, ..., in]
    """
    return jnp.stack(
        jnp.meshgrid(
            *(jnp.arange(d) for d in shape),
            indexing="ij",
        ),
        axis=0,
        dtype=jnp.float32,
    )


def get_centroid(
    mask: jnp.ndarray,
    grid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the centroid of the mask.

    Args:
        mask: boolean mask of shape = (batch, d1, ..., dn, num_classes)
        grid: shape = (n, d1, ..., dn)

    Returns:
        centroid of shape = (batch, n, num_classes).
        nan mask of shape = (batch, num_classes).
    """
    mask_reduce_axes = tuple(range(1, mask.ndim - 1))
    grid_reduce_axes = tuple(range(2, mask.ndim))
    # (batch, n, d1, ..., dn, num_classes)
    masked_grid = jnp.expand_dims(mask, axis=1) * jnp.expand_dims(
        grid, axis=(0, -1)
    )
    # (batch, n, num_classes)
    numerator = jnp.sum(masked_grid, axis=grid_reduce_axes)
    # (batch, num_classes)
    summed_mask = jnp.sum(mask, axis=mask_reduce_axes)
    # (batch, 1, num_classes)
    denominator = summed_mask[:, None, :]
    # if mask is not empty return real centroid, else nan
    centroid = jnp.where(
        condition=denominator > 0, x=numerator / denominator, y=jnp.nan
    )
    return centroid, summed_mask == 0


def centroid_distance(
    mask_true: jnp.ndarray,
    mask_pred: jnp.ndarray,
    grid: jnp.ndarray,
    spacing: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Calculate the L2-distance between two centroids.

    Args:
        mask_true: shape = (batch, d1, ..., dn, num_classes).
        mask_pred: shape = (batch, d1, ..., dn, num_classes).
        grid: shape = (n, d1, ..., dn).
        spacing: spacing of pixel/voxels along each dimension, (n,).

    Returns:
        distance, shape = (batch, num_classes).
    """
    # centroid (batch, n, num_classes) nan_mask (batch, num_classes)
    centroid_true, nan_mask_true = get_centroid(
        mask=mask_true,
        grid=grid,
    )
    centroid_pred, nan_mask_pred = get_centroid(
        mask=mask_pred,
        grid=grid,
    )
    nan_mask = nan_mask_true | nan_mask_pred
    if spacing is not None:
        centroid_true = jnp.where(
            condition=nan_mask[:, None, :],
            x=jnp.nan,
            y=centroid_true * spacing[None, :, None],
        )
        centroid_pred = jnp.where(
            condition=nan_mask[:, None, :],
            x=jnp.nan,
            y=centroid_pred * spacing[None, :, None],
        )

    # return nan if the centroid cannot be defined for one sample with one class
    return jnp.where(
        condition=nan_mask,
        x=jnp.nan,
        y=jnp.linalg.norm(centroid_true - centroid_pred, axis=1),
    )
