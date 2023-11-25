"""Module for image/lavel warping."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax._src.scipy.ndimage import map_coordinates


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


def batch_grid_sample(
    x: jnp.ndarray,
    grid: jnp.ndarray,
    order: int,
    constant_values: float = 0.0,
) -> jnp.ndarray:
    """Apply sampling to input.

    https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

    Args:
        x: shape (batch, d1, ..., dn) or (batch, d1, ..., dn, c).
        grid: grid coordinates, of shape (batch, n, d1, ..., dn).
        order: interpolation order, 0 for nearest, 1 for linear.
        constant_values: constant value for out of bound coordinates.

    Returns:
        Same shape as x.
    """
    if x.ndim not in [grid.ndim - 1, grid.ndim]:
        raise ValueError(f"Input x has shape {x.shape}, grid has shape {grid.shape}.")

    # vmap on batch axis
    sample_vmap = jax.vmap(
        partial(
            map_coordinates,
            order=order,
            mode="constant",
            cval=constant_values,
        ),
        in_axes=(0, 0),
    )
    if x.ndim == grid.ndim:
        # vmap on channel axis
        ch_axis = x.ndim - 1
        sample_vmap = jax.vmap(
            sample_vmap,
            in_axes=(ch_axis, None),
            out_axes=ch_axis,
        )
    return sample_vmap(x, grid)


def warp_image(
    x: jnp.ndarray,
    ddf: jnp.ndarray,
    order: int,
) -> jnp.ndarray:
    """Warp the image with the deformation field.

    TODO: grid is a constant, can be precomputed.

    Args:
        x: shape (batch, d1, ..., dn) or (batch, d1, ..., dn, c).
        ddf: deformation field, of shape (batch, d1, ..., dn, n).
        order: interpolation order, 0 for nearest, 1 for linear.

    Returns:
        warped image, of shape (batch, d1, ..., dn) or (batch, d1, ..., dn, c).
    """
    # (batch, n, d1, ..., dn)
    grid = get_coordinate_grid(shape=ddf.shape[1:-1])
    grid += jnp.moveaxis(ddf, -1, 1)
    return batch_grid_sample(x, grid, order=order)
