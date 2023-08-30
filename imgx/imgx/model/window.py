"""Functions for windowing images."""
from __future__ import annotations

import jax.lax
import numpy as np
from jax import numpy as jnp


def window_partition(
    x: jnp.ndarray, window_shape: tuple[int, ...]
) -> jnp.ndarray:
    """Split image into windows with padding if needed.

    Each window will be processed independently and
    the "sequence" is the patches inside the window.

    If the spatial shape is not divisible by the window shape,
    the input will be padded to fit the window shape.

    Args:
        x: of shape (batch, *spatial_shape, channel).
        window_shape: tuple with same length as spatial_shape.

    Returns:
        windows: (batch, num_windows, window_volume, channel)
    """
    num_space_dims = len(window_shape)
    batch, *spatial_shape, channel = x.shape
    if spatial_shape == window_shape:
        # no need to partition
        return jnp.reshape(x, (batch, 1, np.prod(window_shape), channel))

    pad_shape: tuple[int, ...] = ()
    split_shape: tuple[int, ...] = ()
    for ss, ws in zip(spatial_shape, window_shape):
        pad = (ws - ss % ws) % ws
        pad_shape += (pad,)
        split_shape += ((ss + pad) // ws, ws)
    # pad to fit window shape
    x = jnp.pad(x, pad_width=((0, 0), *((0, p) for p in pad_shape), (0, 0)))
    # split into windows
    x = jnp.reshape(x, (batch, *split_shape, channel))
    # e.g. (batch, height // wh, wh, width // ww, ww, channel)
    # -> (batch, height // wh, width // ww, wh, ww, channel)
    axes = (
        0,
        *tuple(i * 2 + 1 for i in range(num_space_dims)),
        *tuple(i * 2 + 2 for i in range(num_space_dims)),
        num_space_dims * 2 + 1,
    )
    x = jnp.transpose(x, axes)
    # e.g. (batch, height // wh, width // ww, wh, ww, channel)
    # -> (batch, height // wh * width // ww, wh * ww, channel)
    x = jnp.reshape(x, (batch, -1, np.prod(window_shape), channel))
    return x


def window_unpartition(
    x: jnp.ndarray,
    window_shape: tuple[int, ...],
    spatial_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Reverse window partition.

    Input may have been padded to fit the window shape.

    Args:
        x: (batch, num_windows, prod(window_shape), channel).
        window_shape: tuple with same length as spatial_shape.
        spatial_shape: original shape.

    Returns:
        image: (batch, *spatial_shape, channel).
    """
    batch = x.shape[0]
    channel = x.shape[-1]

    if spatial_shape == window_shape:
        # no need to unpartition
        return jnp.reshape(x, (batch, *spatial_shape, channel))

    num_space_dims = len(window_shape)
    # e.g. (batch, height // wh, width // ww, wh, ww, channel)
    padded_shape = tuple(
        ss + (ws - ss % ws) % ws for ss, ws in zip(spatial_shape, window_shape)
    )
    split_shape = (
        tuple(ss // ws for ss, ws in zip(padded_shape, window_shape))
        + window_shape
    )
    x = jnp.reshape(x, (batch, *split_shape, channel))

    # e.g. (batch, height // wh, width // ww, wh, ww, channel)
    # -> (batch, height // wh, wh, width // ww, ww, channel)
    axes: tuple[int, ...] = (0,)
    for i in range(num_space_dims):
        axes += (i + 1, i + 1 + num_space_dims)
    axes += (num_space_dims * 2 + 1,)
    x = jnp.transpose(x, axes)
    # e.g. (batch, height // wh, wh, width // ww, ww, channel)
    # -> (batch, height, width, channel)
    x = jnp.reshape(x, (batch, *padded_shape, channel))

    # remove padding
    num_spatial_dims = len(spatial_shape)
    x = jax.lax.dynamic_slice(
        x,
        start_indices=(0,) * (num_spatial_dims + 2),
        slice_sizes=(x.shape[0], *spatial_shape, x.shape[-1]),
    )

    return x


def get_window_mask_index(
    spatial_shape: tuple[int, ...],
    window_shape: tuple[int, ...],
    shift_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Assign index to separate regions.

    Corresponds to figure 4 in https://arxiv.org/abs/2103.14030.

    Take 2D as an example,
        spatial_shape = (6, 6)
        window_shape = (4, 3)
        shift_shape = (1, 2)

    The region is divided into 9 regions
        [[0, 1, 2],
         [3, 4, 5],
         [6, 7, 8]]
    - 0 is unaffected area
    - 4, 5, 7, 8 is the bottom right corner, corresponds to a window.
    - 6, 7, 8 is shifted area along the first axis.
    - 2, 5, 8 is shifted area along the second axis.

    The output should be:
        [[0, 0, 0, 1, 2, 2],
         [0, 0, 0, 1, 2, 2],
         [3, 3, 3, 4, 5, 5],
         [3, 3, 3, 4, 5, 5],
         [3, 3, 3, 4, 5, 5],
         [6, 6, 6, 7, 8, 8]]

    Args:
        spatial_shape: 2 or 3d or more.
        window_shape: same shape as spatial_shape.
        shift_shape: same shape as spatial_shape, all values should be positive.

    Returns:
        An int array of size spatial_shape.
    """
    coords = []
    for sp, win, sf in zip(spatial_shape, window_shape, shift_shape):
        coords.append(
            jnp.concatenate(
                [
                    jnp.zeros((sp - win,)),
                    jnp.ones((win - sf,)),
                    jnp.ones((sf,)) * 2,
                ]
            ).astype(jnp.int32)
        )
    return jnp.ravel_multi_index(
        tuple(jnp.meshgrid(*coords, indexing="ij")),
        dims=(3,) * len(spatial_shape),
        mode="clip",
    )


def get_window_mask(
    spatial_shape: tuple[int, ...],
    window_shape: tuple[int, ...],
    shift_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Get mask for window attention.

    Args:
        spatial_shape: spatial shape for image.
        window_shape: window shape, same size as spatial_shape.
        shift_shape: same size as spatial_shape, all values should be positive.

    Returns:
        A boolean array of size (num_windows, window_volume, window_volume).
        False means the attention should be masked.
    """
    # (*spatial_shape)
    window_index = get_window_mask_index(
        spatial_shape, window_shape, shift_shape
    )
    # (1, num_windows, window_volume, 1)
    window_index = window_partition(
        x=window_index[None, ..., None], window_shape=window_shape
    )
    # (num_windows, window_volume)
    window_index = window_index[0, :, :, 0]
    # (num_windows, window_volume, window_volume)
    attn_mask = window_index[:, :, None] - window_index[:, None, :]
    attn_mask = attn_mask == 0
    return attn_mask


def get_window_shift_pad_shapes(
    spatial_shape: tuple[int, ...],
    window_shape: tuple[int, ...],
    shift_shape: tuple[int, ...],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Get shapes for padding, window, and shifting.

    For an axis, if the window size is larger than spatial size,
        - window is shrunk to the spatial size to avoid unnecessary padding
        - shift is removed as there is one window only.

    Args:
        spatial_shape: spatial shape for image.
        window_shape: window shape, same size as spatial_shape.
        shift_shape: same size as spatial_shape, all values should be positive.

    Returns:
        - window_shape
        - spatial_padding
        - padded_spatial_shape
        - shift_shape
        - neg_shift_shape
    """
    # ensure window shape no larger than spatial shape
    window_shape = tuple(
        min(ss, ws) for ss, ws in zip(spatial_shape, window_shape)
    )

    # pad to ensure the shape can be evenly divided by window
    spatial_padding = tuple(
        (ws - ss % ws) % ws for ss, ws in zip(spatial_shape, window_shape)
    )
    padded_spatial_shape = tuple(
        ss + sp for ss, sp in zip(spatial_shape, spatial_padding)
    )

    # if there are at least two windows, shift shape remain unchanged
    # if there is only one window, shift shape is zero
    shift_shape = tuple(
        min(ss, pss - ws)
        for ss, pss, ws in zip(shift_shape, padded_spatial_shape, window_shape)
    )
    neg_shift_shape = tuple(-ss for ss in shift_shape)

    return (
        window_shape,
        spatial_padding,
        padded_spatial_shape,
        shift_shape,
        neg_shift_shape,
    )
