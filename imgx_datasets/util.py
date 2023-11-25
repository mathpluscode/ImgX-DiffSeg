"""Util functions for image.

Some are adapted from
https://github.com/google-research/scenic/blob/03735eb81f64fd1241c4efdb946ea6de3d326fe1/scenic/dataset_lib/dataset_utils.py
"""
from __future__ import annotations

import numpy as np


def get_center_pad_shape(
    current_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Get pad sizes for sitk.ConstantPad.

    The padding is added symmetrically.

    Args:
        current_shape: current shape of the image.
        target_shape: target shape of the image.

    Returns:
        pad_lower: shape to pad on the lower side.
        pad_upper: shape to pad on the upper side.
    """
    pad_lower = []
    pad_upper = []
    for i, size_i in enumerate(current_shape):
        pad_i = max(target_shape[i] - size_i, 0)
        pad_lower_i = pad_i // 2
        pad_upper_i = pad_i - pad_lower_i
        pad_lower.append(pad_lower_i)
        pad_upper.append(pad_upper_i)
    return tuple(pad_lower), tuple(pad_upper)


def get_center_crop_shape(
    current_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Get crop sizes for sitk.Crop.

    The crop is performed symmetrically.

    Args:
        current_shape: current shape of the image.
        target_shape: target shape of the image.

    Returns:
        crop_lower: shape to pad on the lower side.
        crop_upper: shape to pad on the upper side.
    """
    crop_lower = []
    crop_upper = []
    for i, size_i in enumerate(current_shape):
        crop_i = max(size_i - target_shape[i], 0)
        crop_lower_i = crop_i // 2
        crop_upper_i = crop_i - crop_lower_i
        crop_lower.append(crop_lower_i)
        crop_upper.append(crop_upper_i)
    return tuple(crop_lower), tuple(crop_upper)


def try_to_get_center_crop_shape(
    label_min: int, label_max: int, current_length: int, target_length: int
) -> tuple[int, int]:
    """Try to crop at the center of label, 1D.

    Args:
        label_min: label index minimum, inclusive.
        label_max: label index maximum, exclusive.
        current_length: current image length.
        target_length: target image length.

    Returns:
        crop_lower: shape to pad on the lower side.
        crop_upper: shape to pad on the upper side.

    Raises:
        ValueError: if label min max is out of range.
    """
    if label_min < 0 or label_max > current_length:
        raise ValueError("Label index out of range.")

    if current_length <= target_length:
        # no need of crop
        return 0, 0
    # attend to perform crop centered at label center
    label_center = (label_max - 1 + label_min) / 2.0
    bbox_lower = int(np.ceil(label_center - target_length / 2.0))
    bbox_upper = bbox_lower + target_length
    # if lower is negative, then have to shift the window to right
    bbox_lower = max(bbox_lower, 0)
    # if upper is too large, then have to shift the window to left
    if bbox_upper > current_length:
        bbox_lower -= bbox_upper - current_length
    # calculate crop
    crop_lower = bbox_lower  # bbox index starts at 0
    crop_upper = current_length - target_length - crop_lower
    return crop_lower, crop_upper


def get_center_crop_shape_from_bbox(
    bbox_min: tuple[int, ...] | np.ndarray,
    bbox_max: tuple[int, ...] | np.ndarray,
    current_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Get crop sizes for sitk.Crop from label bounding box.

    The crop is not necessarily performed symmetrically.

    Args:
        bbox_min: [start_in_1st_spatial_dim, ...], inclusive, starts at zero.
        bbox_max: [end_in_1st_spatial_dim, ...], exclusive, starts at zero.
        current_shape: current shape of the image.
        target_shape: target shape of the image.

    Returns:
        crop_lower: shape to crop on the lower side.
        crop_upper: shape to crop on the upper side.
    """
    crop_lower = []
    crop_upper = []
    for i, current_length in enumerate(current_shape):
        crop_lower_i, crop_upper_i = try_to_get_center_crop_shape(
            label_min=bbox_min[i],
            label_max=bbox_max[i],
            current_length=current_length,
            target_length=target_shape[i],
        )
        crop_lower.append(crop_lower_i)
        crop_upper.append(crop_upper_i)
    return tuple(crop_lower), tuple(crop_upper)
