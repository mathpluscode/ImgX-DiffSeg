"""Preprocess functions using sitk."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813

from imgx_datasets.util import get_center_crop_shape_from_bbox, get_center_pad_shape


def check_image_and_label(
    image_volume: sitk.Image,
    label_volume: sitk.Image,
    image_path: Path,
    label_path: Path,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-3,
) -> None:
    """Check if metadata matches between image and label.

    Args:
        image_volume: loaded image.
        label_volume: loaded label.
        image_path: image file path.
        label_path: label file path.
        rtol: relative tolerance for sanity check, 1E-5 is too big.
        atol: absolute tolerance for sanity check, 1E-8 is too big.

    Raises:
        ValueError: if image and label metadata does not match
    """
    if image_volume.GetSize() != label_volume.GetSize():
        raise ValueError(
            f"Image and label sizes are not the same for "
            f"{image_path} and {label_path}: "
            f"{image_volume.GetSize()} and {label_volume.GetSize()}."
        )
    if not np.allclose(
        image_volume.GetSpacing(),
        label_volume.GetSpacing(),
        rtol=rtol,
        atol=atol,
    ):
        raise ValueError(
            f"Image and label spacing are not the same for "
            f"{image_path} and {label_path}: "
            f"{image_volume.GetSpacing()} and {label_volume.GetSpacing()}."
        )
    if not np.allclose(
        image_volume.GetDirection(),
        label_volume.GetDirection(),
        rtol=rtol,
        atol=atol,
    ):
        arr_image = np.array(image_volume.GetDirection())
        arr_label = np.array(label_volume.GetDirection())
        raise ValueError(
            f"Image and label direction are not the same for "
            f"{image_path} and {label_path}: "
            f"{arr_image} and {arr_label}, "
            f"difference is {arr_image - arr_label} for "
            f"rtol={rtol} and atol = {atol}."
        )
    if not np.allclose(image_volume.GetOrigin(), label_volume.GetOrigin(), rtol=rtol, atol=atol):
        raise ValueError(
            f"Image and label origin are not the same for "
            f"{image_path} and {label_path}: "
            f"{image_volume.GetOrigin()} and {label_volume.GetOrigin()}."
        )


def resample(volume: sitk.Image, is_label: bool, target_spacing: tuple[float, ...]) -> sitk.Image:
    """Resample volume to the target spacing.

    Args:
        volume: volume to resample.
        is_label: True if it represents a label,
            thus nearest neighbour for interpolation.
        target_spacing: target dimension per axis.

    Returns:
        Resampled volume.
    """
    original_spacing = volume.GetSpacing()
    original_shape = volume.GetSize()
    original_center = volume.GetOrigin()

    # calculate shape after resampling
    # round to integers to be robust
    # otherwise, ceiling is sensitive to spacing
    resample_target_shape = tuple(
        int(np.round(orig_sh * orig_sp / trg_sp))
        for orig_sh, orig_sp, trg_sp in zip(original_shape, original_spacing, target_spacing)
    )
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear

    # No transform because we do not want to change the represented
    # physical size of the objects in the image
    transform = sitk.Transform()
    # The origin is middle of the voxel/pixel
    # https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html
    target_center = [
        x + 0.5 * (target_spacing[i] - original_spacing[i]) for i, x in enumerate(original_center)
    ]
    # Do not rotate
    target_direction = volume.GetDirection()
    volume = sitk.Resample(
        volume,
        size=resample_target_shape,
        transform=transform,
        interpolator=interpolator,
        outputOrigin=target_center,
        outputSpacing=target_spacing,
        outputDirection=target_direction,
        defaultPixelValue=0,
        outputPixelType=volume.GetPixelID(),
        useNearestNeighborExtrapolator=False,
    )
    return volume


def get_invalid_bounding_box(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all -1 values as an invalid bounding box.

    Args:
        mask: boolean mask, with n spatial axes.

    Returns:
        - bbox_min, [-1] * n.
        - bbox_min, [-1] * n.
    """
    ndim_spatial = len(mask.shape)
    bbox_min = -np.ones(ndim_spatial, np.int32)
    bbox_max = -np.ones(ndim_spatial, np.int32)
    return bbox_min, bbox_max


def get_valid_binary_mask_bounding_box(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the bounding box of foreground with start-end positions.

    If there is no foreground, return -1 for all outputs.

    Args:
        mask: boolean mask, with only spatial axes.

    Returns:
        - bbox_min, [start_in_1st_spatial_dim, ...], inclusive, starts at zero.
        - bbox_max, [end_in_1st_spatial_dim, ...], exclusive, starts at zero.
    """
    ndim_spatial = len(mask.shape)
    bbox_min = []
    bbox_max = []
    for axes_to_reduce in combinations(reversed(range(ndim_spatial)), ndim_spatial - 1):
        mask_reduced = np.amax(mask, axis=axes_to_reduce)
        bbox_min_axis = np.argmax(mask_reduced)
        bbox_max_axis = mask_reduced.shape[0] - np.argmax(np.flip(mask_reduced))
        bbox_min.append(bbox_min_axis)
        bbox_max.append(bbox_max_axis)
    return np.stack(bbox_min), np.stack(bbox_max)


def get_binary_mask_bounding_box(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the bounding box of foreground with start-end positions.

    If there is no foreground, return -1 for all outputs.

    Args:
        mask: boolean mask, with only spatial axes.

    Returns:
        - bbox_min, [start_in_1st_spatial_dim, ...], inclusive, starts at zero.
        - bbox_max, [end_in_1st_spatial_dim, ...], exclusive, starts at zero.
    """
    if mask.dtype != np.bool_:
        mask = mask > 0

    if np.any(mask):
        return get_valid_binary_mask_bounding_box(mask)
    return get_invalid_bounding_box(mask)


def clip_and_normalise_intensity(
    image_volume: sitk.Image,
    intensity_range: tuple[float, float] | None,
) -> sitk.Image:
    """Clip and normalise the intensity of the image volume.

    Args:
        image_volume: image volume to clip and normalise.
        intensity_range: intensity range to clip to.
            If None, clip to 0.95 and 99.5 percentiles.

    Returns:
        Image volume with clipped and normalised intensity.
    """
    # clip intensity
    if intensity_range is None:
        # if not configured, clip to 0.95 and 99.5 percentiles
        # https://arxiv.org/abs/2304.12306
        image_array = sitk.GetArrayFromImage(image_volume)
        intensity_range = (
            np.percentile(image_array, 0.95),
            np.percentile(image_array, 99.5),
        )
    image_volume = sitk.Clamp(
        image_volume,
        lowerBound=intensity_range[0],
        upperBound=intensity_range[1],
    )

    # normalise the intensity
    image_volume = sitk.Normalize(image_volume)
    image_volume = sitk.RescaleIntensity(image_volume, outputMinimum=0, outputMaximum=1)
    return image_volume


def n4_bias_field_correction(
    image_volume: sitk.Image,
    shrink_factor: int,
) -> sitk.Image:
    """Perform N4 bias field correction on an image.

    https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html

    Args:
        image_volume: image to correct.
        shrink_factor: downsample factor for the image.
    """
    image_volume = sitk.Cast(image_volume, sitk.sitkFloat32)
    input_image_volume = image_volume
    mask_volume = sitk.OtsuThreshold(image_volume, 0, 1, 200)
    mask_volume = sitk.Cast(mask_volume, sitk.sitkUInt8)

    if shrink_factor > 1:
        image_volume = sitk.Shrink(image_volume, [shrink_factor] * image_volume.GetDimension())
        mask_volume = sitk.Shrink(mask_volume, [shrink_factor] * image_volume.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(image_volume, mask_volume)

    if shrink_factor > 1:
        log_bias_field = corrector.GetLogBiasFieldAsImage(input_image_volume)
        corrected_image = input_image_volume / sitk.Exp(log_bias_field)

    return corrected_image


def load_and_preprocess_image_and_label(
    uid: str,
    image_path: Path,
    label_path: Path,
    out_dir: Path,
    target_spacing: tuple[float, ...],
    target_shape: tuple[int, ...],
    intensity_range: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load image and perform resampling/padding/cropping using SimpleITK.

    This function also saves the processed image/masks.

    https://examples.itk.org/src/filtering/imagegrid/resampleanimage/documentation

    Args:
        uid: unique id for the image.
        image_path: file path of Nifti file for image.
        label_path: file path of Nifti file for image.
        out_dir: directory to save preprocessed files.
        target_spacing: size of each voxel, of shape (dx, dy, dz),
        target_shape: size of image, of shape (width, height, depth).
        intensity_range: image intensity range before normalisation.

    Returns:
        image array, of shape (width, height, depth), without channel.
        label array, of shape (width, height, depth).
    """
    if len(target_spacing) != 3:
        raise ValueError(f"Target spacing should have 3 elements, got {target_spacing}.")
    if len(target_shape) != 3:
        raise ValueError(f"Target shape should have 3 elements, got {target_shape}.")

    # load
    image_volume = sitk.ReadImage(str(image_path))
    label_volume = sitk.ReadImage(str(label_path))

    # metadata should be the same
    check_image_and_label(
        image_volume=image_volume,
        label_volume=label_volume,
        image_path=image_path,
        label_path=label_path,
    )

    # resample
    image_volume = resample(volume=image_volume, is_label=False, target_spacing=target_spacing)
    label_volume = resample(volume=label_volume, is_label=True, target_spacing=target_spacing)
    if image_volume.GetSize() != label_volume.GetSize():
        raise ValueError(
            f"After resampling image and label does not match: "
            f"image = {image_volume.GetSize()} "
            f"label = {label_volume.GetSize()}."
        )

    # clip and normalise intensity
    image_volume = clip_and_normalise_intensity(
        image_volume=image_volume,
        intensity_range=intensity_range,
    )

    # cast dtype
    image_volume = sitk.Cast(image_volume, sitk.sitkFloat32)
    label_volume = sitk.Cast(label_volume, sitk.sitkUInt8)

    # pad if the size is smaller than target
    # center-padding can be calculated on image or label
    pad_lower, pad_upper = get_center_pad_shape(
        current_shape=label_volume.GetSize(), target_shape=target_shape
    )
    image_volume = sitk.ConstantPad(image_volume, pad_lower, pad_upper, 0)
    label_volume = sitk.ConstantPad(label_volume, pad_lower, pad_upper, 0)

    # crop if the size is larger than target
    # crop is calculated on labels to ensure maximum area of labels
    label_array = sitk.GetArrayFromImage(label_volume)
    label_array = np.transpose(label_array, axes=[2, 1, 0])
    bbox_min, bbox_max = get_binary_mask_bounding_box(mask=label_array)
    crop_lower, crop_upper = get_center_crop_shape_from_bbox(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        current_shape=label_volume.GetSize(),
        target_shape=target_shape,
    )
    image_volume = sitk.Crop(image_volume, crop_lower, crop_upper)
    label_volume = sitk.Crop(label_volume, crop_lower, crop_upper)

    # check shape
    if image_volume.GetSize() != target_shape:
        raise ValueError(
            f"After resampling/padding/cropping, image shape "
            f"{image_volume.GetSize()} is wrong for "
            f"{image_path}"
        )
    if label_volume.GetSize() != target_shape:
        raise ValueError(
            f"After resampling/padding/cropping, label shape "
            f"{label_volume.GetSize()} is wrong for "
            f"{label_path}"
        )

    # save processed image/mask
    image_out_path = out_dir / (uid + "_img_preprocessed.nii.gz")
    label_out_path = out_dir / (uid + "_mask_preprocessed.nii.gz")
    sitk.WriteImage(image=image_volume, fileName=str(image_out_path), useCompression=True)
    sitk.WriteImage(image=label_volume, fileName=str(label_out_path), useCompression=True)

    # return array and switch axes
    image = np.transpose(sitk.GetArrayFromImage(image_volume), axes=[2, 1, 0])
    label = np.transpose(sitk.GetArrayFromImage(label_volume), axes=[2, 1, 0])
    return image, label
