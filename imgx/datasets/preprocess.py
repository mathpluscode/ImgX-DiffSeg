"""Preprocess functions using sitk."""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import SimpleITK as sitk  # noqa: N813

from imgx.datasets.util import (
    get_center_crop_shape_from_bbox,
    get_center_pad_shape,
)
from imgx.metric.surface_distance import get_binary_mask_bounding_box


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
    if not np.allclose(
        image_volume.GetOrigin(), label_volume.GetOrigin(), rtol=rtol, atol=atol
    ):
        raise ValueError(
            f"Image and label origin are not the same for "
            f"{image_path} and {label_path}: "
            f"{image_volume.GetOrigin()} and {label_volume.GetOrigin()}."
        )


def resample(
    volume: sitk.Image, is_label: bool, target_spacing: Tuple[float, ...]
) -> sitk.Image:
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
        for orig_sh, orig_sp, trg_sp in zip(
            original_shape, original_spacing, target_spacing
        )
    )
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear

    # No transform because we do not want to change the represented
    # physical size of the objects in the image
    transform = sitk.Transform()
    # The origin is middle of the voxel/pixel
    # https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html
    target_center = [
        x + 0.5 * (target_spacing[i] - original_spacing[i])
        for i, x in enumerate(original_center)
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


def load_and_preprocess_image_and_label(
    uid: str,
    image_path: Path,
    label_path: Path,
    out_dir: Path,
    target_spacing: Tuple[float, float, float],
    target_shape: Tuple[int, int, int],
    intensity_range: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
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
        image array, of shape (width, height, depth).
        label array, of shape (width, height, depth).
    """
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
    image_volume = resample(
        volume=image_volume, is_label=False, target_spacing=target_spacing
    )
    label_volume = resample(
        volume=label_volume, is_label=True, target_spacing=target_spacing
    )
    if image_volume.GetSize() != label_volume.GetSize():
        raise ValueError(
            f"After resampling image and label does not match: "
            f"image = {image_volume.GetSize()} "
            f"label = {label_volume.GetSize()}."
        )

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

    # clip intensity if configured
    if intensity_range is not None:
        image_volume = sitk.Clamp(
            image_volume,
            lowerBound=intensity_range[0],
            upperBound=intensity_range[1],
        )

    # for image, normalise the intensity
    image_volume = sitk.Normalize(image_volume)
    image_volume = sitk.RescaleIntensity(
        image_volume, outputMinimum=0, outputMaximum=1
    )

    # cast dtype
    image_volume = sitk.Cast(image_volume, sitk.sitkFloat32)
    label_volume = sitk.Cast(label_volume, sitk.sitkUInt16)

    # save processed image/mask
    image_out_path = out_dir / (uid + "_img_preprocessed.nii.gz")
    label_out_path = out_dir / (uid + "_mask_preprocessed.nii.gz")
    sitk.WriteImage(
        image=image_volume, fileName=str(image_out_path), useCompression=True
    )
    sitk.WriteImage(
        image=label_volume, fileName=str(label_out_path), useCompression=True
    )

    # return array and switch axes
    image = np.transpose(sitk.GetArrayFromImage(image_volume), axes=[2, 1, 0])
    label = np.transpose(sitk.GetArrayFromImage(label_volume), axes=[2, 1, 0])
    return image, label


def save_segmentation_prediction(
    preds: np.ndarray,
    uids: List,
    out_dir: Path,
    tfds_dir: Path,
) -> None:
    """Save segmentation predictions.

    Args:
        preds: (num_samples, ...), the values are integers.
        uids: (num_samples,).
        out_dir: output directory.
        tfds_dir: directory saving preprocessed images and labels.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, uid in enumerate(uids):
        # transform np array to volume
        mask_pred = np.transpose(preds[i, ...], axes=[2, 1, 0]).astype(
            dtype="uint16"
        )
        volume_mask_pred = sitk.GetImageFromArray(mask_pred)

        # copy meta data
        volume_mask_true = sitk.ReadImage(
            tfds_dir / f"{uid}_mask_preprocessed.nii.gz"
        )
        volume_mask_pred.CopyInformation(volume_mask_true)

        # output
        sitk.WriteImage(
            image=volume_mask_pred,
            fileName=out_dir / f"{uid}_mask_pred.nii.gz",
            useCompression=True,
        )
