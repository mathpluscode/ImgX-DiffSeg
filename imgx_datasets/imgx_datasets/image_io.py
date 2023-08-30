"""Module for image io."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813
from PIL import Image


def save_segmentation_prediction(
    preds: np.ndarray,
    uids: list,
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
    if preds.ndim == 3:
        save_2d_segmentation_prediction(
            preds=preds,
            uids=uids,
            out_dir=out_dir,
        )
    elif preds.ndim == 4:
        save_3d_segmentation_prediction(
            preds=preds,
            uids=uids,
            out_dir=out_dir,
            tfds_dir=tfds_dir,
        )
    else:
        raise ValueError(
            f"Prediction should be 3D or 4D with num_samples axis, "
            f"but {preds.ndim}D is given."
        )


def save_2d_segmentation_prediction(
    preds: np.ndarray,
    uids: list,
    out_dir: Path,
) -> None:
    """Save segmentation predictions for 2d images.

    Args:
        preds: (num_samples, ...), the values are integers.
        uids: (num_samples,).
        out_dir: output directory.
    """
    if preds.ndim != 3:
        raise ValueError(
            f"Prediction should be 3D, but {preds.ndim}D is given."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, uid in enumerate(uids):
        mask_pred = preds[i, ...]
        if np.max(mask_pred) > 1:
            raise ValueError(
                f"Prediction values should be 0 or 1, but "
                f"max value is {np.max(mask_pred)} for {uid}. "
                f"Multi-class segmentation for 2D images are not supported."
            )
        save_2d_grayscale_image(
            image=mask_pred,
            out_path=out_dir / f"{uid}_mask_pred.png",
        )


def save_2d_grayscale_image(
    image: np.ndarray,
    out_path: Path,
) -> None:
    """Save grayscale 2d images.

    Args:
        image: (height, width), the values between [0, 1].
        out_path: output path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image * 255, dtype="uint8")
    Image.fromarray(image, "L").save(str(out_path))


def save_3d_segmentation_prediction(
    preds: np.ndarray,
    uids: list,
    out_dir: Path,
    tfds_dir: Path,
) -> None:
    """Save segmentation predictions for 3d volumes.

    Args:
        preds: (num_samples, width, height, depth), the values are integers.
        uids: (num_samples,).
        out_dir: output directory.
        tfds_dir: directory saving preprocessed images and labels.
    """
    if preds.ndim != 4:
        raise ValueError(
            f"Prediction should be 4D, but {preds.ndim}D is given."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, uid in enumerate(uids):
        # (width, height, depth) -> (depth, height, width)
        mask_pred = np.transpose(preds[i, ...], axes=[2, 1, 0])
        mask_pred = mask_pred.astype(dtype="uint8")
        save_3d_mask(
            mask=mask_pred,
            mask_true_path=tfds_dir / f"{uid}_mask_preprocessed.nii.gz",
            out_path=out_dir / f"{uid}_mask_pred.nii.gz",
        )


def save_3d_mask(
    mask: np.ndarray,
    mask_true_path: Path,
    out_path: Path,
) -> None:
    """Save segmentation predictions for 3d volumes.

    Args:
        mask: (depth, height, width), the values are integers.
        mask_true_path: path to the true mask.
        out_path: output path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    volume_mask = sitk.GetImageFromArray(mask)
    # copy meta data
    volume_mask_true = sitk.ReadImage(mask_true_path)
    volume_mask.CopyInformation(volume_mask_true)
    # output
    sitk.WriteImage(
        image=volume_mask,
        fileName=out_path,
        useCompression=True,
    )


def load_2d_grayscale_image(
    image_path: Path,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """Load 2d mask.

    Args:
        image_path: path to the mask.
        dtype: data type of the output.

    Returns:
        mask: (height, width), the values are between [0, 1].
    """
    mask = Image.open(str(image_path)).convert("L")  # value [0, 255]
    mask = np.asarray(mask) / 255  # value [0, 1]
    mask = np.asarray(mask, dtype=dtype)
    return mask


def load_3d_image(
    image_path: Path,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Load 3d images.

    Args:
        image_path: path to the mask.
        dtype: data type of the output.

    Returns:
        mask: (depth, height, width), the values are integers.
    """
    return np.asarray(
        sitk.GetArrayFromImage(sitk.ReadImage(image_path)), dtype=dtype
    )
