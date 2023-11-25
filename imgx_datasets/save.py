"""IO related functions (file cannot be named as io).

https://stackoverflow.com/questions/26569828/pycharm-py-initialize-cant-initialize-sys-standard-streams
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk  # noqa: N813
from absl import logging
from PIL import Image


def save_uids(
    train_uids: list[str],
    valid_uids: list[str],
    test_uids: list[str],
    out_dir: Path,
) -> None:
    """Save uids to csv files.

    Args:
        train_uids: list of training uids.
        valid_uids: list of validation uids.
        test_uids: list of test uids.
        out_dir: directory to save the csv files.
    """
    pd.DataFrame({"uid": train_uids}).to_csv(out_dir / "train_uids.csv", index=False)
    pd.DataFrame({"uid": valid_uids}).to_csv(out_dir / "valid_uids.csv", index=False)
    pd.DataFrame({"uid": test_uids}).to_csv(out_dir / "test_uids.csv", index=False)
    logging.info(f"There are {len(train_uids)} training samples.")
    logging.info(f"There are {len(valid_uids)} validation samples.")
    logging.info(f"There are {len(test_uids)} test samples.")


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


def load_2d_grayscale_image(
    image_path: Path,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """Load 2d images.

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


def save_3d_image(
    image: np.ndarray,
    reference_image: sitk.Image,
    out_path: Path,
) -> None:
    """Save 3d image.

    Args:
        image: (depth, height, width), the values are integers.
        reference_image: reference image for copy meta data.
        out_path: output path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(image)
    image.CopyInformation(reference_image)
    # output
    sitk.WriteImage(
        image=image,
        fileName=out_path,
        useCompression=True,
    )


def save_image(
    image: np.ndarray,
    reference_image: sitk.Image,
    out_path: Path,
    dtype: np.dtype,
) -> None:
    """Save 2d or 3d image.

    Args:
        image: (width, height, depth) or (height, width), 3D is not reversed but 2D is reversed.
        reference_image: reference image for copy metadata.
        out_path: output path.
        dtype: data type of the output.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim not in [2, 3]:
        raise ValueError(
            f"Image should be 2D or 3D, but {image.ndim}D is given with shape {image.shape}."
        )
    if image.ndim == 2:
        save_2d_grayscale_image(
            image=image.astype(dtype=dtype),
            out_path=out_path,
        )
    else:
        # (width, height, depth) -> (depth, height, width)
        image = np.transpose(image, axes=[2, 1, 0]).astype(dtype=dtype)
        save_3d_image(
            image=image,
            reference_image=reference_image,
            out_path=out_path,
        )


def save_ddf(
    ddf: np.ndarray,
    reference_image: sitk.Image,
    out_path: Path,
    dtype: np.dtype = np.float64,
) -> None:
    """Save ddf for 3d volumes.

    Args:
        ddf: (width, height, depth, 3), unit is 1 without spacing.
        reference_image: reference image for copy metadata.
        out_path: output path.
        dtype: data type of the output.
    """
    if ddf.ndim != 4:
        raise ValueError(f"Mask should be 4D, but {ddf.ndim}D is given.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ddf is scaled by spacing
    ddf = np.transpose(ddf, axes=[2, 1, 0, 3]).astype(dtype=dtype)
    ddf *= np.expand_dims(reference_image.GetSpacing(), axis=list(range(ddf.ndim - 1)))

    ddf_volume = sitk.GetImageFromArray(ddf, isVector=True)
    ddf_volume.SetSpacing(reference_image.GetSpacing())
    ddf_volume.CopyInformation(reference_image)
    tx = sitk.DisplacementFieldTransform(ddf_volume)
    sitk.WriteTransform(tx, out_path)
