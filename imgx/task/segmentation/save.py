"""Segmentation related io (file cannot be named as io).

https://stackoverflow.com/questions/26569828/pycharm-py-initialize-cant-initialize-sys-standard-streams
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813

from imgx.datasets.save import save_image


def save_segmentation_prediction(
    label_pred: np.ndarray,
    uids: list[str],
    out_dir: Path | None,
    tfds_dir: Path,
    reference_suffix: str = "mask_preprocessed",
    output_suffix: str = "mask_pred",
) -> None:
    """Save segmentation predictions.

    Args:
        label_pred: (num_samples, ...), the values are integers.
        uids: (num_samples,).
        out_dir: output directory.
        tfds_dir: directory saving preprocessed images and labels.
        reference_suffix: suffix of reference image.
        output_suffix: suffix of output image.
    """
    if out_dir is None:
        return
    if label_pred.ndim == 3 and np.max(label_pred) > 1:
        raise ValueError(
            f"Prediction values should be 0 or 1, but "
            f"max value is {np.max(label_pred)}. "
            f"Multi-class segmentation for 2D images are not supported."
        )
    if label_pred.ndim not in [3, 4]:
        raise ValueError(
            f"Prediction should be 3D or 4D with num_samples axis, but {label_pred.ndim}D is given."
        )
    file_suffix = "nii.gz" if label_pred.ndim == 4 else "png"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, uid in enumerate(uids):
        reference_image = sitk.ReadImage(tfds_dir / f"{uid}_{reference_suffix}.{file_suffix}")
        save_image(
            image=label_pred[i, ...],
            reference_image=reference_image,
            out_path=out_dir / f"{uid}_{output_suffix}.{file_suffix}",
            dtype=np.uint8,
        )
