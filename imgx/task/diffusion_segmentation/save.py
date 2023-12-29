"""Segmentation related io (file cannot be named as io).

https://stackoverflow.com/questions/26569828/pycharm-py-initialize-cant-initialize-sys-standard-streams
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from imgx.task.segmentation.save import save_segmentation_prediction


def save_diffusion_segmentation_prediction(
    label_pred: np.ndarray,
    uids: list[str],
    out_dir: Path | None,
    tfds_dir: Path,
    reference_suffix: str = "mask_preprocessed",
    output_suffix: str = "mask_pred",
) -> None:
    """Save segmentation predictions.

    Args:
        label_pred: (num_samples, ..., num_timesteps), the values are integers.
        uids: (num_samples,).
        out_dir: output directory.
        tfds_dir: directory saving preprocessed images and labels.
        reference_suffix: suffix of reference image.
        output_suffix: suffix of output image.
    """
    if out_dir is None:
        return
    num_timesteps = label_pred.shape[-1]
    for i in range(num_timesteps):
        save_segmentation_prediction(
            label_pred=label_pred[..., i],
            uids=uids,
            out_dir=out_dir / f"step_{i}",
            tfds_dir=tfds_dir,
            reference_suffix=reference_suffix,
            output_suffix=output_suffix,
        )
