"""muscle_us dataset."""
from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

import cv2
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
from scipy.ndimage import binary_fill_holes

from imgx.datasets.constant import (
    IMAGE,
    LABEL,
    TEST_SPLIT,
    TFDS_EXTRACTED_DIR,
    TRAIN_SPLIT,
    UID,
    VALID_SPLIT,
)
from imgx.datasets.dataset_info import OneHotLabeledDatasetInfo
from imgx.datasets.save import load_2d_grayscale_image, save_2d_grayscale_image, save_uids

# https://github.com/tensorflow/datasets/issues/2761
tfds.core.utils.gcs_utils._is_gcs_disabled = True  # pylint: disable=protected-access
os.environ["NO_GCE_CHECK"] = "true"

_DESCRIPTION = """
The dataset included 3917 images of biceps brachii, tibialis anterior and
gastrocnemius medialis acquired on 1283 subjects.
"""

_CITATION = """
@article{marzola2021deep,
  title={Deep learning segmentation of transverse musculoskeletal ultrasound images for neuromuscular disease assessment},
  author={Marzola, Francesco and van Alfen, Nens and Doorduin, Jonne and Meiburger, Kristen M},
  journal={Computers in Biology and Medicine},
  volume={135},
  pages={104623},
  year={2021},
  publisher={Elsevier}
}
"""  # noqa: E501

# matlab strel('disk',3)
STREL_DISK_3 = np.ones((5, 5), dtype=np.uint8)
# matlab strel('disk',5)
STREL_DISK_5 = np.array(
    [
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
    ],
    dtype=np.uint8,
)
# matlab strel('disk',10)
STREL_DISK_10 = np.array(
    [
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)


def select_connected_component(
    mask: np.ndarray,
    threshold: float = 0.75,
) -> np.ndarray:
    """Select connected component.

    https://www.sciencedirect.com/science/article/pii/S0010482521004170

    If more than one structure was located and the second-largest structure
    was smaller than 75% of the largest structure, the structure having
    the largest area was selected.

    If the difference between the two largest connected areas was less
    than the previous case, the selected structure is the one that is the
    most superficial, i.e., towards the top.

    Args:
        mask: shape (W, H).
        threshold: threshold for dropping the second large connected component.

    Returns:
        mask: shape (W, H).
    """
    if mask.ndim != 2:
        raise ValueError(f"mask should be 2D, but {mask.ndim} is given.")

    # input 8 is connectivity, default value in SAM
    # - n is the number of connected components, including background
    # - labels is the label image, same shape as input mask
    #   each connected component is assigned a unique integer label
    # - stats is a (n, 5) matrix
    #   where first row stats[0, :] corresponds to background
    #   each row is x_min, y_min, box_width, box_height, area
    # casting to uint8/int8 is necessary for cv2
    n, labels, stats, _ = cv2.connectedComponentsWithStats(np.asarray(mask, dtype=np.uint8), 8)

    if n <= 2:
        # n = 1, no foreground
        # n = 2, only one component
        return mask

    # get index of from smallest to largest components
    indices = np.argsort(stats[1:, -1]) + 1
    largest_idx = indices[-1]
    second_largest_idx = indices[-2]
    largest_area = stats[largest_idx, -1]
    second_largest_area = stats[second_largest_idx, -1]

    if second_largest_area <= largest_area * threshold:
        # keep largest only
        return (labels == largest_idx).astype(mask.dtype)
    # keep the one is the most superficial
    # image is of shape (480, 521)
    # x axis corresponds to 480, smaller x correspond to top/superficial
    # so x_min smaller means the component is more superficial
    if stats[largest_idx, 0] < stats[second_largest_idx, 0]:
        # largest is more superficial
        # keep largest only
        return (labels == largest_idx).astype(mask.dtype)
    # second-largest is more superficial
    # keep second-largest only
    return (labels == second_largest_idx).astype(mask.dtype)


def post_process_mask(mask: np.ndarray) -> np.ndarray:
    """Post process mask.

    The code follows the MATLAB code in post_process_muscleNet.m from
    https://data.mendeley.com/datasets/3jykz7wz8d/1

    Args:
        mask: shape (W, H).

    Returns:
        mask: shape (W, H).
    """
    dtype = mask.dtype
    mask = binary_fill_holes(mask)

    mask = mask.astype(np.uint8)
    mask = cv2.erode(mask, STREL_DISK_3, iterations=1)
    mask = cv2.dilate(mask, STREL_DISK_5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, STREL_DISK_10)

    mask = select_connected_component(mask)
    mask = binary_fill_holes(mask)
    return mask.astype(dtype)


class MuscleUSDatasetInfo(OneHotLabeledDatasetInfo):
    """BRATS2021 MR dataset info with custom label, mask transformation."""

    def post_process_label(self, label: jnp.ndarray) -> jnp.ndarray:
        """Transform logits to label with post-processing.

        There are only foreground and background

        Args:
            label: (batch, width, height)
                or (batch, width, height, num_timesteps).

        Returns:
            Label with integers.
        """
        if label.ndim not in (3, 4):
            # (batch, width, height) or (batch, width, height, num_timesteps)
            raise ValueError(
                f"Invalid label shape {label.shape}, should be "
                f"(batch, width, height) "
                f"or (batch, width, height, num_timesteps)"
            )

        # post process
        label_np = np.array(label)
        if label.ndim == 3:
            for i in range(label_np.shape[0]):
                label_np[i] = post_process_mask(label_np[i])
        else:
            for i in range(label_np.shape[0]):
                for j in range(label_np.shape[-1]):
                    label_np[i, :, :, j] = post_process_mask(label_np[i, :, :, j])

        # convert back to jnp
        return jnp.array(label_np)


MUSCLE_US_TFDS_FOLD = "ZIP.data.mend.com_publ-file_data_3jyk_file_b160-98XNE6wqHCOxLE8Ap4-__x82VYGr1POiW-quZggxPZSCk"  # noqa: E501, pylint: disable=line-too-long
MUSCLE_US_INFO = MuscleUSDatasetInfo(
    name="muscle_us",
    tfds_preprocessed_dir=TFDS_EXTRACTED_DIR / MUSCLE_US_TFDS_FOLD / "preprocessed",
    image_spacing=(1.0, 1.0),
    image_spatial_shape=(480, 512),
    image_channels=1,
    class_names=("CSA",),  # cross-sectional area
    classes_are_exclusive=True,
)


class Builder(tfds.core.GeneratorBasedBuilder, skip_registration=True):
    """DatasetBuilder for male_pelvic_mr dataset.

    Skip registration due to an error in test, saying already registered.
    https://github.com/tensorflow/datasets/issues/552
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES: ClassVar[dict[str, str]] = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    UID: tfds.features.Text(),
                    IMAGE: tfds.features.Tensor(
                        shape=MUSCLE_US_INFO.input_image_shape,
                        dtype=np.float32,
                        encoding=tfds.features.Encoding.ZLIB,
                    ),
                    LABEL: tfds.features.Tensor(
                        shape=MUSCLE_US_INFO.label_shape,
                        dtype=np.uint8,
                        encoding=tfds.features.Encoding.ZLIB,
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://data.mendeley.com/datasets/3jykz7wz8d/1",
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: tfds.download.DownloadManager
    ) -> dict[str, Generator[tuple[str, dict[str, np.ndarray]], None, None]]:
        """Returns dict of generators.

        Args:
            dl_manager: for downloading files.

        Returns:
            dict mapping split to generators.
        """
        # Download data from zenodo
        zip_dir = dl_manager.download_and_extract(
            "https://data.mendeley.com/public-files/datasets/3jykz7wz8d/files/b1601a22-98a5-41ce-9fbb-54dd21834b55/file_downloaded"  # pylint: disable=line-too-long
        )
        data_dir = (
            zip_dir / "Polito-Radboud-DeepLearningUS" / "Segmentation_models" / "data" / "MuscleCNN"
        )
        preprocessed_dir = zip_dir / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

        # Read the uids
        # Each uid shall have both image and mask
        train_uids = [x.stem for x in (data_dir / "train").glob("*.png")]
        train_mask_uids = [x.stem for x in (data_dir / "trainannot").glob("*.png")]
        train_uids = [x for x in train_uids if x in train_mask_uids]

        valid_uids = [x.stem for x in (data_dir / "val").glob("*.png")]
        valid_mask_uids = [x.stem for x in (data_dir / "valannot").glob("*.png")]
        valid_uids = [x for x in valid_uids if x in valid_mask_uids]

        test_uids = [x.stem for x in (data_dir / "test").glob("*.png")]
        test_mask_uids = [x.stem for x in (data_dir / "testannot").glob("*.png")]
        test_uids = [x for x in test_uids if x in test_mask_uids]

        # save uids for reproducibility
        save_uids(
            train_uids=train_uids,
            valid_uids=valid_uids,
            test_uids=test_uids,
            out_dir=preprocessed_dir,
        )

        # Returns the Dict[split names, Iterator[Key, Example]]
        return {
            TRAIN_SPLIT: self._generate_examples(
                uids=train_uids,
                image_dir=data_dir / "train",
                mask_dir=data_dir / "trainannot",
                preprocessed_dir=preprocessed_dir,
            ),
            VALID_SPLIT: self._generate_examples(
                uids=valid_uids,
                image_dir=data_dir / "val",
                mask_dir=data_dir / "valannot",
                preprocessed_dir=preprocessed_dir,
            ),
            TEST_SPLIT: self._generate_examples(
                uids=test_uids,
                image_dir=data_dir / "test",
                mask_dir=data_dir / "testannot",
                preprocessed_dir=preprocessed_dir,
            ),
        }

    def _generate_examples(
        self,
        uids: list[str],
        image_dir: Path,
        mask_dir: Path,
        preprocessed_dir: Path,
    ) -> Generator[tuple[str, dict[str, np.ndarray]], None, None]:
        """Yields examples.

        Args:
            uids: unique ids for images and masks.
            image_dir: directory saving images.
            mask_dir: directory saving masks.
            preprocessed_dir: directory to save processed images/masks.

        Yields:
            - image_key
            - dict having image and label numpy arrays.
        """
        for uid in uids:
            image = load_2d_grayscale_image(image_dir / f"{uid}.png", dtype=np.float32)
            label = load_2d_grayscale_image(mask_dir / f"{uid}.png")

            save_2d_grayscale_image(label, preprocessed_dir / f"{uid}_mask_preprocessed.png")

            yield uid, {
                UID: uid,
                IMAGE: image[..., None],
                LABEL: label,
            }
