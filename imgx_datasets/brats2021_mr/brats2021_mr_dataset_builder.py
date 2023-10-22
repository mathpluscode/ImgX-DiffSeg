"""brats2021_mr dataset."""
from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import SimpleITK as sitk  # noqa: N813
import tensorflow_datasets as tfds

from imgx_datasets.constant import (
    IMAGE,
    LABEL,
    TEST_SPLIT,
    TFDS_MANUAL_DIR,
    TRAIN_SPLIT,
    UID,
    VALID_SPLIT,
)
from imgx_datasets.dataset_info import DatasetInfo
from imgx_datasets.preprocess import clip_and_normalise_intensity
from imgx_datasets.util import save_uids

_DESCRIPTION = """
All BraTS mpMRI scans are available as NIfTI files (.nii.gz) and describe
a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2),
and d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes,
and were acquired with different clinical protocols and various scanners
from multiple data contributing institutions.

All the imaging datasets have been annotated manually, by one to four raters,
following the same annotation protocol, and their annotations were approved
by experienced neuro-radiologists. Annotations comprise the GD-enhancing
tumor (ET — label 4), the peritumoral edematous/invaded tissue (ED — label 2),
and the necrotic tumor core (NCR — label 1), as described both in
the BraTS 2012-2013 TMI paper and in the latest BraTS summarizing paper.
The ground truth data were created after their pre-processing,
i.e., co-registered to the same anatomical template,
interpolated to the same resolution (1 mm3) and skull-stripped.

All images are of size 240x240x155. The brain is only located in
[37, 216)x[10, 229)x[0, 155), where index starts at zero.
The segmentation masks are defined over
[45, 194)x[23, 216)x[0, 150), where index starts at zero.
Cropping can reduce the image size by more than 30%.
"""

_CITATION = """
@article{baid2021rsna,
  title={The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification},
  author={Baid, Ujjwal and Ghodasara, Satyam and Mohan, Suyash and Bilello, Michel and Calabrese, Evan and Colak, Errol and Farahani, Keyvan and Kalpathy-Cramer, Jayashree and Kitamura, Felipe C and Pati, Sarthak and others},
  journal={arXiv preprint arXiv:2107.02314},
  year={2021}
}
"""  # noqa: E501

BRATS2021_MR_TFDS_FOLD = "BraTS2021_Kaggle"
BRATS2021_MR_CROP_LOWER: tuple[int, int, int] = (37, 10, 0)
BRATS2021_MR_CROP_UPPER: tuple[int, int, int] = (24, 11, 0)
BRATS2021_MR_NUM_IMAGES = 1251
BRATS2021_MR_NUM_TRAIN_IMAGES = 938
BRATS2021_MR_NUM_VALID_IMAGES = 31
BRATS2021_MR_NUM_TEST_IMAGES = 282
BRATS2021_MR_MODALITIES = ["t1", "t1ce", "t2", "flair"]

# threshold values are from https://arxiv.org/abs/2110.03352
BRATS2021_MR_PROBS_WT_THRESHOLD = 0.45
BRATS2021_MR_PROBS_TC_THRESHOLD = 0.4
BRATS2021_MR_PROBS_ET_THRESHOLD = 0.45


class BRATS2021MRNestedDatasetInfo(DatasetInfo):
    """BRATS2021 MR dataset info with custom label, mask transformation."""

    @property
    def num_classes(self) -> int:
        """Number of classes for segmentation."""
        return 2

    def logits_to_label(self, x: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Transform logits to label with integers.

        https://arxiv.org/abs/2110.03352

        Args:
            x: logits.
            axis: axis of num_classes.

        Returns:
            Mask with integers.
        """
        return jnp.argmax(x, axis=axis)

    def label_to_mask(
        self, x: jnp.ndarray, axis: int, dtype: jnp.dtype = jnp.float32
    ) -> jnp.ndarray:
        """Transform label to multi hot mask.

        mask corresponds to [WT, TC, ET].
        0 is background, mask is [0, 0, 0]
        1 is WT, mask is [1, 0, 0]
        2 is TC, mask is [1, 1, 0]
        3 is ET, mask is [1, 1, 1]

        Args:
            x: label.
            axis: axis of num_classes.
            dtype: dtype of output.

        Returns:
            One hot probabilities.
        """
        return jax.nn.one_hot(
            x=jnp.asarray(x >= 1, dtype=jnp.int32),
            num_classes=self.num_classes,
            axis=axis,
            dtype=dtype,
        )


BRATS2021_MR_INFO = BRATS2021MRNestedDatasetInfo(
    tfds_preprocessed_dir=TFDS_MANUAL_DIR / BRATS2021_MR_TFDS_FOLD / "preprocessed",
    image_spacing=(1.0, 1.0, 1.0),
    image_spatial_shape=(179, 219, 155),
    image_channels=4,
    class_names=(
        "WT",  # whole tumor (NCR+ED+ET)
        "TC",  # tumor core (NCR+ET)
        "ET",  # GD-enhancing tumor
    ),
    classes_are_exclusive=True,
)


def convert_brats_label(label: np.ndarray) -> np.ndarray:
    """Convert label from mutual exclusive to inclusive.

    The original label is mutually exclusive, with 3 classes:
        1 is NCR, 2 is ED, 4 is ET.
    New classes are:
        whole tumor (WT) representing classes 1, 2, 4
        tumor core (TC) representing classes 1, 4
        enhancing tumor (ET) representing the class 4

    Args:
        label: label array, of shape (width, height, depth).
            1 is NCR, 2 is ED, 4 is ET.

    Returns:
        label array, of shape (width, height, depth).
            1 is WT, 2 is TC, 3 is ET.
    """
    converted = np.zeros_like(label)
    converted[label >= 1] = 1
    converted[(label == 1) | (label == 4)] = 2
    converted[label == 4] = 3
    return converted


def load_and_preprocess_brats_image_and_label(
    uid: str,
    image_path_dict: dict[str, Path],
    label_path: Path,
    out_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load image and perform resampling/padding/cropping using SimpleITK.

    This function also saves the processed image/masks.
    https://examples.itk.org/src/filtering/imagegrid/resampleanimage/documentation

    No need of resampling and clipping as images have same shape and spacing.

    Args:
        uid: unique id for the image path.
        image_path_dict: map from modality to image path.
        label_path: segmentation path.
        out_dir: directory to save preprocessed files.

    Returns:
        image array, of shape (width, height, depth, 4).
        label array, of shape (width, height, depth).
    """
    label_volume = sitk.ReadImage(str(label_path))
    label_volume = sitk.Crop(label_volume, BRATS2021_MR_CROP_LOWER, BRATS2021_MR_CROP_UPPER)
    label_volume = sitk.Cast(label_volume, sitk.sitkUInt8)

    label = sitk.GetArrayFromImage(label_volume)
    label = convert_brats_label(label)

    # replace label
    label_out_path = out_dir / (uid + "_mask_preprocessed.nii.gz")
    updated_label_volume = sitk.GetImageFromArray(label)
    updated_label_volume.CopyInformation(label_volume)

    sitk.WriteImage(
        image=updated_label_volume,
        fileName=str(label_out_path),
        useCompression=True,
    )

    images = []
    for modality in BRATS2021_MR_MODALITIES:
        image_path = image_path_dict[modality]
        # load and sanity check
        image_volume = sitk.ReadImage(str(image_path))

        # crop
        image_volume = sitk.Crop(image_volume, BRATS2021_MR_CROP_LOWER, BRATS2021_MR_CROP_UPPER)

        # clip and normalise intensity
        image_volume = clip_and_normalise_intensity(
            image_volume=image_volume,
            intensity_range=None,
        )

        # cast to float32
        image_volume = sitk.Cast(image_volume, sitk.sitkFloat32)

        # save
        images.append(sitk.GetArrayFromImage(image_volume))
        image_out_path = out_dir / (uid + f"_{modality}_preprocessed.nii.gz")
        sitk.WriteImage(
            image=image_volume,
            fileName=str(image_out_path),
            useCompression=True,
        )
    image = np.stack(images, axis=-1)

    # move axis
    image = np.transpose(image, axes=[2, 1, 0, 3])
    label = np.transpose(label, axes=[2, 1, 0])

    return image, label


class Builder(tfds.core.GeneratorBasedBuilder, skip_registration=True):
    """DatasetBuilder for brats2021_mr dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES: ClassVar[dict[str, str]] = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Data should be manually downloaded from
    https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1.
    After unzipping `archive.zip` and then `BraTS2021_Training_Data.tar`,
    place `BraTS2021_Training_Data` under
    `~/tensorflow_datasets/downloads/manual/BraTS2021_Kaggle/` such that
    under `BraTS2021_Kaggle/` exist folders per sample, e.g.,
    files corresponding to uid `BraTS2021_01666` should be located at
    `~/tensorflow_datasets/downloads/manual/BraTS2021_Kaggle/BraTS2021_01666/`
    under which there are five files:
    `BraTS2021_01666_flair.nii.gz`,
    `BraTS2021_01666_t1.nii.gz`,
    `BraTS2021_01666_t1ce.nii.gz`,
    `BraTS2021_01666_t2.nii.gz`,
    `BraTS2021_01666_seg.nii.gz`.
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    UID: tfds.features.Text(),
                    IMAGE: tfds.features.Tensor(
                        shape=BRATS2021_MR_INFO.input_image_shape,
                        dtype=np.float32,
                        encoding=tfds.features.Encoding.ZLIB,
                    ),
                    LABEL: tfds.features.Tensor(
                        shape=BRATS2021_MR_INFO.label_shape,
                        dtype=np.uint8,
                        encoding=tfds.features.Encoding.ZLIB,
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1",  # pylint: disable=line-too-long
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
        data_dir = dl_manager.manual_dir / BRATS2021_MR_TFDS_FOLD / "BraTS2021_Training_Data"
        preprocessed_dir = dl_manager.manual_dir / BRATS2021_MR_TFDS_FOLD / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

        # train/valid/test split
        uids_flair = [x.parent.name for x in data_dir.glob("*/*_flair.nii.gz")]
        uids_t1 = [x.parent.name for x in data_dir.glob("*/*_t1.nii.gz")]
        uids_t1ce = [x.parent.name for x in data_dir.glob("*/*_t1ce.nii.gz")]
        uids_t2 = [x.parent.name for x in data_dir.glob("*/*_t2.nii.gz")]
        uids_seg = [x.parent.name for x in data_dir.glob("*/*_seg.nii.gz")]
        uids_set = set(uids_seg)
        for uids_modality in [uids_flair, uids_t1, uids_t1ce, uids_t2]:
            uids_set = uids_set.intersection(set(uids_modality))
        uids = sorted(uids_set)
        train_uids = uids[:BRATS2021_MR_NUM_TRAIN_IMAGES]
        valid_uids = uids[BRATS2021_MR_NUM_TRAIN_IMAGES:-BRATS2021_MR_NUM_TEST_IMAGES]
        test_uids = uids[-BRATS2021_MR_NUM_TEST_IMAGES:]

        # save uids for reproducibility
        save_uids(
            train_uids=train_uids,
            valid_uids=valid_uids,
            test_uids=test_uids,
            out_dir=preprocessed_dir,
        )

        # Returns the Dict[split names, Iterator[uid, example]]
        return {
            TRAIN_SPLIT: self._generate_examples(
                uids=train_uids,
                data_dir=data_dir,
                preprocessed_dir=preprocessed_dir,
            ),
            VALID_SPLIT: self._generate_examples(
                uids=valid_uids,
                data_dir=data_dir,
                preprocessed_dir=preprocessed_dir,
            ),
            TEST_SPLIT: self._generate_examples(
                uids=test_uids,
                data_dir=data_dir,
                preprocessed_dir=preprocessed_dir,
            ),
        }

    def _generate_examples(
        self, uids: list[str], data_dir: Path, preprocessed_dir: Path
    ) -> Generator[tuple[str, dict[str, np.ndarray]], None, None]:
        """Yields examples.

        Args:
            uids: unique ids for images.
            data_dir: directory saving images/masks.
            preprocessed_dir: directory to save processed images/masks.

        Yields:
            - image_key
            - dict having image and label numpy arrays.
        """
        for uid in uids:
            label_path = data_dir / uid / f"{uid}_seg.nii.gz"
            image_path_dict = {}
            for modality in BRATS2021_MR_MODALITIES:
                image_path_dict[modality] = data_dir / uid / f"{uid}_{modality}.nii.gz"
            image, label = load_and_preprocess_brats_image_and_label(
                uid=uid,
                image_path_dict=image_path_dict,
                label_path=label_path,
                out_dir=preprocessed_dir,
            )
            yield uid, {
                UID: uid,
                IMAGE: image,
                LABEL: label,
            }
