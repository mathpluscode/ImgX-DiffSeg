"""AMOS CT image dataset."""

import json
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

import numpy as np
import tensorflow_datasets as tfds

from imgx_datasets.constant import (
    IMAGE,
    LABEL,
    TEST_SPLIT,
    TFDS_EXTRACTED_DIR,
    TRAIN_SPLIT,
    UID,
    VALID_SPLIT,
)
from imgx_datasets.dataset_info import OneHotLabeledDatasetInfo
from imgx_datasets.preprocess import load_and_preprocess_image_and_label
from imgx_datasets.util import save_uids

_DESCRIPTION = """
The data set includes 500 CT and 100 MR images from Amos:
A large-scale abdominal multi-organ benchmark for versatile medical
image segmentation. The data set has been divided into training set,
validation set and test set. The training, validation and test sets contain
200+40, 100+20, and 200+40 CT+MR images. However, test set has no labels.
"""

_CITATION = """
@inproceedings{NEURIPS2022_ee604e1b,
 author = {Ji, Yuanfeng and Bai, Haotian and GE, Chongjian and Yang, Jie and Zhu, Ye and Zhang, Ruimao and Li, Zhen and Zhanng, Lingyan and Ma, Wanling and Wan, Xiang and Luo, Ping},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {36722--36732},
 publisher = {Curran Associates, Inc.},
 title = {AMOS: A Large-Scale Abdominal Multi-Organ Benchmark for Versatile Medical Image Segmentation},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/ee604e1bedbd069d9fc9328b7b9584be-Paper-Datasets_and_Benchmarks.pdf},
 volume = {35},
 year = {2022}
}
"""  # noqa: E501

AMOS_CT_TFDS_FOLD = "ZIP.zenodo.org_record_7262581_files_amos22ZnMFi429bmx93zDuUBTrjdo9oGlndCnbGAVAP0I3p_M.zip"  # noqa: E501, pylint: disable=line-too-long
AMOS_CT_INFO = OneHotLabeledDatasetInfo(
    tfds_preprocessed_dir=TFDS_EXTRACTED_DIR / AMOS_CT_TFDS_FOLD / "preprocessed",
    image_spacing=(1.5, 1.5, 5.0),
    image_spatial_shape=(192, 128, 128),
    image_channels=1,
    class_names=(
        "spleen",  # class 1
        "right kidney",
        "left kidney",
        "gall bladder",
        "esophagus",
        "liver",
        "stomach",
        "arota",
        "postcava",
        "pancreas",
        "right adrenal gland",
        "left adrenal gland",
        "duodenum",
        "bladder",
        "prostate/uterus",
    ),
    classes_are_exclusive=True,
)

AMOS_VALID_RATIO = 0.1


def keep_ct_data(path_pairs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep CT data only.

    ID numbers less than 500 belong to CT data,
    otherwise they belong to MRI data.

    Args:
        path_pairs: pairs of image and label relative paths.
            [{'image': './imagesTr/amos_0001.nii.gz',
              'label': './labelsTr/amos_0001.nii.gz'},
             {'image': './imagesTr/amos_0004.nii.gz',
              'label': './labelsTr/amos_0004.nii.gz'}]

    Returns:
        Filtered data.
    """
    filtered_pairs = []
    for sample in path_pairs:
        image_path = sample["image"]
        uid = int(Path(image_path).name.split(".")[0].split("_")[1])
        if uid > 500:
            # MR
            continue
        filtered_pairs.append(sample)
    return filtered_pairs


def path_to_uid(path: str) -> str:
    """Convert path to uid.

    Args:
        path: path to image or label. For example,
            "./imagesTr/amos_0001.nii.gz".

    Returns:
        uid: uid of the image or label. For example,
            "0001".
    """
    uid = Path(path).name  # amos_0001.nii.gz
    uid = uid.replace("amos_", "").replace(".nii.gz", "")  # 0001
    return uid


class Builder(tfds.core.GeneratorBasedBuilder, skip_registration=True):
    """DatasetBuilder for male_pelvic_mr dataset.

    Skip registration due to an error in test, saying already registered.
    https://github.com/tensorflow/datasets/issues/552

    There are eight classes:
        "BladderMask", "BoneMask", "ObdInternMask", "TZ",
        "CG", "RectumMask", "SV", "NVB",
    corresponding to 1, 2, ..., 8.
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES: ClassVar[dict[str, str]] = {
        "1.0.0": "CT image only.",
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
                        shape=AMOS_CT_INFO.input_image_shape,
                        dtype=np.float32,
                        encoding=tfds.features.Encoding.ZLIB,
                    ),
                    LABEL: tfds.features.Tensor(
                        shape=AMOS_CT_INFO.label_shape,
                        dtype=np.uint8,
                        encoding=tfds.features.Encoding.ZLIB,
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://zenodo.org/record/7262581",
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
            "https://zenodo.org/record/7262581/files/amos22.zip"
        )
        data_dir = zip_dir / "amos22"
        preprocessed_dir = zip_dir / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

        # Read metadata
        with open(data_dir / "dataset.json", encoding="utf-8") as f:
            metadata = json.load(f)

        # keep ct data only
        train_pairs = keep_ct_data(metadata["training"])
        valid_pairs = keep_ct_data(metadata["validation"])

        # split valid into valid and test
        num_valid = int(len(valid_pairs) * AMOS_VALID_RATIO)
        test_pairs = valid_pairs[num_valid:]
        valid_pairs = valid_pairs[:num_valid]

        # save uids for reproducibility
        train_uids = [path_to_uid(sample["image"]) for sample in train_pairs]
        valid_uids = [path_to_uid(sample["image"]) for sample in valid_pairs]
        test_uids = [path_to_uid(sample["image"]) for sample in test_pairs]
        save_uids(
            train_uids=train_uids,
            valid_uids=valid_uids,
            test_uids=test_uids,
            out_dir=preprocessed_dir,
        )

        # Returns the Dict[split names, Iterator[Key, Example]]
        return {
            TRAIN_SPLIT: self._generate_examples(
                image_label_pairs=train_pairs,
                data_dir=data_dir,
                preprocessed_dir=preprocessed_dir,
            ),
            VALID_SPLIT: self._generate_examples(
                image_label_pairs=valid_pairs,
                data_dir=data_dir,
                preprocessed_dir=preprocessed_dir,
            ),
            TEST_SPLIT: self._generate_examples(
                image_label_pairs=test_pairs,
                data_dir=data_dir,
                preprocessed_dir=preprocessed_dir,
            ),
        }

    def _generate_examples(
        self,
        image_label_pairs: list[dict[str, str]],
        data_dir: Path,
        preprocessed_dir: Path,
    ) -> Generator[tuple[str, dict[str, np.ndarray]], None, None]:
        """Yields examples.

        Data Prepossessing Following nnUNet, for the CT data,
        we clip the HU values of each scans to the [-991, 362] range.

        https://arxiv.org/abs/2206.08023 Section C.1

        Args:
            image_label_pairs: pairs of image and label relative paths.
                [{'image': './imagesTr/amos_0001.nii.gz',
                  'label': './labelsTr/amos_0001.nii.gz'},
                 {'image': './imagesTr/amos_0004.nii.gz',
                  'label': './labelsTr/amos_0004.nii.gz'}]
            data_dir: directory saving images/masks.
            preprocessed_dir: directory to save processed images/masks.

        Yields:
            - image_key
            - dict having image and label numpy arrays.
        """
        for sample in image_label_pairs:
            image_path = sample["image"]
            label_path = sample["label"]
            # ./imagesTr/amos_0071.nii.gz -> 0071
            uid = path_to_uid(image_path)
            image, label = load_and_preprocess_image_and_label(
                uid=uid,
                image_path=data_dir / image_path,
                label_path=data_dir / label_path,
                out_dir=preprocessed_dir,
                target_spacing=AMOS_CT_INFO.image_spacing,
                target_shape=AMOS_CT_INFO.image_spatial_shape,
                intensity_range=(-991.0, 362.0),
            )
            yield uid, {
                UID: uid,
                IMAGE: image[..., None],
                LABEL: label,
            }
