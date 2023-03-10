"""AMOS CT image dataset."""

import json
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np
import tensorflow_datasets as tfds

from imgx import IMAGE, LABEL, TEST_SPLIT, TRAIN_SPLIT, UID, VALID_SPLIT
from imgx.datasets.preprocess import load_and_preprocess_image_and_label

_DESCRIPTION = """
The data set includes 500 CT images from  acquired from Amos: A large-scale abdominal multi-organ benchmark for versatile medical image segmentation.
"""  # noqa: E501

_CITATION = """
@article{ji2022amos,
  title={Amos: A large-scale abdominal multi-organ benchmark for versatile medical image segmentation},
  author={Ji, Yuanfeng and Bai, Haotian and Yang, Jie and Ge, Chongjian and Zhu, Ye and Zhang, Ruimao and Li, Zhen and Zhang, Lingyan and Ma, Wanling and Wan, Xiang and others},
  journal={arXiv preprint arXiv:2206.08023},
  year={2022}
}
"""  # noqa: E501

AMOS_CT_IMAGE_SPACING: Tuple[float, float, float] = (1.5, 1.5, 5.0)
AMOS_CT_IMAGE_SHAPE: Tuple[int, int, int] = (192, 128, 128)
AMOS_CLASS_NAMES = [
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
]
AMOS_NUM_CLASSES = len(AMOS_CLASS_NAMES) + 1  # include background

AMOS_TFDS_FOLD = "ZIP.zenodo.org_record_7155725_files_amos22uIHT-rS-kf9k08JEainUcaKYppRMKikiGkm48PK52p0.zip"  # noqa: E501, pylint: disable=line-too-long
AMOS_VALID_RATIO = 0.1


def keep_ct_data(path_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
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
            continue
        filtered_pairs.append(sample)
    return filtered_pairs


class Builder(
    tfds.core.GeneratorBasedBuilder, skip_registration=True
):  # type: ignore[call-arg]
    """DatasetBuilder for male_pelvic_mr dataset.

    Skip registration due to an error in test, saying already registered.
    https://github.com/tensorflow/datasets/issues/552

    There are eight classes:
        "BladderMask", "BoneMask", "ObdInternMask", "TZ",
        "CG", "RectumMask", "SV", "NVB",
    corresponding to 1, 2, ..., 8.
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
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
                        shape=AMOS_CT_IMAGE_SHAPE, dtype=np.float32
                    ),
                    LABEL: tfds.features.Tensor(
                        shape=AMOS_CT_IMAGE_SHAPE, dtype=np.uint16
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://zenodo.org/record/7155725#.ZAN1mOzP2rM",
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: tfds.download.DownloadManager
    ) -> Dict[str, Generator[Tuple[str, Dict[str, np.ndarray]], None, None]]:
        """Returns dict of generators.

        Args:
            dl_manager: for downloading files.

        Returns:
            dict mapping split to generators.
        """
        # Download data from zenodo
        zip_dir = dl_manager.download_and_extract(
            "https://zenodo.org/record/7155725/files/amos22.zip"
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
        image_label_pairs: List[Dict[str, str]],
        data_dir: Path,
        preprocessed_dir: Path,
    ) -> Generator[Tuple[str, Dict[str, np.ndarray]], None, None]:
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
            uid = Path(image_path).name.split(".")[0].split("_")[1]
            image, label = load_and_preprocess_image_and_label(
                uid=uid,
                image_path=data_dir / image_path,
                label_path=data_dir / label_path,
                out_dir=preprocessed_dir,
                target_spacing=AMOS_CT_IMAGE_SPACING,
                target_shape=AMOS_CT_IMAGE_SHAPE,
                intensity_range=(-991.0, 362.0),
            )
            yield uid, {
                UID: uid,
                IMAGE: image,
                LABEL: label,
            }
