"""male_pelvic_mr dataset."""

from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
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
from imgx_datasets.save import save_uids

_DESCRIPTION = """
The data set includes 589 T2-weighted images acquired from the same number of
patients collected by seven studies,
INDEX (Dickinson et al., 2013),
SmartTarget Biopsy Trial (Hamid et al., 2019),
PICTURE (Simmons et al., 2014),
TCIA Prostate3T (Litjens et al., 2015),
Promise12 (Litjens et al., 2014),
TCIA ProstateDx (Diagnosis) (Bloch et al., 2015), and
the Prostate MR Image Database (Choyke et al., 2016).
"""

_CITATION = """
@article{li2022prototypical,
  title={Prototypical few-shot segmentation for cross-institution male pelvic structures with spatial registration},
  author={Li, Yiwen and Fu, Yunguan and Gayo, Iani and Yang, Qianye and Min, Zhe and Saeed, Shaheer and Yan, Wen and Wang, Yipei and Noble, J Alison and Emberton, Mark and others},
  journal={arXiv preprint arXiv:2209.05160},
  year={2022}
}
"""  # noqa: E501

MALE_PELVIC_MR_TFDS_FOLD = "ZIP.zenodo.org_record_7013610_files_dataW0mCI6aH_V-TdeDbM4TdKelNcJ5ZxbAi5isebqCnMr0.zip"  # noqa: E501, pylint: disable=line-too-long
MALE_PELVIR_MR_INFO = OneHotLabeledDatasetInfo(
    name="male_pelvic_mr",
    tfds_preprocessed_dir=TFDS_EXTRACTED_DIR / MALE_PELVIC_MR_TFDS_FOLD / "preprocessed",
    image_spacing=(0.75, 0.75, 2.5),
    image_spatial_shape=(256, 256, 48),
    image_channels=1,
    class_names=(
        "BladderMask",
        "BoneMask",
        "ObdInternMask",
        "TZ",
        "CG",
        "RectumMask",
        "SV",
        "NVB",
    ),
    classes_are_exclusive=True,
)

MALE_PELVIC_MR_TRAIN_RATIO = 0.75
MALE_PELVIC_MR_INSTITUTIONS = [
    "UCL",
    "Prostate3T",
    "ProstateDx",
    "ProstateMRI",
    "bergen",
    "Nijmegen",
    "Rutgers",
]


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
                        shape=MALE_PELVIR_MR_INFO.input_image_shape,
                        dtype=np.float32,
                        encoding=tfds.features.Encoding.ZLIB,
                    ),
                    LABEL: tfds.features.Tensor(
                        shape=MALE_PELVIR_MR_INFO.label_shape,
                        dtype=np.uint8,
                        encoding=tfds.features.Encoding.ZLIB,
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://zenodo.org/record/7013610#.Y1U95-zMKrM",
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
            "https://zenodo.org/record/7013610/files/data.zip"
        )
        data_dir = zip_dir / "data"
        preprocessed_dir = zip_dir / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        institution_path = dl_manager.download(
            "https://zenodo.org/record/7013610/files/institution.txt"
        )

        # Organize metadata
        df = pd.read_csv(
            institution_path,
            sep=" ",
            header=None,
            names=["key", "institution"],
            dtype={"key": object, "institution": object},
        )
        ins_to_uids = df.groupby("institution")["key"].agg(list).to_dict()

        # train/valid/test split
        # for each institution
        # - train+valid:test = 3:1
        # - valid has two images
        train_uids = []
        valid_uids = []
        test_uids = []
        for uids in ins_to_uids.values():
            num_examples = len(uids)
            num_examples_valid = 2
            num_examples_train = int(num_examples * MALE_PELVIC_MR_TRAIN_RATIO) - num_examples_valid
            train_uids += uids[2:num_examples_train]
            valid_uids += uids[:2]
            test_uids += uids[num_examples_train:]

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
            image, label = load_and_preprocess_image_and_label(
                uid=uid,
                image_path=data_dir / f"{uid}_img.nii",
                label_path=data_dir / f"{uid}_mask.nii",
                out_dir=preprocessed_dir,
                target_spacing=MALE_PELVIR_MR_INFO.image_spacing,
                target_shape=MALE_PELVIR_MR_INFO.image_spatial_shape,
                intensity_range=None,
            )
            yield uid, {
                UID: uid,
                IMAGE: image[..., None],
                LABEL: label,
            }
