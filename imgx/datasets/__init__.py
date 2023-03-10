"""Dataset module to build tensorflow datasets."""
from collections import namedtuple
from pathlib import Path

from imgx.datasets.amos_ct.amos_ct_dataset_builder import (
    AMOS_CT_IMAGE_SHAPE,
    AMOS_CT_IMAGE_SPACING,
    AMOS_NUM_CLASSES,
    AMOS_TFDS_FOLD,
)
from imgx.datasets.male_pelvic_mr.male_pelvic_mr_dataset_builder import (
    PELVIC_IMAGE_SHAPE,
    PELVIC_IMAGE_SPACING,
    PELVIC_NUM_CLASSES,
    PELVIC_TFDS_FOLD,
)

Dataset = namedtuple(
    "Dataset",
    [
        "train_iter",
        "valid_iter",
        "test_iter",
        # valid_iter is repeated, needs to know how many batches to eval
        "num_valid_steps",
        # for simplicity, better to know how many batches to eval
        "num_test_steps",
    ],
)

DIR_TFDS: Path = Path.home() / "tensorflow_datasets"

# segmentation task
FOREGROUND_RANGE = "foreground_range"

# supported datasets
MALE_PELVIC_MR = "male_pelvic_mr"
AMOS_CT = "amos_ct"

IMAGE_SPACING_MAP = {
    MALE_PELVIC_MR: PELVIC_IMAGE_SPACING,
    AMOS_CT: AMOS_CT_IMAGE_SPACING,
}
DIR_TFDS_PROCESSED_MAP = {
    MALE_PELVIC_MR: DIR_TFDS
    / "downloads"
    / "extracted"
    / PELVIC_TFDS_FOLD
    / "preprocessed",
    AMOS_CT: DIR_TFDS
    / "downloads"
    / "extracted"
    / AMOS_TFDS_FOLD
    / "preprocessed",
}
IMAGE_SHAPE_MAP = {
    MALE_PELVIC_MR: PELVIC_IMAGE_SHAPE,
    AMOS_CT: AMOS_CT_IMAGE_SHAPE,
}
NUM_CLASSES_MAP = {
    MALE_PELVIC_MR: PELVIC_NUM_CLASSES,
    AMOS_CT: AMOS_NUM_CLASSES,
}
