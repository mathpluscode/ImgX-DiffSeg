"""Constants for imgx_datasets.

Cannot be defined in __init__.py because of circular import.
__init__.py imports from each data set, which imports constants from this file.
"""
from pathlib import Path

# splits
TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
TEST_SPLIT = "test"

# data dict keys
UID = "uid"
IMAGE = "image"  # in a batch, keys having image are also considered as images
LABEL = "label"  # in a batch, keys having label are also considered as labels

TFDS_DIR: Path = Path.home() / "tensorflow_datasets"
TFDS_EXTRACTED_DIR: Path = TFDS_DIR / "downloads" / "extracted"
TFDS_MANUAL_DIR: Path = TFDS_DIR / "downloads" / "manual"

# segmentation task
FOREGROUND_RANGE = "foreground_range"  # added during pre-processing in tf for augmentation
