"""Constants for imgx_datasets."""
from pathlib import Path

# splits
TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
TEST_SPLIT = "test"

# data dict keys
UID = "uid"
IMAGE = "image"
LABEL = "label"

TFDS_DIR: Path = Path.home() / "tensorflow_datasets"
TFDS_EXTRACTED_DIR: Path = TFDS_DIR / "downloads" / "extracted"
TFDS_MANUAL_DIR: Path = TFDS_DIR / "downloads" / "manual"

# segmentation task
FOREGROUND_RANGE = "foreground_range"
