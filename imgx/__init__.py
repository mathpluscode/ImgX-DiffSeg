"""A Jax-based DL toolkit for biomedical and bioinformatics applications."""
from pathlib import Path

# machine error
EPS = 1.0e-5

NAN_MASK = "nan_mask"

# path for all non-tensorflow-dataset data sets
DIR_DATA = Path("datasets")

# splits
TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
TEST_SPLIT = "test"

# jax device
# one model can be stored across multiple shards/slices
# given 8 devices, it can be grouped into 4x2
# if num_devices_per_replica = 2, then one model is stored across 2 devices
# so the replica_axis would be of size 4
SHARD_AXIS = "shard_axis"
REPLICA_AXIS = "replica_axis"

# data dict keys
UID = "uid"
IMAGE = "image"
LABEL = "label"
