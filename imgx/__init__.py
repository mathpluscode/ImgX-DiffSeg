"""A Jax-based DL toolkit for biomedical and bioinformatics applications."""
EPS = 1.0e-5  # machine error

# jax device
# one model can be stored across multiple shards/slices
# given 8 devices, it can be grouped into 4x2
# if num_devices_per_replica = 2, then one model is stored across 2 devices
# so the replica_axis would be of size 4
SHARD_AXIS = "shard_axis"
REPLICA_AXIS = "replica_axis"
