"""Module to handle multi-devices."""
from __future__ import annotations

import chex
import jax
import jax.numpy as jnp


def broadcast_to_local_devices(value: chex.ArrayTree) -> chex.ArrayTree:
    """Broadcasts an object to all local devices.

    Args:
        value: value to be broadcast.

    Returns:
        broadcast value.
    """
    devices = jax.local_devices()
    return jax.tree_map(lambda v: jax.device_put_sharded(len(devices) * [v], devices), value)


def get_first_replica_values(value: chex.ArrayTree) -> chex.ArrayTree:
    """Gets values from the first replica.

    Args:
        value: broadcast value.

    Returns:
        value of the first replica.
    """
    return jax.tree_map(lambda x: x[0], value)


def bind_rng_to_host_or_device(
    rng: jnp.ndarray,
    bind_to: str | None = None,
    axis_name: str | tuple[str, ...] | None = None,
) -> jnp.ndarray:
    """Binds a rng to the host or device.

    https://github.com/google-research/scenic/blob/main/scenic/train_lib/train_utils.py#L577

    Must be called from within a pmapped function. Note that when binding to
    "device", we also bind the rng to hosts, as we fold_in the rng with
    axis_index, which is unique for devices across all hosts.

    Args:
        rng: A jax.random.PRNGKey.
        bind_to: Must be one of the 'host' or 'device'. None means no binding.
        axis_name: The axis of the devices we are binding rng across, necessary
            if bind_to is device.

    Returns:
        jax.random.PRNGKey specialized to host/device.
    """
    if bind_to is None:
        return rng
    if bind_to == "host":
        return jax.random.fold_in(rng, jax.process_index())
    if bind_to == "device":
        return jax.random.fold_in(rng, jax.lax.axis_index(axis_name))
    raise ValueError("`bind_to` should be one of the `[None, 'host', 'device']`")


def shard(
    pytree: chex.ArrayTree,
    num_replicas: int,
) -> chex.ArrayTree:
    """Reshapes all arrays in the pytree to add a leading shard dimension.

    We assume that all arrays in the pytree have leading dimension
    divisible by num_devices_per_replica.

    Args:
        pytree: A pytree of arrays to be sharded.
        num_replicas: number of model replicas.

    Returns:
      Sharded data.
    """

    def _shard_array(array: jnp.ndarray) -> jnp.ndarray:
        return array.reshape((num_replicas, -1) + array.shape[1:])

    return jax.tree_map(_shard_array, pytree)


def unshard(pytree: chex.ArrayTree, device: jax.Device) -> chex.ArrayTree:
    """Reshapes arrays from [ndev, bs, ...] to [host_bs, ...].

    Args:
        pytree: A pytree of arrays to be sharded.
        device: device to put.

    Returns:
        Sharded data.
    """

    def _unshard_array(array: jnp.ndarray) -> jnp.ndarray:
        ndev, bs = array.shape[:2]
        return array.reshape((ndev * bs,) + array.shape[2:])

    pytree = jax.device_put(pytree, device)
    return jax.tree_map(_unshard_array, pytree)
