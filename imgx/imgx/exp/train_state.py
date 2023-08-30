"""Training state and checkpoints."""
from __future__ import annotations

import pickle
from pathlib import Path

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax

from imgx.device import broadcast_to_local_devices, get_first_replica_values

CHECKPOINT_ATTRS = [
    "params",
    "network_state",
    "opt_state",
]


@chex.dataclass
class TrainState:
    """Dataclass to keep track of state of training.

    The state of training is structured as a chex.dataclass, which enables
    instances of this class to be passed into jax transformations like tree_map
    and pmap.

    The stored values are broadcast across devices.
    """

    params: hk.Params
    network_state: hk.State
    opt_state: optax.OptState
    loss_scale: jmp.LossScale
    global_step: jnp.array
    rng: jax.random.PRNGKey


def save_array_tree(ckpt_dir: Path, state: chex.ArrayTree) -> None:
    """Save the state with arrays and tree saved separately.

    Args:
        ckpt_dir: directory to save.
        state: state to save, including params, optimizer, etc.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "arrays.npy", "wb") as f:
        for x in jax.tree_util.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda _: 0, state)
    with open(ckpt_dir / "tree.pkl", "wb") as f:
        pickle.dump(tree_struct, f)


def restore_array_tree(ckpt_dir: Path) -> chex.ArrayTree:
    """Restore the state from saved files.

    Args:
        ckpt_dir: directory to load.

    Returns:
        Restored state, including params, optimizer, etc.
    """
    with open(ckpt_dir / "tree.pkl", "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
    with open(ckpt_dir / "arrays.npy", "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_util.tree_unflatten(treedef, flat_state)


def save_ckpt(train_state: TrainState, ckpt_dir: Path) -> None:
    """Save the state with arrays and tree saved separately.

    Args:
        train_state: checkpoint to save.
        ckpt_dir: directory to save.
    """
    train_state = jax.tree_map(get_first_replica_values, train_state)
    state_dict = dict(train_state)  # type: ignore[call-overload]

    # loss_scale needs to be stored differently
    loss_scale = state_dict["loss_scale"]
    del state_dict["loss_scale"]
    loss_scale_type = loss_scale.__class__.__name__
    state_dict["loss_scale_type"] = loss_scale_type
    if isinstance(loss_scale, jmp.StaticLossScale):
        state_dict["loss_scale"] = loss_scale.loss_scale
    elif isinstance(loss_scale, jmp.DynamicLossScale):
        state_dict["loss_scale"] = loss_scale.loss_scale
        state_dict["loss_scale_counter"] = loss_scale.counter
        state_dict["loss_scale_period"] = loss_scale.period
        state_dict["loss_scale_factor"] = loss_scale.factor

    save_array_tree(ckpt_dir=ckpt_dir, state=state_dict)


def restore_ckpt(ckpt_dir: Path) -> TrainState:
    """Restore the state from saved files.

    Args:
        ckpt_dir: directory to load.

    Returns:
        train_state: checkpoint to save.
        global_step: number of batch consumed.
    """
    state_dict = restore_array_tree(ckpt_dir)

    # loss_scale needs to be loaded differently
    loss_scale_type = state_dict["loss_scale_type"]
    del state_dict["loss_scale_type"]
    if loss_scale_type == "NoOpLossScale":
        loss_scale = jmp.NoOpLossScale()
    elif loss_scale_type == "StaticLossScale":
        loss_scale = state_dict["loss_scale"]
        del state_dict["loss_scale"]
        loss_scale = jmp.StaticLossScale(loss_scale)
    elif loss_scale_type == "DynamicLossScale":
        loss_scale = state_dict["loss_scale"]
        counter = state_dict["loss_scale_counter"]
        # factor and period are ints not arrays
        period = int(state_dict["loss_scale_period"])
        factor = int(state_dict["loss_scale_factor"])
        del state_dict["loss_scale"]
        del state_dict["loss_scale_counter"]
        del state_dict["loss_scale_period"]
        del state_dict["loss_scale_factor"]
        loss_scale = jmp.DynamicLossScale(
            loss_scale=loss_scale, counter=counter, period=period, factor=factor
        )
    else:
        raise ValueError(f"Unknown loss_scale type {loss_scale_type}.")
    # TODO should consider shards
    state_dict = jax.tree_map(broadcast_to_local_devices, state_dict)
    train_state = TrainState(  # type: ignore[call-arg]
        params=state_dict["params"],
        network_state=state_dict["network_state"],
        opt_state=state_dict["opt_state"],
        loss_scale=loss_scale,
        global_step=state_dict["global_step"],
        rng=state_dict["rng"],
    )
    return train_state


def get_eval_params_and_state_from_ckpt(
    ckpt_dir: Path,
) -> tuple[hk.Params, hk.State]:
    """Get the parameters and state for evaluation from checkpoint.

    Args:
        ckpt_dir: directory to load.

    Returns:
        Broadcast params, state.
    """
    state_dict = restore_array_tree(ckpt_dir)
    params = state_dict["params"]
    state = state_dict["network_state"]
    # make sure arrays are initialised in CPU
    with jax.default_device(jax.devices("cpu")[0]):
        params = jax.tree_map(jnp.asarray, params)
        state = jax.tree_map(jnp.asarray, state)
    # broadcast to other devices
    params = broadcast_to_local_devices(params)
    state = broadcast_to_local_devices(state)
    return params, state
