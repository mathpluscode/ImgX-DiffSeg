"""Test TrainState and related functions."""

from pathlib import Path
from typing import Dict

import chex
import jax.numpy as jnp
import jax.random
import jmp
import pytest
from chex._src import fake

from imgx.device import broadcast_to_local_devices
from imgx.exp import train_state


def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


@pytest.fixture(name="dummy_train_state_dict")
def get_dummy_train_state_dict() -> Dict[str, jnp.ndarray]:
    """A dummy dict for train state attribute values.

    Returns:
        A dict of dummy values.
    """
    key = jax.random.PRNGKey(0)
    return {
        "params": jax.random.uniform(key, (3, 5)),
        "network_state": jax.random.uniform(key, (4, 5)),
        "opt_state": jax.random.uniform(key, (5, 5)),
        "global_step": jnp.array(0, dtype=jnp.int32),
        "rng": jax.random.PRNGKey(0),
        "ema_params": jax.random.uniform(key, (3, 3)),
        "ema_network_state": jax.random.uniform(key, (3, 4)),
    }


def test_save_restore_array_tree(
    tmp_path: Path, dummy_train_state_dict: chex.ArrayTree
) -> None:
    """Test by saving and restoring.

    Args:
        tmp_path: fixture for temp path.
        dummy_train_state_dict: dummy data to save.
    """
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    train_state.save_array_tree(ckpt_dir, dummy_train_state_dict)
    restored = train_state.restore_array_tree(ckpt_dir)
    chex.assert_trees_all_equal(dummy_train_state_dict, restored)


@pytest.mark.parametrize(
    "loss_scale_type",
    [
        "NoOpLossScale",
        "StaticLossScale",
        "DynamicLossScale",
    ],
)
def test_save_restore_ckpt(
    loss_scale_type: str, tmp_path: Path, dummy_train_state_dict: chex.ArrayTree
) -> None:
    """Test by saving and restoring.

    Args:
        loss_scale_type: NoOpLossScale, StaticLossScale, DynamicLossScale.
        tmp_path: fixture for temp path.
        dummy_train_state_dict: dummy data to save.
    """
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    train_state_dict = jax.tree_map(
        broadcast_to_local_devices, dummy_train_state_dict
    )
    if loss_scale_type == "NoOpLossScale":
        loss_scale = jmp.NoOpLossScale()
    else:
        scale = jmp.half_dtype()(2**15)
        loss_scale = getattr(jmp, loss_scale_type)(scale)
    loss_scale = broadcast_to_local_devices(loss_scale)
    dummy_train_state = train_state.TrainState(  # type: ignore[call-arg]
        params=train_state_dict["params"],
        network_state=train_state_dict["network_state"],
        opt_state=train_state_dict["opt_state"],
        loss_scale=loss_scale,
        global_step=train_state_dict["global_step"],
        rng=train_state_dict["rng"],
        ema_params=train_state_dict["ema_params"],
        ema_network_state=train_state_dict["ema_network_state"],
    )

    train_state.save_ckpt(dummy_train_state, ckpt_dir)
    restored_train_state = train_state.restore_ckpt(ckpt_dir)

    if loss_scale_type == "DynamicLossScale":
        dummy_loss_scale = dummy_train_state.loss_scale
        restored_loss_scale = restored_train_state.loss_scale
        dummy_train_state.loss_scale = -1
        restored_train_state.loss_scale = -1
        chex.assert_trees_all_equal(dummy_train_state, restored_train_state)
        chex.assert_trees_all_equal(
            dummy_loss_scale.loss_scale, restored_loss_scale.loss_scale
        )
        chex.assert_trees_all_equal(
            dummy_loss_scale.counter, restored_loss_scale.counter
        )
        chex.assert_trees_all_equal(
            dummy_loss_scale.period, restored_loss_scale.period
        )
        chex.assert_trees_all_equal(
            dummy_loss_scale.factor, restored_loss_scale.factor
        )
    else:
        chex.assert_trees_all_equal(dummy_train_state, restored_train_state)
