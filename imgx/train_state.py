"""Training state and checkpoints.

https://github.com/google/flax/blob/main/examples/imagenet/train.py
"""
from __future__ import annotations

from pathlib import Path

import chex
import jax
from flax.training import checkpoints
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state as ts


class TrainState(ts.TrainState):
    """Train state.

    If using nn.BatchNorm, batch_stats needs to be tracked.
    https://flax.readthedocs.io/en/latest/guides/batch_norm.html
    https://github.com/google/flax/blob/main/examples/imagenet/train.py
    """

    dynamic_scale: dynamic_scale_lib.DynamicScale


def restore_checkpoint(
    state: TrainState, ckpt_dir: Path, step: int | None = None
) -> chex.ArrayTree:
    """Restore the latest checkpoint from the given directory.

    Args:
        state: train state.
        ckpt_dir: checkpoint directory.
        step: checkpoint step to restore from, if None use the latest one.

    Returns:
        restored state.
    """
    return checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state, step=step)


def save_checkpoint(train_state: TrainState, ckpt_dir: Path, keep: int) -> str:
    """Save the current training state to the given directory.

    Args:
        train_state: train state.
        ckpt_dir: checkpoint directory.
        keep: maximum number of checkpoints to keep.

    Returns:
        checkpoint path.
    """
    train_state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], train_state))
    step = int(train_state.step)
    ckpt_path = checkpoints.save_checkpoint_multiprocess(ckpt_dir, train_state, step, keep=keep)
    return ckpt_path
