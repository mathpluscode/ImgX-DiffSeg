"""Training state and checkpoints.

https://github.com/google/flax/blob/main/examples/imagenet/train.py
"""
from __future__ import annotations

from typing import Callable

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from absl import logging
from flax.training import dynamic_scale as dynamic_scale_lib
from omegaconf import DictConfig

from imgx.train_state import TrainState as BaseTrainState
from imgx.train_state import init_optimizer


class TrainState(BaseTrainState):
    """Train state.

    If using nn.BatchNorm, batch_stats needs to be tracked.
    https://flax.readthedocs.io/en/latest/guides/batch_norm.html
    https://github.com/google/flax/blob/main/examples/imagenet/train.py
    """

    loss_count_hist: jnp.ndarray  # mutable
    loss_sq_hist: jnp.ndarray  # mutable


def create_train_state(
    key: jax.Array,
    batch: dict[str, jnp.ndarray],
    model: nn.Module,
    config: DictConfig,
    initialized: Callable[[jax.Array, chex.ArrayTree, nn.Module], chex.ArrayTree],
) -> TrainState:
    """Create initial training state.

    Args:
        key: random key.
        batch: batch data for determining input shapes.
        model: model.
        config: entire configuration.
        initialized: function to get initialized model parameters.

    Returns:
        initial training state.
    """
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.half_precision and platform == "gpu":
        dynamic_scale = dynamic_scale_lib.DynamicScale()

    params = initialized(key, batch, model)

    # count params
    params_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logging.info(f"The model has {params_count:,} parameters.")

    # diffusion related
    num_timesteps = config.task.diffusion.num_timesteps
    loss_count_hist = jnp.zeros((num_timesteps,), dtype=jnp.int32)
    loss_sq_hist = jnp.zeros((num_timesteps,), dtype=jnp.float32)

    tx = init_optimizer(config=config)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dynamic_scale=dynamic_scale,
        loss_count_hist=loss_count_hist,
        loss_sq_hist=loss_sq_hist,
    )
    return state
