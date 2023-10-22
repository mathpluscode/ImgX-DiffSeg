"""Training state and checkpoints.

https://github.com/google/flax/blob/main/examples/imagenet/train.py
"""
from __future__ import annotations

from typing import Callable

import chex
import flax.linen as nn
import jax
from absl import logging
from flax.training import dynamic_scale as dynamic_scale_lib
from omegaconf import DictConfig

from imgx.optim import init_optimizer
from imgx.train_state import TrainState


def create_train_state(
    key: jax.random.PRNGKeyArray,
    batch: chex.ArrayTree,
    model: nn.Module,
    config: DictConfig,
    initialized: Callable[[jax.random.PRNGKeyArray, chex.ArrayTree, nn.Module], chex.ArrayTree],
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

    tx = init_optimizer(config=config)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dynamic_scale=dynamic_scale,
    )
    return state