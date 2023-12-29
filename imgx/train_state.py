"""Training state and checkpoints.

https://github.com/google/flax/blob/main/examples/imagenet/train.py
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax
from absl import logging
from flax import linen as nn
from flax.training import checkpoints
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state as ts
from jax import lax
from omegaconf import DictConfig

from imgx import REPLICA_AXIS


class TrainState(ts.TrainState):
    """Train state.

    If using nn.BatchNorm, batch_stats needs to be tracked.
    https://flax.readthedocs.io/en/latest/guides/batch_norm.html
    https://github.com/google/flax/blob/main/examples/imagenet/train.py
    """

    dynamic_scale: dynamic_scale_lib.DynamicScale


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

    tx = init_optimizer(config=config)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dynamic_scale=dynamic_scale,
    )
    return state


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


def get_lr_schedule(config: DictConfig) -> optax.Schedule:
    """Get learning rate scheduler.

    Args:
        config: entire configuration.

    Returns:
        Scheduler
    """
    return optax.warmup_cosine_decay_schedule(**config.optimizer.lr_schedule)


def get_every_k_schedule(config: DictConfig) -> int:
    """Get k for gradient accumulations.

    Args:
        config: entire configuration.

    Returns:
        k, where gradients are accumulated every k step.
    """
    num_devices_per_replica = config.data.trainer.num_devices_per_replica
    batch_size_per_replica = config.data.trainer.batch_size_per_replica
    num_replicas = jax.local_device_count() // num_devices_per_replica
    batch_size_per_step = batch_size_per_replica * num_replicas
    if config.data.trainer.batch_size < batch_size_per_step:
        raise ValueError(
            f"Batch size {config.data.trainer.batch_size} is too small. "
            f"batch_size_per_replica * num_replicas = "
            f"{batch_size_per_replica} * {num_replicas} = "
            f"{batch_size_per_step}."
        )
    if config.data.trainer.batch_size % batch_size_per_step != 0:
        raise ValueError("Batch size cannot be evenly divided by batch size per step.")
    every_k_schedule = config.data.trainer.batch_size // batch_size_per_step
    if every_k_schedule > 1:
        logging.info(
            f"Using gradient accumulation. "
            f"Each model duplicate is stored across {num_devices_per_replica} "
            f"shard{'s' if num_devices_per_replica > 1 else ''}. "
            f"Each step has {batch_size_per_step} samples. "
            f"Gradients are averaged every {every_k_schedule} steps. "
            f"Effective batch size is {config.data.trainer.batch_size}."
        )
    return every_k_schedule


def init_optimizer(
    config: DictConfig,
) -> optax.GradientTransformation:
    """Initialize optimizer.

    Args:
        config: entire configuration.

    Returns:
        optimizer.
    """
    lr_schedule = get_lr_schedule(config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.optimizer.grad_norm),
        getattr(optax, config.optimizer.name)(learning_rate=lr_schedule, **config.optimizer.kwargs),
    )
    # accumulate gradient when needed
    every_k_schedule = get_every_k_schedule(config)
    if every_k_schedule == 1:
        # no need to accumulate gradient
        return optimizer
    return optax.MultiSteps(optimizer, every_k_schedule=every_k_schedule)


def get_gradients(
    train_state: TrainState,
    loss_step: Callable[[chex.ArrayTree, chex.ArrayTree], tuple[jnp.ndarray, chex.ArrayTree]]
    | Callable[[chex.ArrayTree, chex.ArrayTree, jax.Array], tuple[jnp.ndarray, chex.ArrayTree]],
    input_dict: dict[str, chex.ArrayTree],
) -> tuple[dynamic_scale_lib.DynamicScale, jnp.ndarray, chex.ArrayTree, chex.ArrayTree]:
    """Get gradients.

    Args:
        train_state: training state.
        loss_step: loss step function.
        input_dict: input to loss_step in additional to params.

    Returns:
        dynamic_scale: dynamic scale.
        is_fin: whether the gradients are finite.
        aux: auxiliary outputs from loss_step.
        grads: gradients.
    """
    is_fin = None
    dynamic_scale = train_state.dynamic_scale
    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_step, has_aux=True, axis_name=REPLICA_AXIS)
        dynamic_scale, is_fin, aux, grads = grad_fn(train_state.params, **input_dict)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_step, has_aux=True)
        aux, grads = grad_fn(train_state.params, **input_dict)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        grads = lax.pmean(grads, axis_name=REPLICA_AXIS)
    return dynamic_scale, is_fin, aux, grads


def update_train_state(
    train_state: TrainState,
    dynamic_scale: dynamic_scale_lib.DynamicScale,
    is_fin: jnp.ndarray,
    grads: chex.ArrayTree,
) -> TrainState:
    """Update training state.

    Args:
        train_state: training state.
        dynamic_scale: dynamic scale.
        is_fin: whether the gradients are finite.
        grads: gradients.

    Returns:
        new training state.
    """
    new_state = train_state.apply_gradients(grads=grads)
    if dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # params should be restored (= skip this step).
        new_state = new_state.replace(
            opt_state=jax.tree_util.tree_map(
                partial(jnp.where, is_fin),
                new_state.opt_state,
                train_state.opt_state,
            ),
            params=jax.tree_util.tree_map(
                partial(jnp.where, is_fin), new_state.params, train_state.params
            ),
            dynamic_scale=dynamic_scale,
        )
    return new_state


def get_optimization_metrics(
    grads: chex.ArrayTree,
    train_state: TrainState,
    config: DictConfig,
) -> dict[str, float]:
    """Get optimization metrics.

    Args:
        grads: gradients.
        train_state: training state.
        config: entire configuration.

    Returns:
        metrics.
    """
    metrics = {
        "grad_norm": optax.global_norm(grads),
        "params_norm": optax.global_norm(train_state.params),
    }
    if train_state.dynamic_scale:
        metrics["scale"] = train_state.dynamic_scale.scale

    lr_schedule = get_lr_schedule(config)
    every_k_schedule = get_every_k_schedule(config)
    metrics["lr"] = lr_schedule(train_state.step // every_k_schedule)

    return metrics


def get_half_precision_dtype(half_precision: bool) -> jnp.dtype:
    """Get half precision dtype.

    Args:
        half_precision: whether to use half precision.

    Returns:
        dtype.
    """
    if not half_precision:
        return jnp.float32
    platform = jax.local_devices()[0].platform
    return jnp.bfloat16 if platform == "tpu" else jnp.float16
