"""Module for launching experiments."""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Callable

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import optax
import tensorflow as tf
from absl import logging
from omegaconf import DictConfig

from imgx import REPLICA_AXIS
from imgx.data.augmentation import build_aug_fn_from_config
from imgx.data.iterator import get_image_tfds_dataset, py_prefetch
from imgx.device import (
    bind_rng_to_host_or_device,
    broadcast_to_local_devices,
    get_first_replica_values,
    is_tpu,
)
from imgx.exp.eval.build import build_batch_inference_fn, build_dataset_eval_fn
from imgx.exp.loss.build import build_loss_fn
from imgx.exp.mixed_precision import get_mixed_precision_policy, select_tree
from imgx.exp.optim import get_lr_schedule, init_optimizer
from imgx.exp.train_state import TrainState
from imgx_datasets import INFO_MAP
from imgx_datasets.constant import IMAGE, TEST_SPLIT, VALID_SPLIT


def init_train_state(
    batch: chex.ArrayTree,
    rng: jax.random.PRNGKey,
    loss_init: Callable,
    config: DictConfig,
) -> TrainState:
    """Initialize train_state.

    Args:
        batch: a batch example.
        rng: random key.
        loss_init: init function of loss.
        config: entire configuration.
    """
    use_mp = config.mixed_precision

    # init network
    rng, train_rng = jax.random.split(rng)
    params, network_state = loss_init(rng, batch)

    # count params on one device
    params_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logging.info(f"The model has {params_count:,} parameters.")

    # init optimizer state
    optimizer, _ = init_optimizer(config=config)
    opt_state = optimizer.init(params)

    # init loss_scale
    # it is necessary to use NoOpLossScale even not intended to use mp
    # otherwise, some unknown default policy may be used
    # resulted in non-converging losses and nans
    loss_scale = jmp.NoOpLossScale()
    if use_mp and (not is_tpu()):
        # no need to scale on TPU
        # https://cloud.google.com/tpu/docs/bfloat16
        # jmp readme uses "jmp.DynamicLossScale(jmp.half_dtype()(2 ** 15))"
        # but adjust will mix float16 and float32 as min_loss_scale is float32
        # min_loss_scale is forced to float32 by unknown reasons
        # therefore scale here shall be float32 as the gradients are float32
        scale = jnp.float32(2**15)
        loss_scale = jmp.DynamicLossScale(scale)

    global_step = jnp.array(0, dtype=jnp.int32)
    return TrainState(  # type: ignore[call-arg]
        params=params,
        network_state=network_state,
        opt_state=opt_state,
        loss_scale=loss_scale,
        global_step=global_step,
        rng=train_rng,
    )


def update_parameters(
    train_state: TrainState,
    batch: Mapping[str, chex.ArrayTree],
    loss_apply: Callable,
    config: DictConfig,
) -> tuple[TrainState, chex.ArrayTree]:
    """Updates parameters.

    Mixed precision references:
      - https://github.com/deepmind/jmp
      - https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet

    Args:
        train_state: training state.
        batch: training data.
        loss_apply: apply of loss function.
        config: entire configuration.

    Returns:
        train_state: training state.
        scalars: metric dict.
    """

    def loss_fn(
        params: hk.Params,
        network_state: hk.State,
        loss_scale: jmp.LossScale,
        rng_key: jax.random.PRNGKey,
        batch_data: chex.ArrayTree,
    ) -> tuple[chex.ArrayTree, tuple[chex.ArrayTree, hk.State]]:
        """Regroup loss output.

        Args:
            params: network parameters.
            network_state: network state.
            loss_scale: scale loss for mixed precision.
            rng_key: random key.
            batch_data: data of a batch.

        Returns:
            - loss
            - (metric dict, model state)
        """
        (loss, batch_scalars), network_state = loss_apply(
            params, network_state, rng_key, batch_data
        )
        return loss_scale.scale(loss), (batch_scalars, network_state)

    aug_fn = build_aug_fn_from_config(config)

    # get random key for the step
    rng, step_rng = jax.random.split(train_state.rng)
    aug_rng, step_rng = jax.random.split(step_rng)
    aug_rng = bind_rng_to_host_or_device(
        aug_rng, bind_to="device", axis_name=REPLICA_AXIS
    )
    step_rng = bind_rng_to_host_or_device(
        step_rng, bind_to="device", axis_name=REPLICA_AXIS
    )

    # data augmentation
    batch = aug_fn(aug_rng, batch)

    # gradient calculation
    grad_loss_fn = jax.grad(loss_fn, has_aux=True)
    grads, (scalars, updated_network_state) = grad_loss_fn(
        train_state.params,
        train_state.network_state,
        train_state.loss_scale,
        step_rng,
        batch,
    )
    scalars["grad_norm_before_pmean"] = optax.global_norm(grads)
    scalars["params_norm"] = optax.global_norm(train_state.params)

    # grads are in "param_dtype" (likely float32)
    # cast them back to compute dtype such that
    # we do the all-reduce below in the compute precision
    # which is typically lower than the param precision
    policy = get_mixed_precision_policy(config.mixed_precision)
    grads = policy.cast_to_compute(grads)
    grads = train_state.loss_scale.unscale(grads)

    # take the mean across all replicas to keep params in sync
    grads = jax.lax.pmean(grads, axis_name=REPLICA_AXIS)

    # compute our optimizer update in the same precision as params
    grads = policy.cast_to_param(grads)

    # update parameters
    optimizer, every_k_schedule = init_optimizer(config=config)
    updates, updated_opt_state = optimizer.update(
        grads, train_state.opt_state, train_state.params
    )
    updated_params = optax.apply_updates(train_state.params, updates)
    scalars["lr"] = get_lr_schedule(config)(
        train_state.global_step // every_k_schedule
    )
    scalars["grad_norm"] = optax.global_norm(grads)
    scalars["grad_update_norm"] = optax.global_norm(updates)

    grads_finite = jmp.all_finite(grads)
    updated_loss_scale = train_state.loss_scale.adjust(grads_finite)
    # mixed precision or not, skip non-finite gradients
    (updated_params, updated_network_state, updated_opt_state) = select_tree(
        grads_finite,
        (updated_params, updated_network_state, updated_opt_state),
        (
            train_state.params,
            train_state.network_state,
            train_state.opt_state,
        ),
    )
    scalars["loss_scale"] = updated_loss_scale.loss_scale

    # average metrics across replicas
    min_scalars = {}
    max_scalars = {}
    mean_scalars = {}
    for k in scalars:
        if k.startswith("min_"):
            min_scalars[k] = scalars[k]
        elif k.startswith("max_"):
            max_scalars[k] = scalars[k]
        else:
            mean_scalars[k] = scalars[k]
    min_scalars = jax.lax.pmin(min_scalars, axis_name=REPLICA_AXIS)
    max_scalars = jax.lax.pmax(max_scalars, axis_name=REPLICA_AXIS)
    mean_scalars = jax.lax.pmean(mean_scalars, axis_name=REPLICA_AXIS)
    scalars = {
        **min_scalars,
        **max_scalars,
        **mean_scalars,
    }

    # update train_state
    train_state = train_state.replace(
        params=updated_params,
        network_state=updated_network_state,
        opt_state=updated_opt_state,
        loss_scale=updated_loss_scale,
        rng=rng,
        global_step=train_state.global_step + 1,
    )
    return train_state, scalars


class Experiment:
    """Experiment for supervised training."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes experiment.

        Args:
            config: experiment config.
        """
        # Do not use accelerators in data pipeline.
        tf.config.experimental.set_visible_devices([], device_type="GPU")
        tf.config.experimental.set_visible_devices([], device_type="TPU")

        # save args
        self.config = config

        # init data loaders and networks
        self.dataset = get_image_tfds_dataset(
            dataset_name=self.config.data.name,
            config=self.config,
        )
        self.train_iter = py_prefetch(lambda: self.dataset.train_iter)
        self.valid_iter = py_prefetch(lambda: self.dataset.valid_iter)
        self.test_iter = py_prefetch(lambda: self.dataset.test_iter)

    def train_init(self) -> TrainState:
        """Initialize data loader, loss, networks for training.

        Returns:
            initialized training state.
        """
        # init loss
        loss = hk.transform_with_state(build_loss_fn(config=self.config))

        # the batch is for multi-devices
        # (num_models, ...)
        # num_models is not the same as num_devices_per_replica
        batch = next(self.train_iter)
        batch = get_first_replica_values(batch)

        # check image size
        data_config = self.config["data"]
        dataset_info = INFO_MAP[data_config["name"]]
        image_shape = dataset_info.image_spatial_shape
        chex.assert_equal(batch[IMAGE].shape[1:-1], image_shape)

        aug_fn = build_aug_fn_from_config(self.config)
        aug_rng = jax.random.PRNGKey(self.config["seed"])
        batch = aug_fn(aug_rng, batch)

        # init train state on cpu first
        rng = jax.random.PRNGKey(self.config.seed)
        train_state = jax.jit(
            partial(init_train_state, loss_init=loss.init, config=self.config),
            backend="cpu",
        )(
            batch=batch,
            rng=rng,
        )
        # then broadcast train_state to devices
        train_state = broadcast_to_local_devices(train_state)

        # define pmap-ed update func
        self.update_params_pmap = jax.pmap(
            partial(
                update_parameters,
                loss_apply=loss.apply,
                config=self.config,
            ),
            axis_name=REPLICA_AXIS,
            donate_argnums=(0,),
        )

        return train_state

    def train_step(
        self,
        train_state: TrainState,
    ) -> TrainState | chex.ArrayTree:
        """Training step.

        Args:
            train_state: training state.

        Returns:
            - updated train_state.
            - metric dict.
        """
        batch = next(self.train_iter)
        train_state, scalars = self.update_params_pmap(
            train_state,
            batch,
        )
        scalars = get_first_replica_values(scalars)
        scalars = jax.tree_map(lambda x: x.item(), scalars)  # tensor to values
        return train_state, scalars

    def eval_init(self) -> None:
        """Initialize data loader, loss, networks for validation."""
        inference = hk.transform_with_state(
            partial(build_batch_inference_fn, config=self.config)
        )
        self.inference_pmap = jax.pmap(
            inference.apply,
            axis_name=REPLICA_AXIS,
        )
        self.eval_dataset = build_dataset_eval_fn(self.config)

    def eval_step(
        self,
        split: str,
        params: hk.Params,
        state: hk.State,
        rng: jax.random.PRNGKey,
        out_dir: Path | None,
        save_predictions: bool,
    ) -> dict:
        """Validation step on entire validation data set.

        Args:
            split: data split.
            params: network parameters.
            state: network state.
            rng: random key.
            out_dir: output directory to save metrics and predictions,
                if None, no files will be saved.
            save_predictions: if True, save predicted masks.

        Returns:
            metric dict.

        Raises:
            ValueError: if split is not supported.
        """
        if split not in [VALID_SPLIT, TEST_SPLIT]:
            raise ValueError(
                "Evaluation can only be performed on valid and test splits."
            )
        if split == VALID_SPLIT:
            batch_iterator = self.valid_iter
            num_steps = self.dataset.num_valid_steps
        else:
            batch_iterator = self.test_iter
            num_steps = self.dataset.num_test_steps
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
        return self.eval_dataset(
            inference_pmap=self.inference_pmap,
            params=params,
            state=state,
            rng=rng,
            batch_iterator=batch_iterator,
            num_steps=num_steps,
            out_dir=out_dir,
            save_predictions=save_predictions,
        )
