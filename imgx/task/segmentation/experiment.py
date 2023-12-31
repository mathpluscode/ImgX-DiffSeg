"""Module for launching experiments."""
from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Callable

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import jax_utils
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from imgx import REPLICA_AXIS
from imgx.data.augmentation import AugmentationFn, chain_aug_fns
from imgx.data.augmentation.affine import get_random_affine_augmentation_fn
from imgx.data.augmentation.intensity import (
    get_random_gamma_augmentation_fn,
    get_rescale_intensity_fn,
    rescale_intensity,
)
from imgx.data.augmentation.patch import (
    batch_patch_grid_mean_aggregate,
    batch_patch_grid_sample,
    get_patch_grid,
    get_random_patch_fn,
)
from imgx.data.util import unpad
from imgx.datasets.constant import IMAGE, LABEL, LABEL_PRED, UID
from imgx.datasets.dataset_info import DatasetInfo
from imgx.device import bind_rng_to_host_or_device, get_first_replica_values, unshard
from imgx.experiment import Experiment
from imgx.loss.segmentation import segmentation_loss
from imgx.metric.segmentation import get_segmentation_metrics
from imgx.metric.util import aggregate_metrics, merge_aggregated_metrics
from imgx.task.segmentation.save import save_segmentation_prediction
from imgx.task.util import decode_uids
from imgx.train_state import (
    TrainState,
    create_train_state,
    get_gradients,
    get_half_precision_dtype,
    get_optimization_metrics,
    restore_checkpoint,
    update_train_state,
)


def initialized(key: jax.Array, batch: dict[str, jnp.ndarray], model: nn.Module) -> chex.ArrayTree:
    """Initialize model parameters and batch statistics.

    Args:
        key: random key.
        batch: batch data for determining input shapes.
        model: model.

    Returns:
        model parameters.
    """

    def init(*args) -> chex.ArrayTree:  # type: ignore[no-untyped-def]
        return model.init(*args)

    variables = jax.jit(init, backend="cpu", static_argnums=(1,))(
        {"params": key}, False, batch[IMAGE]  # is_train
    )
    return variables["params"]


def get_loss_step(
    train_state: TrainState,
    dataset_info: DatasetInfo,
    config: DictConfig,
) -> Callable[
    [chex.ArrayTree, chex.ArrayTree, jax.Array],
    tuple[jnp.ndarray, tuple[jnp.ndarray, chex.ArrayTree]],
]:
    """Return loss_step.

    Args:
        train_state: train state.
        dataset_info: dataset info to transform label to mask.
        config: entire configuration.

    Returns:
        loss_step: loss step function.

    """

    def loss_step(
        params: chex.ArrayTree,
        batch: dict[str, jnp.ndarray],
        key: jax.Array,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, chex.ArrayTree]]:
        """Apply forward and calculate loss."""
        logits = train_state.apply_fn(
            {"params": params},
            True,  # is_train
            batch[IMAGE],
            rngs={"dropout": key},
        )
        loss_batch, loss_metrics = segmentation_loss(
            logits=logits,
            label=batch[LABEL],
            dataset_info=dataset_info,
            loss_config=config.task.loss,
        )
        loss = jnp.mean(loss_batch)
        loss_metrics = aggregate_metrics(loss_metrics)
        return loss, (logits, loss_metrics)

    return loss_step


def train_step(
    train_state: TrainState,
    batch: dict[str, jnp.ndarray],
    key: jax.Array,
    aug_fn: Callable[[jax.Array, chex.ArrayTree], chex.ArrayTree],
    dataset_info: DatasetInfo,
    config: DictConfig,
) -> tuple[TrainState, chex.ArrayTree]:
    """Perform a training step.

    Args:
        train_state: training state.
        batch: training data.
        key: random key.
        aug_fn: data augmentation function.
        dataset_info: dataset info with helper functions.
        config: entire configuration.

    Returns:
        - new training state.
        - metric dict.
    """
    # define loss step
    loss_step = get_loss_step(
        train_state=train_state,
        dataset_info=dataset_info,
        config=config,
    )

    # augment, calculate gradients, update train state
    key = bind_rng_to_host_or_device(key, bind_to="device", axis_name=REPLICA_AXIS)
    key = jax.random.fold_in(key=key, data=train_state.step)
    key_aug, key_loss = jax.random.split(key)
    batch = aug_fn(key_aug, batch)
    dynamic_scale, is_fin, aux, grads = get_gradients(
        train_state, loss_step, input_dict={"batch": batch, "key": key_loss}
    )
    new_state = update_train_state(train_state, dynamic_scale, is_fin, grads)

    # values in metrics are scalars
    _, metrics = aux[1]

    # add optimization metrics
    metrics_optim = get_optimization_metrics(
        grads=grads,
        train_state=new_state,
        config=config,
    )
    metrics = {**metrics, **metrics_optim}
    return new_state, metrics


def eval_step(
    train_state: TrainState,
    batch: dict[str, jnp.ndarray],
    config: DictConfig,
) -> jnp.ndarray:
    """Perform an evaluation step.

    Args:
        train_state: training state.
        batch: training data without shard axis.
        config: entire config.

    Returns:
        logits, shape starts with (batch, ...).
    """
    patch_shape = tuple(config.data.loader.patch_shape)
    patch_overlap = tuple(config.data.loader.patch_overlap)
    image_shape = batch[IMAGE].shape[1:-1]
    patch_start_indices = get_patch_grid(
        image_shape=image_shape,
        patch_shape=patch_shape,
        patch_overlap=patch_overlap,
    )
    num_patches = patch_start_indices.shape[0]
    # (batch, num_patches, *patch_shape, num_channels)
    image_patches = batch_patch_grid_sample(
        x=batch[IMAGE],
        start_indices=patch_start_indices,
        patch_shape=patch_shape,
    )
    # inference per patch
    logits_patches = []
    for i in range(num_patches):
        image_i = rescale_intensity(
            image_patches[:, i, ...],
            v_min=config.data.loader.data_augmentation.v_min,
            v_max=config.data.loader.data_augmentation.v_max,
        )
        # (batch, *spatial_shape, num_classes)
        logits_i = train_state.apply_fn(
            {"params": train_state.params},
            False,  # is_train
            image_i,
        )
        logits_patches.append(logits_i)
    # (batch, num_patches, *patch_shape, num_classes)
    logits = jnp.stack(logits_patches, axis=1)

    # aggregate patch logits
    # (batch, *image_shape, num_classes)
    logits = batch_patch_grid_mean_aggregate(
        x_patch=logits,
        start_indices=patch_start_indices,
        image_shape=image_shape,
    )
    return logits


class SegmentationExperiment(Experiment):
    """Experiment for supervised training."""

    def train_init(
        self, batch: dict[str, jnp.ndarray], ckpt_dir: Path | None = None, step: int | None = None
    ) -> tuple[TrainState, int]:
        """Initialize data loader, loss, networks for training.

        Args:
            batch: training data for multi-devices.
            ckpt_dir: checkpoint directory to restore from.
            step: checkpoint step to restore from, if None use the latest one.

        Returns:
            initialized training state.
        """
        # the batch is for multi-devices
        # (num_models, ...)
        # num_models is not the same as num_devices_per_replica
        batch = get_first_replica_values(batch)

        # data augmentation
        aug_fns: list[AugmentationFn] = []
        aug_fns += [
            get_random_affine_augmentation_fn(self.config),
            get_random_patch_fn(self.config),
            get_random_gamma_augmentation_fn(self.config),
            get_rescale_intensity_fn(self.config),
        ]
        aug_fn = chain_aug_fns(aug_fns)
        aug_rng = jax.random.PRNGKey(self.config["seed"])
        batch = aug_fn(aug_rng, batch)

        # init train state on cpu first
        dtype = get_half_precision_dtype(self.config.half_precision)
        model = instantiate(self.config.task.model, dtype=dtype)
        with jax.default_device(jax.devices("cpu")[0]):
            train_state = create_train_state(
                key=jax.random.PRNGKey(self.config.seed),
                batch=batch,
                model=model,
                config=self.config,
                initialized=initialized,
            )
        # resume training
        if ckpt_dir is not None:
            train_state = restore_checkpoint(state=train_state, ckpt_dir=ckpt_dir, step=step)
        # step_offset > 0 if restarting from checkpoint
        step_offset = int(train_state.step)
        train_state = jax_utils.replicate(train_state)

        self.p_train_step = jax.pmap(
            partial(
                train_step,
                aug_fn=aug_fn,
                dataset_info=self.dataset_info,
                config=self.config,
            ),
            axis_name=REPLICA_AXIS,
            donate_argnums=(0,),
        )
        self.p_eval_step = jax.pmap(
            partial(
                eval_step,
                config=self.config,
            ),
            axis_name=REPLICA_AXIS,
        )

        return train_state, step_offset

    def eval_step(  # pylint:disable=too-many-statements
        self,
        train_state: TrainState,
        iterator: Iterator[dict[str, jnp.ndarray]],
        num_steps: int,
        key: jax.Array,
        out_dir: Path | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Evaluation on entire validation data set.

        Args:
            train_state: training state.
            iterator: data iterator.
            num_steps: number of steps for evaluation.
            key: random key, (num_shards,).
            out_dir: output directory, if not None, predictions will be saved.

        Returns:
            metric dict.

        Raises:
            ValueError: if split is not supported.
        """
        device_cpu = jax.devices("cpu")[0]
        lst_metrics = []
        lst_uids = []
        for _ in tqdm(range(num_steps), total=num_steps):
            batch = next(iterator)

            # get uids
            uids = batch.pop(UID)
            uids = uids.reshape(-1)  # remove shard axis
            uids = decode_uids(uids)

            # evaluate the batch
            uids, metrics, preds = self.eval_batch(
                train_state=train_state,
                key=key,
                batch=batch,
                uids=uids,
                device_cpu=device_cpu,
            )
            save_segmentation_prediction(
                label_pred=preds[LABEL_PRED],
                uids=uids,
                out_dir=out_dir,
                tfds_dir=self.dataset_info.tfds_preprocessed_dir,
            )

            lst_uids += uids
            lst_metrics.append(metrics)

            # https://github.com/google/jax/issues/10828
            jax.clear_caches()

        # concatenate metrics across all samples
        # metrics, values of shape (num_samples,)
        metrics = jax.tree_map(lambda *args: np.concatenate(args), *lst_metrics)

        # aggregate metrics
        agg_metrics = merge_aggregated_metrics(metrics)
        agg_metrics = jax.tree_map(lambda x: x.item(), agg_metrics)
        agg_metrics["num_samples"] = len(lst_uids)

        # save a csv file with per-sample metrics
        if out_dir is not None:
            metrics = jax.tree_map(lambda x: x.tolist(), metrics)
            df_metric = pd.DataFrame(metrics)
            df_metric[UID] = lst_uids
            df_metric.to_csv(out_dir / "metrics_per_sample.csv", index=False)

        return agg_metrics

    def eval_batch(
        self,
        train_state: TrainState,
        key: jax.Array,  # noqa: ARG002
        batch: dict[str, jnp.ndarray],
        uids: list[str],
        device_cpu: jax.Device,
    ) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Evaluate a batch.

        Args:
            train_state: training state.
            key: random key.
            batch: batch data without uid.
            uids: uids in the batch, potentially including padded samples.
            device_cpu: cpu device.

        Returns:
            uids: uids in the batch, excluding padded samples.
            metrics: each item has shape (num_samples,).
            prediction dict: each item has shape (num_samples, ...).
        """
        logits = self.p_eval_step(train_state, batch)
        # logits (num_shards*batch, *spatial_shape, num_classes)
        # label (num_shards*batch, *spatial_shape)
        logits = unshard(logits, device=device_cpu)
        label = unshard(batch[LABEL], device=device_cpu)

        # remove padded examples
        num_samples_in_batch = len(uids)
        if "" in uids:
            num_samples_in_batch = uids.index("")
            logits = unpad(logits, num_samples_in_batch)
            label = unpad(label, num_samples_in_batch)
            uids = uids[:num_samples_in_batch]

        # (batch,) per metric
        # TODO: when parsing data there may be zero padding, should remove these
        metrics, label_pred = get_segmentation_metrics(
            logits=logits,
            label_pred=None,
            label_true=label,
            dataset_info=self.dataset_info,
        )

        # change to numpy array
        metrics = jax.tree_map(np.asarray, metrics)
        label_pred = np.asarray(label_pred, dtype=np.int8)
        preds = {LABEL_PRED: label_pred}
        return uids, metrics, preds
