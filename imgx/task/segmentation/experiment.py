"""Module for launching experiments."""
from __future__ import annotations

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
from imgx.data import AugmentationFn
from imgx.data.affine import get_random_affine_augmentation_fn
from imgx.data.augmentation import chain_aug_fns
from imgx.data.patch import (
    batch_patch_grid_mean_aggregate,
    batch_patch_grid_sample,
    get_patch_shape_grid_from_config,
    get_random_patch_fn,
)
from imgx.data.util import unpad
from imgx.device import bind_rng_to_host_or_device, get_first_replica_values, unshard
from imgx.experiment import Experiment
from imgx.loss.segmentation import segmentation_loss
from imgx.metric.segmentation import get_segmentation_metrics
from imgx.metric.util import aggregate_metrics, aggregate_pmap_metrics
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
from imgx_datasets.constant import IMAGE, LABEL, TEST_SPLIT, UID, VALID_SPLIT
from imgx_datasets.dataset_info import DatasetInfo


def initialized(key: jax.Array, batch: chex.ArrayTree, model: nn.Module) -> chex.ArrayTree:
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

    variables = jax.jit(init, backend="cpu")({"params": key}, batch[IMAGE])
    return variables["params"]


def get_loss_step(
    train_state: TrainState,
    dataset_info: DatasetInfo,
    config: DictConfig,
) -> Callable[
    [chex.ArrayTree, chex.ArrayTree],
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
        params: chex.ArrayTree, batch: chex.ArrayTree
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, chex.ArrayTree]]:
        """Apply forward and calculate loss."""
        logits = train_state.apply_fn(
            {"params": params},
            batch[IMAGE],
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
    batch: chex.ArrayTree,
    key: jax.Array,
    aug_fn: Callable[[jax.Array, chex.ArrayTree], chex.ArrayTree],
    dataset_info: DatasetInfo,
    config: DictConfig,
) -> tuple[TrainState, jax.Array, chex.ArrayTree]:
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
        - new random key.
        - metric dict.
    """
    # define loss step
    loss_step = get_loss_step(
        train_state=train_state,
        dataset_info=dataset_info,
        config=config,
    )

    # augment, calculate gradients, update train state
    aug_key, new_key = jax.random.split(key)
    aug_key = bind_rng_to_host_or_device(aug_key, bind_to="device", axis_name=REPLICA_AXIS)
    batch = aug_fn(aug_key, batch)
    dynamic_scale, is_fin, aux, grads = get_gradients(
        train_state, loss_step, input_dict={"batch": batch}
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
    return new_state, new_key, metrics


def eval_step(
    train_state: TrainState,
    batch: chex.ArrayTree,
    patch_start_indices: np.ndarray,
    patch_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Perform an evaluation step.

    Args:
        train_state: training state.
        batch: training data without shard axis.
        patch_start_indices: patch start indices.
        patch_shape: patch shape.

    Returns:
        logits, shape starts with (batch, ...).
    """
    image_shape = batch[IMAGE].shape[1:-1]
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
        # (batch, *spatial_shape, num_classes)
        logits_i = train_state.apply_fn(
            {"params": train_state.params},
            image_patches[:, i, ...],
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
        self, ckpt_dir: Path | None = None, step: int | None = None
    ) -> tuple[TrainState, int]:
        """Initialize data loader, loss, networks for training.

        Args:
            ckpt_dir: checkpoint directory to restore from.
            step: checkpoint step to restore from, if None use the latest one.

        Returns:
            initialized training state.
        """
        # the batch is for multi-devices
        # (num_models, ...)
        # num_models is not the same as num_devices_per_replica
        batch = next(self.train_iter)
        batch = get_first_replica_values(batch)

        # check image size
        image_shape = self.dataset_info.image_spatial_shape
        chex.assert_equal(batch[IMAGE].shape[1:-1], image_shape)

        # data augmentation
        aug_fns: list[AugmentationFn] = [
            get_random_affine_augmentation_fn(self.config),
            get_random_patch_fn(self.config),
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

        patch_shape, patch_start_indices = get_patch_shape_grid_from_config(self.config)
        self.p_train_step = jax.pmap(
            partial(
                train_step,
                aug_fn=aug_fn,
                dataset_info=self.dataset_info,
                config=self.config,
            ),
            axis_name=REPLICA_AXIS,
        )
        self.p_eval_step = jax.pmap(
            partial(
                eval_step,
                patch_start_indices=patch_start_indices,
                patch_shape=patch_shape,
            ),
            axis_name=REPLICA_AXIS,
        )

        return train_state, step_offset

    def eval_step(  # pylint:disable=too-many-statements
        self,
        train_state: TrainState,
        key: jax.Array,
        split: str,
        out_dir: Path | None = None,
    ) -> tuple[jax.Array, chex.ArrayTree]:
        """Evaluation on entire validation data set.

        Args:
            train_state: training state.
            key: random key, not used.
            split: split to evaluate.
            out_dir: output directory, if not None, predictions will be saved.

        Returns:
            random key.
            metric dict.

        Raises:
            ValueError: if split is not supported.
        """
        if split == VALID_SPLIT:
            num_steps = self.dataset.num_valid_steps
            split_iter = self.valid_iter
        elif split == TEST_SPLIT:
            num_steps = self.dataset.num_test_steps
            split_iter = self.test_iter
        else:
            raise ValueError(f"Split {split} not supported for evaluation.")

        device_cpu = jax.devices("cpu")[0]
        num_samples = 0
        lst_metrics = []
        lst_uids = []
        for _ in tqdm(range(num_steps), total=num_steps):
            batch = next(split_iter)

            # get uids
            uids = batch.pop(UID)
            uids = uids.reshape(-1)  # remove shard axis
            uids = decode_uids(uids)

            # evaluate the batch
            metrics, label_pred, key = self.eval_batch(
                train_state=train_state,
                key=key,
                batch=batch,
                uids=uids,
                device_cpu=device_cpu,
                out_dir=out_dir,
            )
            num_samples_in_batch = label_pred.shape[0]
            lst_uids += uids[:num_samples_in_batch]
            num_samples += num_samples_in_batch
            lst_metrics.append(metrics)

        # concatenate metrics across all samples
        # metrics, values of shape (num_samples,)
        metrics = jax.tree_map(lambda *args: jnp.concatenate(args), *lst_metrics)

        # aggregate metrics
        agg_metrics = aggregate_pmap_metrics(metrics)
        agg_metrics = jax.tree_map(lambda x: x.item(), agg_metrics)
        agg_metrics["num_samples"] = num_samples

        # save a csv file with per-sample metrics
        if out_dir is not None:
            metrics = jax.tree_map(lambda x: np.asarray(x).tolist(), metrics)
            df_metric = pd.DataFrame(metrics)
            df_metric[UID] = lst_uids
            df_metric.to_csv(out_dir / "metrics_per_sample.csv", index=False)

        return key, agg_metrics

    def eval_batch(
        self,
        train_state: TrainState,
        key: jax.Array,
        batch: dict[str, jnp.ndarray],
        uids: list[str],
        device_cpu: jax.Device,
        out_dir: Path | None,
        reference_suffix: str = "mask_preprocessed",
        output_suffix: str = "mask_pred",
    ) -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jax.Array]:
        """Evaluate a batch.

        Args:
            train_state: training state.
            key: random key.
            batch: batch data without uid.
            uids: uids in the batch.
            device_cpu: cpu device.
            out_dir: output directory, if not None, predictions will be saved.
            reference_suffix: suffix of reference image.
            output_suffix: suffix of output image.

        Returns:
            metrics, each item has shape (num_shards*batch,).
            label_pred: predicted label, of shape (num_shards*batch, *spatial_shape).
            key: random key.
        """
        # logits (num_shards*batch, *spatial_shape, num_classes)
        # label (num_shards*batch, *spatial_shape)
        logits = self.p_eval_step(train_state, batch)
        logits = unshard(logits, device=device_cpu)
        label = unshard(batch[LABEL], device=device_cpu)

        # remove padded examples
        if "" in uids:
            num_samples_in_batch = uids.index("")
            uids = uids[:num_samples_in_batch]
            logits = unpad(logits, num_samples_in_batch)
            label = unpad(label, num_samples_in_batch)

        # (batch,) per metric
        metrics, label_pred = get_segmentation_metrics(
            logits=logits,
            label_pred=None,
            label_true=label,
            dataset_info=self.dataset_info,
        )

        if out_dir is None:
            return metrics, label_pred, key

        save_segmentation_prediction(
            preds=np.array(label_pred, dtype=int),
            uids=uids,
            out_dir=out_dir,
            tfds_dir=self.dataset_info.tfds_preprocessed_dir,
            reference_suffix=reference_suffix,
            output_suffix=output_suffix,
        )
        return metrics, label_pred, key
