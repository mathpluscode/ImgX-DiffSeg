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
from jax import lax
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
from imgx.diffusion.time_sampler import TimeSampler
from imgx.experiment import Experiment
from imgx.metric.segmentation import get_segmentation_metrics_per_step
from imgx.metric.util import merge_aggregated_metrics
from imgx.task.diffusion_segmentation.diffusion import DiffusionSegmentation
from imgx.task.diffusion_segmentation.diffusion_step import get_diffusion_loss_step
from imgx.task.diffusion_segmentation.gaussian_diffusion import (
    DDIMSegmentationSampler,
    DDPMSegmentationSampler,
    GaussianDiffusionSegmentation,
)
from imgx.task.diffusion_segmentation.recycling_step import get_recycling_loss_step
from imgx.task.diffusion_segmentation.save import save_diffusion_segmentation_prediction
from imgx.task.diffusion_segmentation.self_conditioning_step import get_self_conditioning_loss_step
from imgx.task.diffusion_segmentation.train_state import TrainState, create_train_state
from imgx.task.util import decode_uids
from imgx.train_state import (
    get_gradients,
    get_half_precision_dtype,
    get_optimization_metrics,
    restore_checkpoint,
    update_train_state,
)


def initialized(
    key: jax.Array,
    batch: dict[str, jnp.ndarray],
    model: nn.Module,
    dataset_info: DatasetInfo,
    self_conditioning: bool,
) -> chex.ArrayTree:
    """Initialize model parameters and batch statistics.

    Args:
        key: random key.
        batch: batch data for determining input shapes.
        model: model.
        dataset_info: dataset info to transform label to mask.
        self_conditioning: whether to use self conditioning.

    Returns:
        model parameters.
    """

    def init(*args) -> chex.ArrayTree:  # type: ignore[no-untyped-def]
        return model.init(*args)

    image = batch[IMAGE]
    label = batch[LABEL]
    batch_size = image.shape[0]
    mask = dataset_info.label_to_mask(label, axis=-1, dtype=image.dtype)
    if self_conditioning:
        mask = jnp.concatenate([mask, jnp.zeros_like(mask)], axis=-1)
    t = jnp.zeros((batch_size,), dtype=image.dtype)
    variables = jax.jit(init, backend="cpu", static_argnums=(1,))(
        {"params": key}, False, image, mask, t  # is_train
    )
    return variables["params"]


def get_importance_sampling_metrics(
    loss_count_hist: jnp.ndarray,
    loss_sq_hist: jnp.ndarray,
    time_sampler: TimeSampler,
) -> dict[str, jnp.ndarray]:
    """Get importance sampling metrics.

    Args:
        loss_count_hist: count of time steps, shape (num_timesteps, ).
        loss_sq_hist: weighted average of squared loss, shape (num_timesteps, ).
        time_sampler: time sampler for training.

    Returns:
        metrics: metrics dict.
    """
    metrics = {}
    probs = time_sampler.t_probs_from_loss_sq(loss_sq_hist)
    entropy = -jnp.sum(probs * jnp.log(probs))
    metrics["loss_hist_entropy"] = entropy
    metrics["mean_loss_count_hist"] = jnp.mean(loss_count_hist)
    metrics["median_loss_count_hist"] = jnp.median(loss_count_hist)
    metrics["min_loss_count_hist"] = jnp.min(loss_count_hist)
    metrics["max_loss_count_hist"] = jnp.max(loss_count_hist)
    metrics["mean_loss_sq_hist"] = jnp.mean(loss_sq_hist)
    metrics["min_loss_sq_hist"] = jnp.min(loss_sq_hist)
    metrics["max_loss_sq_hist"] = jnp.max(loss_sq_hist)
    return metrics


def sample_logits_progressive(
    train_state: TrainState,
    image: jnp.ndarray,
    key: jax.Array,
    dataset_info: DatasetInfo,
    diffusion_model: DiffusionSegmentation,
    self_conditioning: bool,
) -> Iterator[jnp.ndarray]:
    """Generate segmentation mask logits conditioned on image.

    The noise here is defined on segmentation mask.

    Args:
        train_state: training state.
        image: image, (batch, ..., in_channels).
        key: random key.
        dataset_info: dataset info to transform label to mask.
        diffusion_model: segmentation diffusion model.
        self_conditioning: whether to use self conditioning.

    Yields:
        logits, (batch, ..., num_classes)
    """
    batch_size = image.shape[0]
    noise_shape = (*image.shape[:-1], dataset_info.num_classes)
    key_noise, key = jax.random.split(key=key)
    x_t = diffusion_model.sample_noise(key=key_noise, shape=noise_shape, dtype=image.dtype)
    mask_pred = jnp.zeros_like(diffusion_model.x_to_mask(x_t))
    for t_index_scalar in reversed(range(diffusion_model.num_timesteps)):
        key_t, key = jax.random.split(key=key)
        t_index = jnp.full((batch_size,), t_index_scalar, dtype=jnp.int32)
        t = diffusion_model.t_index_to_t(t_index)
        mask_t = diffusion_model.x_to_mask(x_t)
        if self_conditioning:
            mask_t = jnp.concatenate([mask_t, mask_pred], axis=-1)
        model_out = train_state.apply_fn(
            {"params": train_state.params},
            False,  # is_train
            image,
            mask_t,
            t,
        )
        x_t, x_start = diffusion_model.sample(
            key=key_t,
            model_out=model_out,
            x_t=x_t,
            t_index=t_index,
        )
        mask_pred = diffusion_model.x_to_mask(x_start)
        yield diffusion_model.x_to_logits(x_start)


def train_step(
    train_state: TrainState,
    batch: dict[str, jnp.ndarray],
    key: jax.Array,
    aug_fn: Callable[[jax.Array, chex.ArrayTree], chex.ArrayTree],
    dataset_info: DatasetInfo,
    config: DictConfig,
    diffusion_model: DiffusionSegmentation,
    time_sampler: TimeSampler,
) -> tuple[TrainState, chex.ArrayTree]:
    """Perform a training step.

    Args:
        train_state: training state.
        batch: training data.
        key: random key.
        aug_fn: data augmentation function.
        dataset_info: dataset info with helper functions.
        config: entire config.
        diffusion_model: segmentation diffusion model.
        time_sampler: time sampler for training.

    Returns:
        - new training state.
        - metric dict.
    """
    if config.task.recycling.use and config.task.self_conditioning.use:
        raise ValueError("recycling and self-conditioning cannot be used together.")

    if config.task.recycling.use:
        loss_step = get_recycling_loss_step(
            train_state=train_state,
            dataset_info=dataset_info,
            diffusion_model=diffusion_model,
            time_sampler=time_sampler,
            loss_config=config.task.loss,
            prev_step=config.task.recycling.prev_step,
            reverse_step=config.task.recycling.reverse_step,
        )
    elif config.task.self_conditioning.use:
        loss_step = get_self_conditioning_loss_step(
            train_state=train_state,
            dataset_info=dataset_info,
            diffusion_model=diffusion_model,
            time_sampler=time_sampler,
            loss_config=config.task.loss,
            prev_step=config.task.self_conditioning.prev_step,
            probability=config.task.self_conditioning.probability,
        )
    else:
        loss_step = get_diffusion_loss_step(
            train_state=train_state,
            dataset_info=dataset_info,
            diffusion_model=diffusion_model,
            time_sampler=time_sampler,
            loss_config=config.task.loss,
        )

    # augment, calculate gradients, update train state
    key = bind_rng_to_host_or_device(key, bind_to="device", axis_name=REPLICA_AXIS)
    key = jax.random.fold_in(key=key, data=train_state.step)
    key_aug, key_loss = jax.random.split(key)
    batch = aug_fn(key_aug, batch)
    dynamic_scale, is_fin, aux, grads = get_gradients(
        train_state,
        loss_step,
        input_dict={"batch": batch, "key": key_loss},
    )
    new_state = update_train_state(train_state, dynamic_scale, is_fin, grads)

    # values in metrics are scalars
    _, metrics, loss_batch, t_index = aux[1]
    # sync loss_batch, t_index across replicas
    # then update loss history
    # (num_shards*batch, )
    loss_batch = lax.all_gather(loss_batch, axis_name=REPLICA_AXIS).reshape((-1,))
    t_index = lax.all_gather(t_index, axis_name=REPLICA_AXIS).reshape((-1,))
    loss_count_hist, loss_sq_hist = time_sampler.update_stats(
        loss_batch=loss_batch,
        t_index=t_index,
        loss_count_hist=train_state.loss_count_hist,
        loss_sq_hist=train_state.loss_sq_hist,
    )
    new_state = new_state.replace(
        loss_count_hist=loss_count_hist,
        loss_sq_hist=loss_sq_hist,
    )
    metrics_hist = get_importance_sampling_metrics(loss_count_hist, loss_sq_hist, time_sampler)
    metrics = {
        **metrics,
        **metrics_hist,
    }

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
    key: jax.Array,
    dataset_info: DatasetInfo,
    config: DictConfig,
    diffusion_model: DiffusionSegmentation,
) -> jnp.ndarray:
    """Perform an evaluation step.

    Args:
        train_state: training state.
        batch: training data without shard axis.
        key: random key.
        dataset_info: dataset info with helper functions.
        config: entire config.
        diffusion_model: segmentation diffusion model.

    Returns:
        logits, shape starts with (batch, ..., num_timesteps).
    """
    key = bind_rng_to_host_or_device(key, bind_to="device", axis_name=REPLICA_AXIS)

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
        # (batch, *spatial_shape, num_classes, num_timesteps)
        key_i, key = jax.random.split(key)
        image_i = rescale_intensity(
            image_patches[:, i, ...],
            v_min=config.data.loader.data_augmentation.v_min,
            v_max=config.data.loader.data_augmentation.v_max,
        )
        lst_logits_i = list(
            sample_logits_progressive(
                train_state=train_state,
                image=image_i,
                key=key_i,
                dataset_info=dataset_info,
                diffusion_model=diffusion_model,
                self_conditioning=config.task.self_conditioning.use,
            )
        )
        logits_i = jnp.stack(lst_logits_i, axis=-1)
        logits_patches.append(logits_i)
    # (batch, num_patches, *patch_shape, num_classes, num_timesteps)
    logits = jnp.stack(logits_patches, axis=1)

    # aggregate patch logits
    # (batch, *image_shape, num_classes, num_timesteps)
    logits = batch_patch_grid_mean_aggregate(
        x_patch=logits,
        start_indices=patch_start_indices,
        image_shape=image_shape,
    )
    return logits


class DiffusionSegmentationExperiment(Experiment):
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
                initialized=partial(
                    initialized,
                    dataset_info=self.dataset_info,
                    self_conditioning=self.config.task.self_conditioning.use,
                ),
            )
        # resume training
        if ckpt_dir is not None:
            train_state = restore_checkpoint(state=train_state, ckpt_dir=ckpt_dir, step=step)
        # step_offset > 0 if restarting from checkpoint
        step_offset = int(train_state.step)
        train_state = jax_utils.replicate(train_state)

        # training only
        train_diffusion_model = GaussianDiffusionSegmentation.create(
            classes_are_exclusive=self.dataset_info.classes_are_exclusive,
            num_timesteps=self.config.task.diffusion.num_timesteps,
            num_timesteps_beta=self.config.task.diffusion.num_timesteps_beta,
            beta_schedule=self.config.task.diffusion.beta_schedule,
            beta_start=self.config.task.diffusion.beta_start,
            beta_end=self.config.task.diffusion.beta_end,
            model_out_type=self.config.task.diffusion.model_out_type,
            model_var_type=self.config.task.diffusion.model_var_type,
        )
        time_sampler = TimeSampler(
            num_timesteps=self.config.task.diffusion.num_timesteps,
            uniform_time_sampling=self.config.task.uniform_time_sampling,
        )

        # evaluation only
        if self.config.task.sampler.name == "DDPM":
            eval_diffusion_model_cls = DDPMSegmentationSampler
        elif self.config.task.sampler.name == "DDIM":
            eval_diffusion_model_cls = DDIMSegmentationSampler  # type: ignore[assignment]
        else:
            raise ValueError(f"Sampler {self.config.task.diffusion.sampler.name} not supported.")
        eval_diffusion_model = eval_diffusion_model_cls.create(
            classes_are_exclusive=self.dataset_info.classes_are_exclusive,
            num_timesteps=self.config.task.sampler.num_inference_timesteps,  # different from train
            num_timesteps_beta=self.config.task.diffusion.num_timesteps_beta,
            beta_schedule=self.config.task.diffusion.beta_schedule,
            beta_start=self.config.task.diffusion.beta_start,
            beta_end=self.config.task.diffusion.beta_end,
            model_out_type=self.config.task.diffusion.model_out_type,
            model_var_type=self.config.task.diffusion.model_var_type,
        )

        self.p_train_step = jax.pmap(
            partial(
                train_step,
                aug_fn=aug_fn,
                dataset_info=self.dataset_info,
                config=self.config,
                diffusion_model=train_diffusion_model,
                time_sampler=time_sampler,
            ),
            axis_name=REPLICA_AXIS,
            donate_argnums=(0,),
        )
        self.p_eval_step = jax.pmap(
            partial(
                eval_step,
                dataset_info=self.dataset_info,
                config=self.config,
                diffusion_model=eval_diffusion_model,
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
            key: random key.
            out_dir: output directory, if not None, predictions will be saved.

        Returns:
            metric dict.
        """
        device_cpu = jax.devices("cpu")[0]
        lst_metrics = []
        lst_uids = []
        for i in tqdm(range(num_steps), total=num_steps):
            key_batch = jax.vmap(jax.random.fold_in, in_axes=(0, None))(key, i)
            batch = next(iterator)

            # get uids
            uids = batch.pop(UID)
            uids = uids.reshape(-1)  # remove shard axis
            uids = decode_uids(uids)

            # evaluate the batch
            uids, metrics, preds = self.eval_batch(
                train_state=train_state,
                key=key_batch,
                batch=batch,
                uids=uids,
                device_cpu=device_cpu,
            )
            save_diffusion_segmentation_prediction(
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
        key: jax.Array,
        batch: dict[str, jnp.ndarray],
        uids: list[str],
        device_cpu: jax.Device,
    ) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Evaluate a batch.

        Args:
            train_state: training state.
            key: random key.
            batch: batch data without uid.
            uids: uids in the batch.
            device_cpu: cpu device.

        Returns:
            uids: uids in the batch, excluding padded samples.
            metrics: each item has shape (num_samples,).
            prediction dict: each item has shape (num_samples, ...).
        """
        # logits (num_shards*batch, *spatial_shape, num_classes)
        # label (num_shards*batch, *spatial_shape)
        logits = self.p_eval_step(train_state, batch, key)
        logits = unshard(logits, device=device_cpu)
        label = unshard(batch[LABEL], device=device_cpu)

        # remove padded examples
        num_samples_in_batch = len(uids)
        if "" in uids:
            num_samples_in_batch = uids.index("")
            uids = uids[:num_samples_in_batch]
            logits = unpad(logits, num_samples_in_batch)
            label = unpad(label, num_samples_in_batch)

        # (batch,) per metric
        metrics, label_pred = get_segmentation_metrics_per_step(
            logits=logits,
            label=label,
            dataset_info=self.dataset_info,
        )

        # change to numpy array
        metrics = jax.tree_map(np.asarray, metrics)
        label_pred = np.asarray(label_pred, dtype=np.int8)
        preds = {LABEL_PRED: label_pred}
        return uids, metrics, preds
