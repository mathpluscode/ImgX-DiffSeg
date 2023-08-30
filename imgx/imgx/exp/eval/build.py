"""Factory functions."""
from __future__ import annotations

import json
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Callable

import haiku as hk
import jax
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from jax import numpy as jnp
from omegaconf import DictConfig
from tqdm import tqdm

from imgx import REPLICA_AXIS
from imgx.data.patch import (
    batch_patch_grid_mean_aggregate,
    batch_patch_grid_sample,
    get_patch_shape_grid_from_config,
)
from imgx.data.util import unpad
from imgx.device import unshard
from imgx.exp.eval.diff_seg_eval import (
    batch_diffusion_segmentation_evaluation,
    batch_diffusion_segmentation_inference,
)
from imgx.exp.eval.seg_eval import (
    batch_segmentation_evaluation,
    batch_segmentation_inference,
)
from imgx.exp.mixed_precision import set_mixed_precision_policy
from imgx_datasets import INFO_MAP
from imgx_datasets.constant import IMAGE, LABEL, UID
from imgx_datasets.dataset_info import DatasetInfo
from imgx_datasets.image_io import save_segmentation_prediction


def build_batch_inference_fn(
    image: jnp.ndarray,
    config: DictConfig,
) -> jnp.ndarray:
    """Build model from config.

    Args:
        image: (batch, *image_shape, num_channels).
        config: entire config.

    Returns:
        logits: (batch, *image_shape, num_classes) for vanilla segmentation,
            and (batch, *image_shape, num_classes, num_timesteps_sample)
                for diffusion segmentation.

    Raises:
        ValueError: if config is wrong or not supported.
    """
    model_cls_name = (
        config.task.model._target_  # pylint: disable=protected-access
    )
    model_cls_name = model_cls_name.split(".")[-1]
    set_mixed_precision_policy(
        use_mp=config.mixed_precision, model_cls_name=model_cls_name
    )

    # build model
    dataset_info = INFO_MAP[config.data.name]
    num_classes = dataset_info.num_classes
    task_config = config.task

    if task_config.name == "segmentation":
        # (batch, *image_shape, num_classes)
        vision_model = instantiate(config.task.model)
        return batch_segmentation_inference(image=image, model=vision_model)
    if task_config.name == "diffusion_segmentation":
        sampler_name = task_config.sampler.name
        if sampler_name == "DDPM":
            cls = "imgx.diffusion.DDPMSegmentationSampler"
        elif sampler_name == "DDIM":
            cls = "imgx.diffusion.DDIMSegmentationSampler"
        else:
            raise ValueError(f"Unknown sampler {sampler_name}.")
        task_config.diffusion._target_ = cls  # pylint: disable=protected-access

        diffusion_model = instantiate(
            task_config.diffusion,
            model=config.task.model,  # hydra will instantiate this
            classes_are_exclusive=dataset_info.classes_are_exclusive,
        )
        # modify variance schedule
        diffusion_model.reset_beta_schedule(
            num_inference_timesteps=task_config.sampler.num_inference_timesteps
        )

        # (batch, *image_shape, num_classes, num_timesteps_sample)
        return batch_diffusion_segmentation_inference(
            image=image,
            num_classes=num_classes,
            sd=diffusion_model,
            self_conditioning=task_config.self_conditioning.use,
        )
    raise ValueError(f"Unknown task {task_config['name']}.")


def inference_with_patch(
    batch: dict[str, jnp.ndarray],
    inference_pmap: Callable,
    params: hk.Params,
    state: hk.State,
    rng: jnp.ndarray,
    patch_start_indices: np.ndarray,
    patch_shape: tuple[int, ...],
    device: jax.Device,
) -> tuple[list[str | int], jnp.ndarray, jnp.ndarray]:
    """Inference with patch.

    The patching is not done in pmap function to reduce device memory usage.

    Args:
        batch: batch of data.
        inference_pmap: inference function.
        params: network parameters.
        state: network state.
        rng: random number generator.
        patch_start_indices: patch start indices.
        patch_shape: patch shape.
        device: device to store results.

    Returns:
        uids: list of uids.
        logits: of shape (shards*batch, *image_shape, num_classes).
            or (shards*batch, *image_shape, num_classes, num_timesteps).
        label: of shape (shards*batch, *image_shape).
    """
    # get UID and parse to string
    uids = batch[UID]
    uids = uids.reshape(-1)  # remove shard axis
    uids = [
        x.decode("utf-8") if isinstance(x, bytes) else x for x in uids.tolist()
    ]

    # get label
    label = batch[LABEL]  # (shards, batch, *image_shape)
    label = jax.device_put(label, device)
    label = unshard(label)  # (shards*batch, *image_shape)
    image_shape = label.shape[1:]

    # get image and split into patches
    image = batch[IMAGE]  # (shards, batch, *image_shape, num_channels)
    num_patches = patch_start_indices.shape[0]
    # (shards, batch, num_patches, *patch_shape, num_channels)
    image_patches = jax.pmap(
        partial(
            batch_patch_grid_sample,
            start_indices=patch_start_indices,
            patch_shape=patch_shape,
        ),
        axis_name=REPLICA_AXIS,
    )(image)

    # inference per patch
    patch_logits = []
    for i in range(num_patches):
        # (shards, batch, *spatial_shape, num_classes)
        # or (shards, batch, *spatial_shape, num_classes, num_timesteps)
        logits_i, _ = inference_pmap(
            params, state, rng, image_patches[:, :, i, ...]
        )
        # (shards*batch, *spatial_shape, num_classes)
        # or (shards*batch, *spatial_shape, num_classes, num_timesteps)
        logits_i = unshard(logits_i)
        logits_i = jax.device_put(logits_i, device)
        patch_logits.append(logits_i)
    # (shards*batch, num_patches, *patch_shape, num_classes)
    # or (shards*batch, num_patches, *patch_shape, num_classes, num_timesteps)
    logits = jnp.stack(patch_logits, axis=1)

    # aggregate patch logits
    # (shards*batch, *image_shape, num_classes)
    # or (shards*batch, *image_shape, num_classes, num_timesteps)
    logits = batch_patch_grid_mean_aggregate(
        x_patch=logits,
        start_indices=patch_start_indices,
        image_shape=image_shape,
    )
    return uids, label, logits


def dataset_segmentation_evaluation(  # pylint:disable=R0912,R0915
    inference_pmap: Callable,
    params: hk.Params,
    state: hk.State,
    rng: jnp.ndarray,
    batch_iterator: Iterable[dict[str, jnp.ndarray]],
    patch_start_indices: np.ndarray,
    patch_shape: tuple[int, ...],
    num_steps: int,
    is_diffusion: bool,
    dataset_info: DatasetInfo,
    out_dir: Path | None,
    save_predictions: bool,
) -> dict:
    """Get predictions and perform evaluations of a data set.

    Args:
        inference_pmap: forward function that returns logits.
        params: model parameters.
        state: model state, EMA or not.
        rng: random key.
        batch_iterator: iterator of a data set.
        patch_start_indices: start indices of patches.
        patch_shape: shape of patches.
        num_steps: number of steps.
        is_diffusion: the method is diffusion or not.
        dataset_info: dataset information.
        out_dir: output directory for metrics and predictions,
            if None, no files will be saved.
        save_predictions: if True, save predicted masks.

    Returns:
        - predicted values
        - indices
        - metrics
    """
    device_cpu = jax.devices("cpu")[0]

    lst_df_scalar = []
    for _ in tqdm(range(num_steps), total=num_steps):
        batch = next(batch_iterator)  # type: ignore[call-overload]

        # get UID and parse to string
        uids, label, logits = inference_with_patch(
            batch=batch,
            inference_pmap=inference_pmap,
            params=params,
            state=state,
            rng=rng,
            patch_start_indices=patch_start_indices,
            patch_shape=patch_shape,
            device=device_cpu,
        )

        # remove padded examples
        if 0 in uids:
            # the batch was not complete, padded with zero
            num_samples = uids.index(0)
            uids = uids[:num_samples]
            logits = unpad(logits, num_samples)
            label = unpad(label, num_samples)

        if is_diffusion:
            # (batch, num_timesteps_sample) per metric
            label_pred, scalars = batch_diffusion_segmentation_evaluation(
                logits=logits,
                label=label,
                dataset_info=dataset_info,
            )
            # separate metrics per step
            scalars_flatten = {}
            for k, v in scalars.items():
                for i in range(v.shape[-1]):
                    scalars_flatten[f"{k}_step_{i}"] = v[..., i]
                scalars_flatten[k] = v[..., -1]
            scalars = scalars_flatten
        else:
            # (batch, ) per metric
            label_pred, scalars = batch_segmentation_evaluation(
                logits=logits,
                label=label,
                dataset_info=dataset_info,
            )

        # save output
        if save_predictions and (out_dir is not None):
            if is_diffusion:
                for i in range(logits.shape[-1]):
                    label_pred_np = np.array(label_pred[..., i], dtype=int)
                    save_segmentation_prediction(
                        preds=label_pred_np,
                        uids=uids,
                        out_dir=out_dir / f"step_{i}",
                        tfds_dir=dataset_info.tfds_preprocessed_dir,
                    )
            else:
                label_pred_np = np.array(label_pred, dtype=int)
                save_segmentation_prediction(
                    preds=label_pred_np,
                    uids=uids,
                    out_dir=out_dir,
                    tfds_dir=dataset_info.tfds_preprocessed_dir,
                )

        # save metrics
        scalars = jax.tree_map(lambda x: np.asarray(x).tolist(), scalars)
        scalars["uid"] = uids
        lst_df_scalar.append(pd.DataFrame(scalars))

    # assemble metrics
    df_scalar = pd.concat(lst_df_scalar)
    df_scalar = df_scalar.sort_values("uid")
    if out_dir is not None:
        df_scalar.to_csv(out_dir / "metrics_per_sample.csv", index=False)

    # average over samples in the dataset
    scalars = df_scalar.drop("uid", axis=1).mean().to_dict()
    scalars["num_images_in_total"] = len(df_scalar)
    if out_dir is not None:
        with open(out_dir / "mean_metrics.json", "w", encoding="utf-8") as f:
            json.dump(scalars, f, sort_keys=True, indent=4)
    return scalars


def build_dataset_eval_fn(config: DictConfig) -> Callable:
    """Return a function to eval data set.

    Args:
        config: entire config.

    Returns:
        A function.

    Raises:
        ValueError: if data set in unknown.
    """
    task_name = config["task"]["name"]
    if task_name not in ["segmentation", "diffusion_segmentation"]:
        raise ValueError(f"Unknown task {task_name}.")

    is_diffusion = config["task"]["name"] == "diffusion_segmentation"
    dataset_info = INFO_MAP[config.data.name]
    (
        patch_shape,
        patch_start_indices,
    ) = get_patch_shape_grid_from_config(config)

    return partial(
        dataset_segmentation_evaluation,
        patch_start_indices=patch_start_indices,
        patch_shape=patch_shape,
        is_diffusion=is_diffusion,
        dataset_info=dataset_info,
    )
