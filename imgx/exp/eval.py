"""Module for building evaluation functions."""
import json
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import chex
import haiku as hk
import jax
import numpy as np
import pandas as pd
from jax import numpy as jnp
from omegaconf import DictConfig

from imgx import IMAGE, LABEL, UID
from imgx.datasets import (
    DIR_TFDS_PROCESSED_MAP,
    IMAGE_SPACING_MAP,
    NUM_CLASSES_MAP,
)
from imgx.datasets.preprocess import save_segmentation_prediction
from imgx.datasets.util import unpad
from imgx.device import unshard
from imgx.diffusion.gaussian_diffusion import GaussianDiffusion
from imgx.exp.mixed_precision import set_mixed_precision_policy
from imgx.exp.model import build_diffusion_model, build_vision_model
from imgx.math_util import logits_to_mask
from imgx.metric import (
    aggregated_surface_distance,
    centroid_distance,
    class_proportion,
    dice_score,
    iou,
    normalized_surface_dice_from_distances,
)
from imgx.metric.centroid import get_coordinate_grid


def get_jit_segmentation_metrics(
    mask_pred: jnp.ndarray, mask_true: jnp.ndarray, spacing: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """Calculate segmentation metrics.

    Use nanmean in case some classes do not exist.

    Args:
        mask_true: shape = (batch, ..., num_classes).
        mask_pred: shape = (batch, ..., num_classes).
        spacing: spacing of pixel/voxels along each dimension, (3,).

    Returns:
        Dict of metrics, each value is of shape (batch,).
    """
    chex.assert_equal_shape([mask_pred, mask_true])
    scalars = {}
    # binary dice (batch, num_classes)
    dice_score_bc = dice_score(
        mask_pred=mask_pred,
        mask_true=mask_true,
    )
    for i in range(dice_score_bc.shape[-1]):
        scalars[f"binary_dice_score_class_{i}"] = dice_score_bc[:, i]
    scalars["mean_binary_dice_score"] = jnp.nanmean(dice_score_bc, axis=1)
    scalars["mean_binary_dice_score_without_background"] = jnp.nanmean(
        dice_score_bc[:, 1:], axis=1
    )

    # IoU (batch, num_classes)
    iou_bc = iou(
        mask_pred=mask_pred,
        mask_true=mask_true,
    )
    for i in range(iou_bc.shape[-1]):
        scalars[f"iou_class_{i}"] = iou_bc[:, i]
    scalars["mean_iou"] = jnp.nanmean(iou_bc, axis=1)
    scalars["mean_iou_without_background"] = jnp.nanmean(iou_bc[:, 1:], axis=1)

    # centroid distance (batch, num_classes)
    grid = get_coordinate_grid(shape=mask_pred.shape[1:-1])
    centroid_dist_bc = centroid_distance(
        mask_pred=mask_pred,
        mask_true=mask_true,
        grid=grid,
        spacing=spacing,
    )
    for i in range(centroid_dist_bc.shape[-1]):
        scalars[f"centroid_dist_class_{i}"] = centroid_dist_bc[:, i]
    scalars["mean_centroid_dist"] = jnp.nanmean(centroid_dist_bc, axis=1)
    scalars["mean_centroid_dist_without_background"] = jnp.nanmean(
        centroid_dist_bc[:, 1:], axis=1
    )

    # class proportion (batch, num_classes)
    for mask, mask_name in zip([mask_pred, mask_true], ["pred", "label"]):
        class_prop_bc = class_proportion(mask)
        for i in range(class_prop_bc.shape[-1]):
            scalars[f"class_{i}_proportion_{mask_name}"] = class_prop_bc[:, i]

    return scalars


def get_non_jit_segmentation_metrics(
    mask_pred: jnp.ndarray,
    mask_true: jnp.ndarray,
    spacing: Optional[jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """Calculate non-jittable segmentation metrics for batch.

    Use nanmean in case some classes do not exist.

    Args:
        mask_pred: (batch, w, h, d, num_classes)
        mask_true: (batch, w, h, d, num_classes)
        spacing: spacing of pixel/voxels along each dimension.

    Returns:
        Dict of metrics, each value is of shape (batch,).
    """
    chex.assert_equal_shape([mask_pred, mask_true])
    batch_scalars = {}

    # (3, batch, num_classes)
    # mean surface distance
    # hausdorff distance, 95 percentile
    # normalised surface dice
    sur_dist_bc = aggregated_surface_distance(
        mask_pred=np.array(mask_pred),
        mask_true=np.array(mask_true),
        agg_fns=[
            np.mean,
            partial(np.percentile, q=95),
            normalized_surface_dice_from_distances,
        ],
        num_args=[1, 1, 2],
        spacing=spacing,
    )
    for i in range(sur_dist_bc.shape[-1]):
        batch_scalars[f"mean_surface_dist_class_{i}"] = sur_dist_bc[0, :, i]
        batch_scalars[f"hausdorff_dist_class_{i}"] = sur_dist_bc[1, :, i]
        batch_scalars[f"surface_dice_class_{i}"] = sur_dist_bc[2, :, i]
    batch_scalars["mean_mean_surface_dist"] = np.nanmean(
        sur_dist_bc[0, ...], axis=-1
    )
    batch_scalars["mean_hausdorff_dist"] = np.nanmean(
        sur_dist_bc[1, ...], axis=-1
    )
    batch_scalars["mean_surface_dice"] = np.nanmean(
        sur_dist_bc[2, ...], axis=-1
    )
    batch_scalars["mean_mean_surface_dist_without_background"] = np.nanmean(
        sur_dist_bc[0, :, 1:], axis=-1
    )
    batch_scalars["mean_hausdorff_dist_without_background"] = np.nanmean(
        sur_dist_bc[1, :, 1:], axis=-1
    )
    batch_scalars["mean_surface_dice_without_background"] = np.nanmean(
        sur_dist_bc[2, :, 1:], axis=-1
    )
    return batch_scalars


def batch_segmentation_evaluation(
    input_dict: Dict[str, jnp.ndarray],
    model: hk.Module,
    spacing: jnp.ndarray,
    num_classes: int,
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
    """Evaluate binary predictions.

    Args:
        input_dict: input data having image and label.
        model: network instance.
        spacing: spacing of pixel/voxels along each dimension.
        num_classes: number of classes including background.

    Returns:
        - metrics, each metric value has shape (batch, ).
        - logits.
    """
    # (batch, ..., 1)
    image = jnp.expand_dims(input_dict[IMAGE], axis=-1)
    # (batch, ..., num_classes)
    logits = model(image=image, is_train=False)
    # (batch, ..., num_classes)
    mask_true = jax.nn.one_hot(
        input_dict[LABEL], num_classes=num_classes, axis=-1
    )
    # (batch, ..., num_classes)
    mask_pred = logits_to_mask(logits, axis=-1)
    # evaluate
    scalars = get_jit_segmentation_metrics(
        mask_pred=mask_pred, mask_true=mask_true, spacing=spacing
    )
    return scalars, logits


def batch_diffusion_evaluation(
    input_dict: Dict[str, jnp.ndarray],
    spacing: jnp.ndarray,
    num_classes: int,
    diffusion_model: GaussianDiffusion,
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
    """Evaluate predictions from diffusion model.

    Args:
        input_dict: input data having image and label.
        spacing: spacing of pixel/voxels along each dimension.
        num_classes: number of classes including background.
        diffusion_model: model for sampling.

    Returns:
        - metrics for all time steps, each metric value has shape (batch, ).
        - logits for all time steps.
    """
    # (batch, ..., 1)
    image = jnp.expand_dims(input_dict[IMAGE], axis=-1)
    # (batch, ..., num_classes)
    mask_true = jax.nn.one_hot(
        input_dict[LABEL], num_classes=num_classes, axis=-1
    )
    # (batch, ..., num_classes)
    x_t = diffusion_model.noise_sample(shape=mask_true.shape, dtype=image.dtype)
    # (batch, ..., num_classes)
    x_start = jnp.stack(
        list(diffusion_model.sample_mask_progressive(image=image, x_t=x_t)),
        axis=-1,
    )

    # evaluate
    # (batch, ..., num_classes, num_timesteps_sample)
    mask_pred = logits_to_mask(x_start, axis=-2)
    scalars = jax.vmap(
        partial(
            get_jit_segmentation_metrics,
            mask_true=mask_true,
            spacing=spacing,
        ),
        in_axes=-1,
        out_axes=-1,
    )(mask_pred)
    return scalars, x_start


def build_batch_eval_fn(
    config: DictConfig,
) -> Callable:
    """Build model from config.

    Args:
        config: entire config.

    Returns:
        Evaluate function.

    Raises:
        ValueError: if config is wrong or not supported.
    """
    if not hasattr(config.model, "name"):
        raise ValueError("Config does have model name.")

    set_mixed_precision_policy(
        use_mp=config.training.mixed_precision.use, model_name=config.model.name
    )

    # image spacing
    data_config = config.data
    dataset_name = data_config.name
    spacing = jnp.array(IMAGE_SPACING_MAP[dataset_name])
    num_classes = NUM_CLASSES_MAP[dataset_name]
    task_config = config.task
    vision_model = build_vision_model(
        data_config=data_config,
        task_config=task_config,
        model_config=config.model,
    )

    if task_config["name"] == "segmentation":
        return partial(
            batch_segmentation_evaluation,
            model=vision_model,
            spacing=spacing,
            num_classes=num_classes,
        )
    if task_config["name"] == "diffusion":
        diffusion_model = build_diffusion_model(
            model=vision_model,
            diffusion_config=task_config["diffusion"],
        )
        return partial(
            batch_diffusion_evaluation,
            spacing=spacing,
            num_classes=num_classes,
            diffusion_model=diffusion_model,
        )
    raise ValueError(f"Unknown task {task_config['name']}.")


def get_non_jit_segmentation_metrics_per_step(
    mask_pred: jnp.ndarray,
    mask_true: jnp.ndarray,
    spacing: Optional[jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """Calculate non-jittable segmentation metrics for batch.

    Cannot use VMAP as it requires jittable functions.

    Args:
        mask_pred: (batch, w, h, d, num_classes, num_steps)
        mask_true: (batch, w, h, d, num_classes)
        spacing: spacing of pixel/voxels along each dimension.

    Returns:
        Metrics dict, each value is a list corresponding to steps.
    """
    if mask_pred.ndim != 6:
        raise ValueError(
            "mask_pred should have shape "
            "(batch, w, h, d, num_classes, num_timesteps_sample) ,"
            f"got {mask_pred.shape}."
        )
    lst_scalars = []
    for i in range(mask_pred.shape[-1]):
        scalars = get_non_jit_segmentation_metrics(
            mask_pred=mask_pred[..., i],
            mask_true=mask_true,
            spacing=spacing,
        )
        lst_scalars.append(scalars)
    scalar_keys = lst_scalars[0].keys()
    scalars = {}
    for k in scalar_keys:
        scalars[k] = np.stack([x[k] for x in lst_scalars], axis=-1)
    return scalars


def dataset_segmentation_evaluation(  # pylint:disable=R0912,R0915
    evaluate_pmap: Callable,
    params: hk.Params,
    state: hk.State,
    rng: jnp.ndarray,
    batch_iterator: Iterable[Dict[str, chex.ArrayTree]],
    num_steps: int,
    is_diffusion: bool,
    spacing: Optional[jnp.ndarray],
    out_dir: Optional[Path],
    tfds_dir: Path,
    save_predictions: bool,
) -> Dict:
    """Get predictions and perform evaluations of a data set.

    Args:
        evaluate_pmap: forward function to call.
        params: model parameters.
        state: model state, EMA or not.
        rng: random key.
        batch_iterator: iterator of a data set.
        num_steps: number of steps.
        is_diffusion: the method is diffusion or not.
        spacing: spacing of pixel/voxels along each dimension.
        out_dir: output directory for metrics and predictions,
            if None, no files will be saved.
        tfds_dir: directory saving preprocessed images and labels.
        save_predictions: if True, save predicted masks.

    Returns:
        - predicted values
        - indices
        - metrics
    """
    lst_df_scalar = []
    for _ in range(num_steps):
        batch = next(batch_iterator)  # type: ignore[call-overload]

        # get UID and parse to string
        uids = batch.pop(UID)
        uids = uids.reshape(-1)  # remove shard axis
        uids = [
            x.decode("utf-8") if isinstance(x, bytes) else x
            for x in uids.tolist()
        ]

        # non diffusion
        #   logits (num_shards, batch, w, h, d, num_classes)
        #   metrics (num_shards, batch)
        # diffusion
        #   logits (num_shards, batch, w, h, d, num_classes, num_timesteps)
        #   metrics (num_shards, batch, num_timesteps)
        # arrays are across all devices
        batch = jax.lax.stop_gradient(batch)
        (scalars, logits), _ = evaluate_pmap(params, state, rng, batch)
        label = batch[LABEL]

        # put on cpu
        device_cpu = jax.devices("cpu")[0]
        scalars = jax.device_put(scalars, device_cpu)
        logits = jax.device_put(logits, device_cpu)
        label = jax.device_put(label, device_cpu)

        # remove shard axis
        # array are on device 0
        scalars = unshard(scalars)
        logits = unshard(logits)
        label = unshard(label)

        # remove padded examples
        if 0 in uids:
            # the batch was not complete, padded with zero
            num_samples = uids.index(0)
            uids = uids[:num_samples]
            scalars = unpad(scalars, num_samples)
            logits = unpad(logits, num_samples)
            label = unpad(label, num_samples)

        if is_diffusion:
            num_classes = logits.shape[-2]
            mask_true = jax.nn.one_hot(label, num_classes=num_classes, axis=-1)
            mask_pred = logits_to_mask(logits, axis=-2)
            scalars_non_jit = get_non_jit_segmentation_metrics_per_step(
                mask_pred=mask_pred,
                mask_true=mask_true,
                spacing=spacing,
            )
        else:
            num_classes = logits.shape[-1]
            mask_true = jax.nn.one_hot(label, num_classes=num_classes, axis=-1)
            scalars_non_jit = get_non_jit_segmentation_metrics(
                mask_pred=logits_to_mask(logits, axis=-1),
                mask_true=mask_true,
                spacing=spacing,
            )
        scalars = {**scalars, **scalars_non_jit}

        # for diffusion separate metrics per step
        if is_diffusion:
            scalars_flatten = {}
            for k, v in scalars.items():
                for i in range(v.shape[-1]):
                    scalars_flatten[f"{k}_step_{i}"] = v[..., i]
                scalars_flatten[k] = v[..., -1]
            scalars = scalars_flatten

        # save output
        if save_predictions and (out_dir is not None):
            if is_diffusion:
                for i in range(logits.shape[-1]):
                    mask_pred = np.array(
                        jnp.argmax(logits[..., i], axis=-1), dtype=int
                    )
                    save_segmentation_prediction(
                        preds=mask_pred,
                        uids=uids,
                        out_dir=out_dir / f"step_{i}",
                        tfds_dir=tfds_dir,
                    )
            else:
                mask_pred = np.array(jnp.argmax(logits, axis=-1), dtype=int)
                save_segmentation_prediction(
                    preds=mask_pred,
                    uids=uids,
                    out_dir=out_dir,
                    tfds_dir=tfds_dir,
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
    """Return a function to evaluate data set.

    Args:
        config: entire config.

    Returns:
        A function.

    Raises:
        ValueError: if data set in unknown.
    """
    is_diffusion = config["task"]["name"] == "diffusion"
    spacing = jnp.array(IMAGE_SPACING_MAP[config.data.name])
    tfds_dir = DIR_TFDS_PROCESSED_MAP[config.data.name]
    return partial(
        dataset_segmentation_evaluation,
        is_diffusion=is_diffusion,
        spacing=spacing,
        tfds_dir=tfds_dir,
    )
