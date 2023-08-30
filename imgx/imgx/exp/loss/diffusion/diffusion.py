"""Segmentation loss with diffusion."""
from __future__ import annotations

from jax import numpy as jnp
from omegaconf import DictConfig

from imgx.diffusion import DiffusionSegmentation
from imgx.diffusion.time_sampler import TimeSampler
from imgx.exp.loss.diffusion.diffusion_step import diffusion_step
from imgx.exp.loss.diffusion.recycling_step import recycling_step
from imgx.exp.loss.diffusion.self_conditioning_step import (
    self_conditioning_step,
)
from imgx.exp.loss.segmentation import segmentation_loss_step
from imgx.exp.loss.util import (
    aggregate_batch_scalars,
    aggregate_batch_scalars_for_diffusion,
)
from imgx_datasets.constant import IMAGE, LABEL
from imgx_datasets.dataset_info import DatasetInfo


def diffusion_segmentation_loss_step(
    x_start: jnp.ndarray,
    x_t: jnp.ndarray,
    t_index: jnp.ndarray,
    noise: jnp.ndarray,
    model_out: jnp.ndarray,
    mask_true: jnp.ndarray,
    dataset_info: DatasetInfo,
    sd: DiffusionSegmentation,
    loss_config: DictConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Calculate diffusion loss with auxiliary losses and return metrics.

    Args:
        x_start: label at time 0 of shape (batch, ..., num_classes).
        x_t: label at time t of shape (batch, ..., num_classes).
        t_index: time of shape (batch, ...).
        noise: noise of shape (batch, ..., num_classes).
        model_out: unnormalised logits of
            shape (batch, ..., num_classes) or (batch, ..., 2*num_classes)
        sd: segmentation diffusion model.
        mask_true: boolean label of shape (batch, ..., num_classes).
        dataset_info: dataset info with helper functions.
        loss_config: have weights of diff losses.

    Returns:
        - calculated loss, of shape (batch,).
        - metrics, each of shape (batch,).
    """
    # diffusion loss, VLB, MSE, etc.
    scalars_batch, model_out = sd.diffusion_loss(
        x_start=x_start,
        x_t=x_t,
        t_index=t_index,
        noise=noise,
        model_out=model_out,
    )

    # segmentation loss
    logits = sd.model_out_to_logits_start(
        model_out=model_out,
        x_t=x_t,
        t_index=t_index,
    )
    loss_batch, seg_scalars_batch = segmentation_loss_step(
        logits=logits,
        mask_true=mask_true,
        dataset_info=dataset_info,
        loss_config=loss_config,
    )
    scalars_batch = {**scalars_batch, **seg_scalars_batch}

    # add aux loss
    if loss_config["mse"] > 0:
        loss_batch += loss_config["mse"] * scalars_batch["mse_loss"]
    if loss_config["vlb"] > 0:
        loss_batch += loss_config["vlb"] * scalars_batch["vlb_loss"]
    scalars_batch["total_loss"] = loss_batch

    return loss_batch, scalars_batch


def diffusion_segmentation_loss(  # pylint:disable=R0915
    input_dict: dict[str, jnp.ndarray],
    dataset_info: DatasetInfo,
    seg_dm: DiffusionSegmentation,
    time_sampler: TimeSampler,
    loss_config: DictConfig,
    diff_config: DictConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Calculate diffusion loss and return metrics.

    In diffusion, the noise is defined on segmentation mask.
    That is, x_t is segmentation logits.

    Args:
        input_dict: input data having image, label, and time_step.
            image: (batch, ...)
            label: (batch, ..., num_classes)
            time_step: (batch, )
        dataset_info: dataset info with helper functions.
        seg_dm: segmentation diffusion model.
        time_sampler: time sampler for training.
        loss_config: have weights of diff losses.
        diff_config: diffusion configuration.

    Returns:
        - calculated loss.
        - metrics.
    """
    if diff_config.recycling.use and diff_config.self_conditioning.use:
        raise ValueError(
            "recycling and self-conditioning cannot be used together."
        )

    image, label = input_dict[IMAGE], input_dict[LABEL]
    mask_true = dataset_info.label_to_mask(label, axis=-1, dtype=image.dtype)
    x_start = seg_dm.mask_to_x(mask=mask_true)

    if diff_config.recycling.use:
        loss_args_aug, loss_args, probs_t = recycling_step(
            image=image,
            x_start=x_start,
            prev_step=diff_config.recycling.prev_step,
            reverse_step=diff_config.recycling.reverse_step,
            seg_dm=seg_dm,
            time_sampler=time_sampler,
        )
    elif diff_config.self_conditioning.use:
        loss_args_aug, loss_args, probs_t = self_conditioning_step(
            image=image,
            x_start=x_start,
            probability=diff_config.self_conditioning.probability,
            prev_step=diff_config.self_conditioning.prev_step,
            seg_dm=seg_dm,
            time_sampler=time_sampler,
        )
    else:
        loss_args, probs_t = diffusion_step(
            image=image,
            x_start=x_start,
            seg_dm=seg_dm,
            time_sampler=time_sampler,
        )

    # standard diffusion loss
    loss_batch, scalars_batch = diffusion_segmentation_loss_step(
        x_start=x_start,
        x_t=loss_args.x_t,
        t_index=loss_args.t_index,
        noise=loss_args.noise,
        model_out=loss_args.model_out,
        mask_true=mask_true,
        dataset_info=dataset_info,
        sd=seg_dm,
        loss_config=loss_config,
    )
    scalars_loss_hist = time_sampler.update_loss_sq_hist(
        loss_batch=loss_batch,
        t_index=loss_args.t_index,
    )

    # importance sampling
    scalars_batch["t_index"] = loss_args.t_index
    scalars_batch["probs_t"] = probs_t
    weights_t = probs_t * seg_dm.num_timesteps  # so that mean(weights_t) ~ 1
    loss_scalar = jnp.mean(loss_batch * weights_t)
    scalars = aggregate_batch_scalars(scalars_batch)
    scalars_diff = aggregate_batch_scalars_for_diffusion(
        scalars=scalars_batch,
        t_index=loss_args.t_index,
        num_timesteps=seg_dm.num_timesteps,
    )

    # additional loss
    aux_loss_re = diff_config.recycling.use and diff_config.recycling.aux_loss
    aux_loss_sc = (
        diff_config.self_conditioning.use
        and diff_config.self_conditioning.aux_loss
    )
    if aux_loss_re or aux_loss_sc:
        loss_batch_aug, scalars_batch_aug = diffusion_segmentation_loss_step(
            x_start=x_start,
            x_t=loss_args_aug.x_t,
            t_index=loss_args_aug.t_index,
            noise=loss_args_aug.noise,
            model_out=loss_args_aug.model_out,
            mask_true=mask_true,
            dataset_info=dataset_info,
            sd=seg_dm,
            loss_config=loss_config,
        )
        suffix = "recycling" if aux_loss_re else "self_conditioning"
        scalars_batch_aug = {
            f"{k}_{suffix}": v for k, v in scalars_batch_aug.items()
        }
        loss_scalar_aug = jnp.mean(loss_batch_aug)
        loss_scalar += loss_scalar_aug
        scalars_aug = aggregate_batch_scalars(scalars_batch_aug)
        scalars = {**scalars, **scalars_aug}

    # assemble all metrics
    scalars = {
        "total_loss": loss_scalar,
        **scalars,
        **scalars_diff,
        **scalars_loss_hist,
    }
    return loss_scalar, scalars
