"""Module for building models and losses."""
from typing import Callable, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from imgx import IMAGE, LABEL
from imgx.datasets import NUM_CLASSES_MAP
from imgx.diffusion.gaussian_diffusion import (
    DiffusionModelOutputType,
    DiffusionModelVarianceType,
    GaussianDiffusion,
)
from imgx.exp.mixed_precision import set_mixed_precision_policy
from imgx.exp.model import build_diffusion_model, build_vision_model
from imgx.loss import mean_cross_entropy, mean_focal_loss
from imgx.loss.dice import (
    dice_loss,
    mean_with_background,
    mean_without_background,
)
from imgx.metric import class_proportion


def segmentation_loss_with_aux(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
    loss_config: DictConfig,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Calculate segmentation loss with auxiliary losses and return metrics.

    Args:
        logits: unnormalised logits of shape (batch, ..., num_classes).
        mask_true: one hot label of shape (batch, ..., num_classes).
        loss_config: have weights of diff losses.

    Returns:
        - calculated loss.
        - metrics.
    """
    scalars = {}

    # Dice
    # (batch, num_classes)
    dice_loss_batch_cls = dice_loss(
        logits=logits,
        mask_true=mask_true,
    )
    # (1,)
    dice_loss_scalar = jax.lax.cond(
        loss_config["dice_include_background"],
        mean_with_background,
        mean_without_background,
        dice_loss_batch_cls,
    )
    scalars["mean_dice_loss"] = dice_loss_scalar

    # metrics
    for i in range(dice_loss_batch_cls.shape[-1]):
        scalars[f"mean_dice_loss_class_{i}"] = jnp.nanmean(
            dice_loss_batch_cls[:, i]
        )
        scalars[f"min_dice_loss_class_{i}"] = jnp.nanmin(
            dice_loss_batch_cls[:, i]
        )
        scalars[f"max_dice_loss_class_{i}"] = jnp.nanmax(
            dice_loss_batch_cls[:, i]
        )
    scalars["mean_dice_loss"] = jnp.nanmean(dice_loss_batch_cls)
    scalars["min_dice_loss"] = jnp.nanmin(dice_loss_batch_cls)
    scalars["max_dice_loss"] = jnp.nanmax(dice_loss_batch_cls)

    # cross entropy
    ce_loss_scalar = mean_cross_entropy(
        logits=logits,
        mask_true=mask_true,
    )
    scalars["mean_cross_entropy_loss"] = ce_loss_scalar

    # focal loss
    focal_loss_scalar = mean_focal_loss(
        logits=logits,
        mask_true=mask_true,
    )
    scalars["mean_focal_loss"] = focal_loss_scalar

    # total loss
    loss_scalar = 0
    if loss_config["dice"] > 0:
        loss_scalar += dice_loss_scalar * loss_config["dice"]
    if loss_config["cross_entropy"] > 0:
        loss_scalar += ce_loss_scalar * loss_config["cross_entropy"]
    if loss_config["focal"] > 0:
        loss_scalar += focal_loss_scalar * loss_config["focal"]

    # class proportion
    # (batch, num_classes)
    cls_prop = class_proportion(mask_true)
    for i in range(dice_loss_batch_cls.shape[-1]):
        scalars[f"mean_proportion_class_{i}"] = jnp.nanmean(cls_prop[:, i])
        scalars[f"min_proportion_class_{i}"] = jnp.nanmin(cls_prop[:, i])
        scalars[f"max_proportion_class_{i}"] = jnp.nanmax(cls_prop[:, i])

    return loss_scalar, scalars


def segmentation_loss(
    input_dict: Dict[str, jnp.ndarray],
    model: hk.Module,
    num_classes: int,
    loss_config: DictConfig,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Calculate segmentation loss and return metrics.

    Args:
        input_dict: input data having image and label.
        model: network instance.
        num_classes: number of classes including background.
        loss_config: have weights of diff losses.

    Returns:
        - calculated loss.
        - metrics.
    """
    # (batch, ..., 1)
    image = jnp.expand_dims(input_dict[IMAGE], axis=-1)
    # (batch, ..., num_classes)
    logits = model(image=image, is_train=True)
    # (batch, ..., num_classes)
    mask_true = jax.nn.one_hot(
        input_dict[LABEL], num_classes=num_classes, axis=-1
    )
    return segmentation_loss_with_aux(
        logits=logits,
        mask_true=mask_true,
        loss_config=loss_config,
    )


def diffusion_loss(  # pylint:disable=R0915
    input_dict: Dict[str, jnp.ndarray],
    num_classes: int,
    gd: GaussianDiffusion,
    loss_config: DictConfig,
    recycle: bool,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Calculate diffusion loss and return metrics.

    In diffusion, the noise is defined on segmentation mask.
    That is, x_t is segmentation logits.

    Args:
        input_dict: input data having image, label, and time_step.
            image: (batch, ...)
            label: (batch, ..., num_classes)
            time_step: (batch, )
        num_classes: number of classes including background.
        gd: model for sampling.
        loss_config: have weights of diff losses.
        recycle: recycle model prediction or not.

    Returns:
        - calculated loss.
        - metrics.
    """
    scalars = {}

    # (batch, ..., 1)
    image = jnp.expand_dims(input_dict[IMAGE], axis=-1)
    # (batch, ..., num_classes)
    mask_true = jax.nn.one_hot(
        input_dict[LABEL],
        num_classes=num_classes,
        axis=-1,
        dtype=image.dtype,
    )
    # noise are standard normal distribution
    x_start = mask_true * 2 - 1

    # (batch, )
    t = gd.sample_timestep(batch_size=image.shape[0])

    if recycle:
        # (batch, ..., num_classes)
        noise_recyle = gd.noise_sample(shape=x_start.shape, dtype=x_start.dtype)
        t_recycle = jnp.minimum(t + 1, gd.num_timesteps - 1)
        x_t_recycle = gd.q_sample(
            x_start=x_start, noise=noise_recyle, t=t_recycle
        )
        # (batch, ..., ch_input + num_classes)
        model_in_recycle = jnp.concatenate([image, x_t_recycle], axis=-1)
        # (batch, ..., num_classes) or (batch, ..., 2*num_classes)
        # model outputs are always logits
        model_out_recycle = gd.model(model_in_recycle, t_recycle, is_train=True)
        x_start_recycle, _, _ = gd.p_mean_variance(
            model_out=model_out_recycle,
            x_t=x_t_recycle,
            t=t_recycle,
        )
        x_start = jax.lax.stop_gradient(x_start_recycle)

    # (batch, ..., num_classes)
    noise = gd.noise_sample(shape=x_start.shape, dtype=x_start.dtype)
    x_t = gd.q_sample(x_start=x_start, noise=noise, t=t)
    # (batch, ..., ch_input + num_classes)
    model_in = jnp.concatenate([image, x_t], axis=-1)
    # (batch, ..., num_classes) or (batch, ..., 2*num_classes)
    # model outputs are always logits
    model_out = gd.model(model_in, t, is_train=True)

    model_out_vlb = jax.lax.stop_gradient(model_out)
    if gd.model_var_type in [
        DiffusionModelVarianceType.LEARNED,
        DiffusionModelVarianceType.LEARNED_RANGE,
    ]:
        # model_out (batch, ..., num_classes)
        model_out, log_variance = jnp.split(
            model_out, indices_or_sections=2, axis=-1
        )
        # apply a stop-gradient to the mean output for the vlb to prevent
        # this loss change mean prediction
        model_out_vlb = jax.lax.stop_gradient(model_out)
        # model_out (batch, ..., num_classes*2)
        model_out_vlb = jnp.concatenate([model_out_vlb, log_variance], axis=-1)

    vlb_scalar = gd.variational_lower_bound(
        model_out=model_out_vlb,
        x_start=x_start,
        x_t=x_t,
        t=t,
    )
    vlb_scalar = jnp.nanmean(vlb_scalar)
    scalars["vlb_loss"] = vlb_scalar

    if gd.model_out_type == DiffusionModelOutputType.EPSILON:
        mse_loss_scalar = jnp.mean((model_out - noise) ** 2)
        scalars["mse_loss"] = mse_loss_scalar

        x_start = gd.predict_xstart_from_epsilon_xt(
            x_t=x_t, epsilon=model_out, t=t
        )
        logits = gd.x_to_logits(x_start)
        seg_loss_scalar, seg_scalars = segmentation_loss_with_aux(
            logits=logits,
            mask_true=mask_true,
            loss_config=loss_config,
        )
        scalars = {**scalars, **seg_scalars}

        loss_scalar = loss_config["mse"] * mse_loss_scalar + seg_loss_scalar
    elif gd.model_out_type == DiffusionModelOutputType.X_START:
        logits = model_out
        loss_scalar, seg_scalars = segmentation_loss_with_aux(
            logits=logits,
            mask_true=mask_true,
            loss_config=loss_config,
        )
        scalars = {**scalars, **seg_scalars}
    else:
        raise ValueError(
            f"Unknown DiffusionModelOutputType {gd.model_out_type}."
        )

    if gd.model_var_type in [
        DiffusionModelVarianceType.LEARNED,
        DiffusionModelVarianceType.LEARNED_RANGE,
    ]:
        # TODO nan values may happen
        loss_scalar += vlb_scalar * gd.num_timesteps / gd.num_timesteps_beta

    scalars["total_loss"] = loss_scalar
    scalars["mean_t"] = jnp.mean(t)
    scalars["max_t"] = jnp.max(t)
    scalars["min_t"] = jnp.min(t)

    return loss_scalar, scalars


def build_loss_fn(
    config: DictConfig,
) -> Callable[
    [Dict[str, jnp.ndarray]], Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]
]:
    """Build model from config.

    Args:
        config: entire config.

    Returns:
        Loss function.

    Raises:
        ValueError: if config is wrong or not supported.
    """
    data_config = config.data
    task_config = config.task
    model_config = config.model
    loss_config = config.loss
    mp_config = config.training.mixed_precision

    set_mixed_precision_policy(
        use_mp=mp_config.use, model_name=model_config.name
    )

    # number of classes including background
    num_classes = NUM_CLASSES_MAP[data_config["name"]]

    if task_config["name"] == "segmentation":

        def seg_loss_fn(
            input_dict: Dict[str, jnp.ndarray]
        ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            vision_model = build_vision_model(
                data_config=data_config,
                task_config=task_config,
                model_config=model_config,
            )
            return segmentation_loss(
                input_dict=input_dict,
                model=vision_model,
                num_classes=num_classes,
                loss_config=loss_config,
            )

        return seg_loss_fn
    if task_config["name"] == "diffusion":

        def diffusion_loss_fn(
            input_dict: Dict[str, jnp.ndarray]
        ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            vision_model = build_vision_model(
                data_config=data_config,
                task_config=task_config,
                model_config=model_config,
            )
            diffusion_model = build_diffusion_model(
                model=vision_model,
                diffusion_config=task_config["diffusion"],
            )
            recycle = task_config["diffusion"]["recycle"]
            return diffusion_loss(
                input_dict=input_dict,
                gd=diffusion_model,
                num_classes=num_classes,
                loss_config=loss_config,
                recycle=recycle,
            )

        return diffusion_loss_fn
    raise ValueError(f"Unknown task {task_config['name']}.")
