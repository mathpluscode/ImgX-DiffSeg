"""Vanilla segmentation loss."""
from __future__ import annotations

import haiku as hk
from jax import numpy as jnp
from omegaconf import DictConfig

from imgx.exp.loss.util import aggregate_batch_scalars
from imgx.loss import cross_entropy, focal_loss
from imgx.loss.dice import dice_loss
from imgx.metric.area import class_proportion
from imgx_datasets.constant import IMAGE, LABEL
from imgx_datasets.dataset_info import DatasetInfo


def segmentation_loss_step(
    logits: jnp.ndarray,
    mask_true: jnp.ndarray,
    dataset_info: DatasetInfo,
    loss_config: DictConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Calculate segmentation loss with auxiliary losses and return metrics.

    Args:
        logits: unnormalised logits of shape (batch, ..., num_classes).
        mask_true: one hot label of shape (batch, ..., num_classes).
        dataset_info: dataset info with helper functions.
        loss_config: have weights of diff losses.

    Returns:
        - calculated loss, of shape (batch,).
        - metrics, each of shape (batch,).
    """
    scalars = {}

    # Dice
    # (batch, num_classes)
    dice_loss_batch_cls = dice_loss(
        logits=logits,
        mask_true=mask_true,
        classes_are_exclusive=dataset_info.classes_are_exclusive,
    )
    # (batch, )
    # without background
    dice_loss_batch = jnp.nanmean(dice_loss_batch_cls[:, 1:], axis=-1)
    scalars["dice_loss"] = dice_loss_batch
    for i in range(dice_loss_batch_cls.shape[-1]):
        scalars[f"dice_loss_class_{i}"] = dice_loss_batch_cls[:, i]

    # cross entropy
    ce_loss_batch = cross_entropy(
        logits=logits,
        mask_true=mask_true,
        classes_are_exclusive=dataset_info.classes_are_exclusive,
    )
    scalars["cross_entropy_loss"] = ce_loss_batch

    # focal loss
    focal_loss_batch = focal_loss(
        logits=logits,
        mask_true=mask_true,
        classes_are_exclusive=dataset_info.classes_are_exclusive,
    )
    scalars["focal_loss"] = focal_loss_batch

    # total loss
    loss_batch = jnp.zeros_like(dice_loss_batch)
    if loss_config["dice"] > 0:
        loss_batch += dice_loss_batch * loss_config["dice"]
    if loss_config["cross_entropy"] > 0:
        loss_batch += ce_loss_batch * loss_config["cross_entropy"]
    if loss_config["focal"] > 0:
        loss_batch += focal_loss_batch * loss_config["focal"]
    scalars["total_loss"] = loss_batch

    # class proportion
    mask_pred = dataset_info.label_to_mask(
        dataset_info.logits_to_label(logits, axis=-1), axis=-1
    )
    for mask, mask_name in zip([mask_pred, mask_true], ["pred", "true"]):
        # (batch, num_classes)
        class_prop_cls = class_proportion(mask)
        for i in range(dataset_info.num_classes):
            scalars[f"class_{i}_proportion_{mask_name}"] = class_prop_cls[:, i]

    return loss_batch, scalars


def segmentation_loss(
    input_dict: dict[str, jnp.ndarray],
    dataset_info: DatasetInfo,
    model: hk.Module,
    loss_config: DictConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Calculate segmentation loss and return metrics.

    Args:
        input_dict: input data having image and label.
        dataset_info: dataset info with helper functions.
        model: network instance.
        loss_config: have weights of diff losses.

    Returns:
        - calculated loss.
        - metrics.
    """
    image, label = input_dict[IMAGE], input_dict[LABEL]

    # (batch, ..., num_classes).
    logits = model(image=image)
    mask_true = dataset_info.label_to_mask(label, axis=-1)
    loss_batch, scalars_batch = segmentation_loss_step(
        logits=logits,
        mask_true=mask_true,
        dataset_info=dataset_info,
        loss_config=loss_config,
    )
    loss_scalar = jnp.mean(loss_batch)
    scalars = aggregate_batch_scalars(scalars_batch)
    return loss_scalar, scalars
