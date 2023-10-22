"""Vanilla segmentation loss."""
from __future__ import annotations

import jax.numpy as jnp
from omegaconf import DictConfig

from imgx.loss import cross_entropy, focal_loss
from imgx.loss.dice import dice_loss
from imgx.metric.area import class_proportion
from imgx_datasets.dataset_info import DatasetInfo


def segmentation_loss(
    logits: jnp.ndarray,
    label: jnp.ndarray,
    dataset_info: DatasetInfo,
    loss_config: DictConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Calculate segmentation loss with auxiliary losses and return metrics.

    Args:
        logits: unnormalised logits of shape (batch, ..., num_classes).
        label: label of shape (batch, ...).
        dataset_info: dataset info with helper functions.
        loss_config: have weights of diff losses.

    Returns:
        - calculated loss, of shape (batch,).
        - metrics, values of shape (batch,).
    """
    mask_true = dataset_info.label_to_mask(label, axis=-1)
    metrics = {}

    # class proportion
    # (batch, num_classes)
    class_prop_batch_cls = class_proportion(mask_true)
    for i in range(dataset_info.num_classes):
        metrics[f"class_{i}_proportion_true"] = class_prop_batch_cls[:, i]

    # Dice
    # (batch, num_classes)
    dice_loss_batch_cls = dice_loss(
        logits=logits,
        mask_true=mask_true,
        classes_are_exclusive=dataset_info.classes_are_exclusive,
    )
    # (batch, )
    # without background
    # mask out non-existing classes
    dice_loss_batch = jnp.mean(
        dice_loss_batch_cls[:, 1:], axis=-1, where=class_prop_batch_cls[:, 1:] > 0
    )
    metrics["dice_loss"] = dice_loss_batch
    for i in range(dice_loss_batch_cls.shape[-1]):
        metrics[f"dice_loss_class_{i}"] = dice_loss_batch_cls[:, i]

    # cross entropy
    ce_loss_batch = cross_entropy(
        logits=logits,
        mask_true=mask_true,
        classes_are_exclusive=dataset_info.classes_are_exclusive,
    )
    metrics["cross_entropy_loss"] = ce_loss_batch

    # focal loss
    focal_loss_batch = focal_loss(
        logits=logits,
        mask_true=mask_true,
        classes_are_exclusive=dataset_info.classes_are_exclusive,
    )
    metrics["focal_loss"] = focal_loss_batch

    # total loss
    loss_batch = jnp.zeros_like(dice_loss_batch)
    if loss_config["dice"] > 0:
        loss_batch += dice_loss_batch * loss_config["dice"]
    if loss_config["cross_entropy"] > 0:
        loss_batch += ce_loss_batch * loss_config["cross_entropy"]
    if loss_config["focal"] > 0:
        loss_batch += focal_loss_batch * loss_config["focal"]
    metrics["total_loss"] = loss_batch
    return loss_batch, metrics
