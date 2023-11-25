"""Standard diffusion training."""
from typing import Callable

import chex
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from imgx.diffusion.time_sampler import TimeSampler
from imgx.loss.segmentation import segmentation_loss
from imgx.metric.util import aggregate_metrics, aggregate_metrics_for_diffusion
from imgx.task.diffusion_segmentation.diffusion import DiffusionSegmentation
from imgx.task.diffusion_segmentation.train_state import TrainState
from imgx_datasets.constant import IMAGE, LABEL
from imgx_datasets.dataset_info import DatasetInfo


def get_loss_logits_metrics(
    batch: dict[str, jnp.ndarray],
    x_start: jnp.ndarray,
    x_t: jnp.ndarray,
    t_index: jnp.ndarray,
    probs_t: jnp.ndarray,
    noise: jnp.ndarray,
    model_out: jnp.ndarray,
    dataset_info: DatasetInfo,
    loss_config: DictConfig,
    diffusion_model: DiffusionSegmentation,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Get loss, logits, and metrics.

    Args:
        batch: batch of data.
        x_start: x_start, shape (batch, ..., num_classes).
        x_t: x_t, shape (batch, ..., num_classes).
        t_index: t_index, shape (batch, ).
        probs_t: probs_t, shape (batch, ).
        noise: noise, shape (batch, ..., num_classes).
        model_out: model_out, shape (batch, ..., num_classes).
        dataset_info: dataset info to transform label to mask.
        loss_config: have weights of diff losses.
        diffusion_model: segmentation diffusion model.

    Returns:
        loss, loss_batch, logits, metrics.
    """
    # diffusion loss, VLB, MSE, etc.
    metrics_batch_diff, model_out = diffusion_model.diffusion_loss(
        x_start=x_start,
        x_t=x_t,
        t_index=t_index,
        noise=noise,
        model_out=model_out,
    )
    # segmentation loss
    logits = diffusion_model.model_out_to_logits_start(
        model_out=model_out,
        x_t=x_t,
        t_index=t_index,
    )
    loss_batch, metrics_batch_seg = segmentation_loss(
        logits=logits,
        label=batch[LABEL],
        dataset_info=dataset_info,
        loss_config=loss_config,
    )
    # add aux loss
    if loss_config["mse"] > 0:
        loss_batch += loss_config["mse"] * metrics_batch_diff["mse_loss"]
    if loss_config["vlb"] > 0:
        loss_batch += loss_config["vlb"] * metrics_batch_diff["vlb_loss"]

    # importance sampling
    weights_t = probs_t * diffusion_model.num_timesteps  # so that mean(weights_t) ~ 1
    loss = jnp.mean(loss_batch * weights_t)

    # record metrics
    # each value is of shape (batch_size, )
    metrics_batch = {
        "t_index": t_index,
        "probs_t": probs_t,
        **metrics_batch_diff,
        **metrics_batch_seg,
    }
    metrics = aggregate_metrics(metrics_batch)
    metrics_diff = aggregate_metrics_for_diffusion(
        metrics={
            k: v
            for k, v in metrics_batch_seg.items()
            if ("class" not in k) and k.startswith("mean_")
        },
        t_index=t_index,
    )
    metrics = {"total_loss": loss, **metrics, **metrics_diff}
    return loss, loss_batch, logits, metrics


def get_diffusion_loss_step(
    train_state: TrainState,
    dataset_info: DatasetInfo,
    loss_config: DictConfig,
    diffusion_model: DiffusionSegmentation,
    time_sampler: TimeSampler,
) -> Callable[
    [chex.ArrayTree, chex.ArrayTree, jax.Array],
    tuple[jnp.ndarray, tuple[jnp.ndarray, chex.ArrayTree, jnp.ndarray, jnp.ndarray]],
]:
    """Return loss_step for vanilla diffusion.

    Args:
        train_state: train state.
        dataset_info: dataset info to transform label to mask.
        loss_config: have weights of diff losses.
        diffusion_model: segmentation diffusion model.
        time_sampler: time sampler for training.

    Returns:
        loss_step: loss step function.
    """

    def loss_step(
        params: chex.ArrayTree,
        batch: chex.ArrayTree,
        key: jax.Array,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, chex.ArrayTree, jnp.ndarray, jnp.ndarray]]:
        """Apply forward and calculate loss."""
        image, label = batch[IMAGE], batch[LABEL]
        mask_true = dataset_info.label_to_mask(label, axis=-1, dtype=image.dtype)
        x_start = diffusion_model.mask_to_x(mask=mask_true)
        batch_size = image.shape[0]
        key_t, key_noise = jax.random.split(key=key)

        # t, t_index, probs_t
        t, t_index, probs_t = time_sampler.sample(
            key=key_t,
            batch_size=batch_size,
            t_index_min=0,  # inclusive
            t_index_max=time_sampler.num_timesteps,  # exclusive
            loss_count_hist=train_state.loss_count_hist,
            loss_sq_hist=train_state.loss_sq_hist,
        )

        # x_t
        noise = diffusion_model.sample_noise(
            key=key_noise, shape=x_start.shape, dtype=x_start.dtype
        )
        x_t = diffusion_model.q_sample(x_start=x_start, noise=noise, t_index=t_index)
        mask_t = diffusion_model.x_to_mask(x_t)

        # forward
        model_out = train_state.apply_fn(
            {"params": params},
            image,
            mask_t,
            t,
        )

        # loss
        loss, loss_batch, logits, metrics = get_loss_logits_metrics(
            batch=batch,
            x_start=x_start,
            x_t=x_t,
            t_index=t_index,
            probs_t=probs_t,
            noise=noise,
            model_out=model_out,
            dataset_info=dataset_info,
            loss_config=loss_config,
            diffusion_model=diffusion_model,
        )

        return loss, (logits, metrics, loss_batch, t_index)

    return loss_step
