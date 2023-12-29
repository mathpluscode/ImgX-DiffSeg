"""Recycling strategy for diffusion training."""
from typing import Callable

import chex
import jax
import jax.numpy as jnp
from jax import lax
from omegaconf import DictConfig

from imgx.datasets.constant import IMAGE, LABEL
from imgx.datasets.dataset_info import DatasetInfo
from imgx.diffusion.time_sampler import TimeSampler
from imgx.task.diffusion_segmentation.diffusion import DiffusionSegmentation
from imgx.task.diffusion_segmentation.diffusion_step import get_loss_logits_metrics
from imgx.task.diffusion_segmentation.train_state import TrainState


def get_self_conditioning_loss_step(
    train_state: TrainState,
    dataset_info: DatasetInfo,
    loss_config: DictConfig,
    diffusion_model: DiffusionSegmentation,
    time_sampler: TimeSampler,
    prev_step: str,
    probability: float,
) -> Callable[
    [chex.ArrayTree, chex.ArrayTree, jax.Array],
    tuple[jnp.ndarray, tuple[jnp.ndarray, chex.ArrayTree, jnp.ndarray, jnp.ndarray]],
]:
    """Return loss_step for self-conditioning diffusion.

    Args:
        train_state: train state.
        dataset_info: dataset info to transform label to mask.
        loss_config: have weights of diff losses.
        diffusion_model: segmentation diffusion model.
        time_sampler: time sampler for training.
        prev_step: same or next.
        probability: self conditioning probability.

    Returns:
        loss_step: loss step function.
    """

    def loss_step(
        params: chex.ArrayTree,
        batch: dict[str, jnp.ndarray],
        key: jax.Array,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, chex.ArrayTree, jnp.ndarray, jnp.ndarray]]:
        """Apply forward and calculate loss."""
        key_dropout_sc, key_dropout, key_t, key_noise_sc, key_sc = jax.random.split(key=key, num=5)
        image, label = batch[IMAGE], batch[LABEL]
        mask_true = dataset_info.label_to_mask(label, axis=-1, dtype=image.dtype)
        x_start = diffusion_model.mask_to_x(mask=mask_true)
        batch_size = image.shape[0]

        # t, t_index, probs_t
        # t_sc, t_index_sc
        if prev_step == "same":
            t, t_index, probs_t = time_sampler.sample(
                key=key_t,
                batch_size=batch_size,
                t_index_min=0,  # inclusive
                t_index_max=diffusion_model.num_timesteps,  # exclusive
                loss_count_hist=train_state.loss_count_hist,
                loss_sq_hist=train_state.loss_sq_hist,
            )
            t_index_sc = t_index
            t_sc = time_sampler.t_index_to_t(t_index=t_index_sc)
        elif prev_step == "next":
            t, t_index, probs_t = time_sampler.sample(
                key=key_t,
                batch_size=batch_size,
                t_index_min=0,  # inclusive
                t_index_max=diffusion_model.num_timesteps - 1,  # exclusive
                loss_count_hist=train_state.loss_count_hist,
                loss_sq_hist=train_state.loss_sq_hist,
            )
            t_index_sc = t_index + 1
            t_sc = time_sampler.t_index_to_t(t_index=t_index_sc)
        else:
            raise ValueError(f"prev_step {prev_step} not recognised.")

        # get predicted x_start
        noise_sc = diffusion_model.sample_noise(
            key=key_noise_sc, shape=x_start.shape, dtype=x_start.dtype
        )
        x_t_sc = diffusion_model.q_sample(x_start=x_start, noise=noise_sc, t_index=t_index_sc)
        mask_t_sc = diffusion_model.x_to_mask(x_t_sc)
        mask_t_sc = jnp.concatenate([mask_t_sc, jnp.zeros_like(mask_t_sc)], axis=-1)
        model_out_sc = train_state.apply_fn(
            {"params": params},
            True,  # is_train
            image,
            mask_t_sc,
            t_sc,
            rngs={"dropout": key_dropout_sc},
        )
        x_start_pred_sc = diffusion_model.predict_xstart_from_model_out_xt(
            model_out=model_out_sc,
            x_t=x_t_sc,
            t_index=t_index_sc,
        )

        # x_t
        if prev_step == "same":
            noise = noise_sc
            x_t = x_t_sc
        elif prev_step == "next":
            noise = noise_sc  # with interpolation, it's the same noise
            x_t = diffusion_model.predict_xprev_from_xstart_xt(
                x_start=x_start_pred_sc,
                x_t=x_t_sc,
                t_index=t_index_sc,
            )
        else:
            raise ValueError(f"prev_step {prev_step} not supported, has to be same or next.")

        batch_size = x_t.shape[0]
        mask_t = diffusion_model.x_to_mask(x_t_sc)
        mask_pred = diffusion_model.x_to_mask(x_start_pred_sc)
        if_self_cond = (
            jax.random.uniform(key=key_sc, shape=(batch_size,), dtype=mask_pred.dtype)
            <= probability
        )
        mask_pred *= jnp.expand_dims(if_self_cond, axis=range(1, mask_pred.ndim))
        mask_t = jnp.concatenate([mask_t, mask_pred], axis=-1)
        mask_t = lax.stop_gradient(mask_t)

        # forward
        model_out = train_state.apply_fn(
            {"params": params},
            True,  # is_train
            image,
            mask_t,
            t,
            rngs={"dropout": key_dropout},
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
