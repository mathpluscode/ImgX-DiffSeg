"""Recycling strategy for diffusion training."""
import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax

from imgx.diffusion import DiffusionSegmentation
from imgx.diffusion.time_sampler import TimeSampler
from imgx.exp.loss.diffusion.args import DiffusionLossArgs


def self_conditioning_step(
    image: jnp.ndarray,
    x_start: jnp.ndarray,
    probability: float,
    prev_step: str,
    seg_dm: DiffusionSegmentation,
    time_sampler: TimeSampler,
) -> tuple[DiffusionLossArgs, DiffusionLossArgs, jnp.ndarray]:
    """Perform recycling.

    Args:
        image: image, (batch, ..., in_channels).
        x_start: ground truth, (batch, ..., num_classes)
        probability: self conditioning probability.
        prev_step: same or next.
        seg_dm: segmentation diffusion model.
        time_sampler: time sampler for training.

    Returns:
        loss_args_sc: Args for recycling step.
        loss_args: Args for normal step.
        probs_t: probs of sampling t.
    """
    batch_size = image.shape[0]

    # t, t_index, probs_t
    # t_sc, t_index_sc
    if prev_step == "same":
        t, t_index, probs_t = time_sampler.sample(
            batch_size=batch_size,
            t_index_min=0,  # inclusive
            t_index_max=seg_dm.num_timesteps,  # exclusive
        )
        t_index_sc = t_index
        t_sc = time_sampler.t_index_to_t(t_index=t_index_sc)
    elif prev_step == "next":
        t, t_index, probs_t = time_sampler.sample(
            batch_size=batch_size,
            t_index_min=0,  # inclusive
            t_index_max=seg_dm.num_timesteps - 1,  # exclusive
        )
        t_index_sc = t_index + 1
        t_sc = time_sampler.t_index_to_t(t_index=t_index_sc)
    else:
        raise ValueError(
            f"prev_step {prev_step} not supported, has to be same or next."
        )

    # get predicted x_start
    noise_sc = seg_dm.sample_noise(shape=x_start.shape, dtype=x_start.dtype)
    x_t_sc = seg_dm.q_sample(
        x_start=x_start, noise=noise_sc, t_index=t_index_sc
    )
    mask_t = seg_dm.x_to_mask(x_t_sc)
    mask = jnp.concatenate([mask_t, jnp.zeros_like(mask_t)], axis=-1)
    model_out_sc = seg_dm.model(
        image=image,
        mask=mask,
        t=t_sc,
    )
    x_start_pred_sc = seg_dm.predict_xstart_from_model_out_xt(
        model_out=model_out_sc,
        x_t=x_t_sc,
        t_index=t_index_sc,
    )

    # get x_t
    if prev_step == "same":
        noise = noise_sc
        x_t = x_t_sc
    elif prev_step == "next":
        noise = noise_sc  # with interpolation, it's the same noise
        x_t = seg_dm.predict_xprev_from_xstart_xt(
            x_start=x_start_pred_sc,
            x_t=x_t_sc,
            t_index=t_index_sc,
        )
    else:
        raise ValueError(
            f"prev_step {prev_step} not supported, has to be same or next."
        )

    batch_size = x_t.shape[0]
    mask_t = seg_dm.x_to_mask(x_t_sc)
    mask_pred = seg_dm.x_to_mask(x_start_pred_sc)
    if_self_cond = (
        jax.random.uniform(
            key=hk.next_rng_key(), shape=(batch_size,), dtype=mask_pred.dtype
        )
        <= probability
    )
    mask_pred *= jnp.expand_dims(if_self_cond, axis=range(1, mask_pred.ndim))
    mask = jnp.concatenate([mask_t, mask_pred], axis=-1)
    mask = lax.stop_gradient(mask)
    model_out = seg_dm.model(
        image=image,
        mask=mask,
        t=t,
    )

    loss_args_sc = DiffusionLossArgs(
        t_index=t_index_sc,
        x_t=x_t_sc,
        noise=noise_sc,
        model_out=model_out_sc,
    )
    loss_args = DiffusionLossArgs(
        t_index=t_index,
        x_t=x_t,
        noise=noise,
        model_out=model_out,
    )
    return loss_args_sc, loss_args, probs_t
