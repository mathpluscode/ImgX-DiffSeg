"""Recycling strategy for diffusion training."""
import jax.numpy as jnp
from jax import lax

from imgx.diffusion import DiffusionSegmentation
from imgx.diffusion.time_sampler import TimeSampler
from imgx.exp.loss.diffusion.args import DiffusionLossArgs


def recycling_step(
    image: jnp.ndarray,
    x_start: jnp.ndarray,
    prev_step: str,
    reverse_step: bool,
    seg_dm: DiffusionSegmentation,
    time_sampler: TimeSampler,
) -> tuple[DiffusionLossArgs, DiffusionLossArgs, jnp.ndarray]:
    """Perform recycling.

    Args:
        image: image, (batch, ..., in_channels).
        x_start: ground truth, (batch, ..., num_classes)
        prev_step: max or next.
            max means the previous step is num_timesteps - 1
            next means the previous step is min(t+1, num_timesteps - 1)
        reverse_step: reverse the previous step or not.
        seg_dm: segmentation diffusion model.
        time_sampler: time sampler for training.

    Returns:
        loss_args_re: Args for recycling step.
        loss_args: Args for normal step.
        probs_t: probs of sampling t.
    """
    batch_size = image.shape[0]

    # t, t_index, probs_t
    # t_re, t_index_re
    # empirically, sample t from [0, num_timesteps - 1) is better than
    # sampling from [0, num_timesteps), for most max / next options.
    if prev_step == "max":
        if reverse_step:
            raise ValueError(
                "reverse_step should be False when prev_step is max."
            )
        t, t_index, probs_t = time_sampler.sample(
            batch_size=batch_size,
            t_index_min=0,  # inclusive
            t_index_max=seg_dm.num_timesteps - 1,  # exclusive
        )
        t_index_re = jnp.full(
            shape=(batch_size,),
            fill_value=seg_dm.num_timesteps - 1,
            dtype=jnp.int32,
        )
        t_re = time_sampler.t_index_to_t(t_index=t_index_re)
    elif prev_step == "next":
        t, t_index, probs_t = time_sampler.sample(
            batch_size=batch_size,
            t_index_min=0,  # inclusive
            t_index_max=seg_dm.num_timesteps - 1,  # exclusive
        )
        t_index_re = t_index + 1
        t_re = time_sampler.t_index_to_t(t_index=t_index_re)
    else:
        raise ValueError(f"prev_step {prev_step} not recognised.")

    # recycling to get predicted x_start
    noise_re = seg_dm.sample_noise(shape=x_start.shape, dtype=x_start.dtype)
    x_t_re = seg_dm.q_sample(
        x_start=x_start, noise=noise_re, t_index=t_index_re
    )
    model_out_re = seg_dm.model(
        image=image,
        mask=seg_dm.x_to_mask(x_t_re),
        t=t_re,
    )
    x_start_pred_re = seg_dm.predict_xstart_from_model_out_xt(
        model_out=model_out_re,
        x_t=x_t_re,
        t_index=t_index_re,
    )

    # get x_t
    if reverse_step and (prev_step == "next"):
        noise = noise_re  # with interpolation, it's the same noise
        x_t = seg_dm.predict_xprev_from_xstart_xt(
            x_start=x_start_pred_re,
            x_t=x_t_re,
            t_index=t_index_re,
        )
    else:
        noise = seg_dm.sample_noise(
            shape=x_start_pred_re.shape, dtype=x_start_pred_re.dtype
        )
        x_t = seg_dm.q_sample(
            x_start=x_start_pred_re, noise=noise, t_index=t_index
        )

    mask = seg_dm.x_to_mask(x_t)
    mask = lax.stop_gradient(mask)
    model_out = seg_dm.model(
        image=image,
        mask=mask,
        t=t,
    )

    loss_args_re = DiffusionLossArgs(
        t_index=t_index_re,
        x_t=x_t_re,
        noise=noise_re,
        model_out=model_out_re,
    )
    loss_args = DiffusionLossArgs(
        t_index=t_index,
        x_t=x_t,
        noise=noise,
        model_out=model_out,
    )
    return loss_args_re, loss_args, probs_t
