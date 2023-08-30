"""Standard diffusion training."""
import jax.numpy as jnp

from imgx.diffusion import DiffusionSegmentation
from imgx.diffusion.time_sampler import TimeSampler
from imgx.exp.loss.diffusion.args import DiffusionLossArgs


def diffusion_step(
    image: jnp.ndarray,
    x_start: jnp.ndarray,
    seg_dm: DiffusionSegmentation,
    time_sampler: TimeSampler,
) -> tuple[DiffusionLossArgs, jnp.ndarray]:
    """Perform recycling.

    Args:
        image: image, (batch, ..., in_channels).
        x_start: ground truth, (batch, ..., num_classes)
        seg_dm: segmentation diffusion model.
        time_sampler: time sampler for training.

    Returns:
        loss_args: Args for normal step.
        probs_t: probs of sampling t.
    """
    batch_size = image.shape[0]

    # t, t_index, probs_t
    t, t_index, probs_t = time_sampler.sample(
        batch_size=batch_size,
        t_index_min=0,  # inclusive
        t_index_max=seg_dm.num_timesteps,  # exclusive
    )

    # x_t
    noise = seg_dm.sample_noise(shape=x_start.shape, dtype=x_start.dtype)
    x_t = seg_dm.q_sample(x_start=x_start, noise=noise, t_index=t_index)
    model_out = seg_dm.model(
        image=image,
        mask=seg_dm.x_to_mask(x_t),
        t=t,
    )

    loss_args = DiffusionLossArgs(
        t_index=t_index,
        x_t=x_t,
        noise=noise,
        model_out=model_out,
    )
    return loss_args, probs_t
