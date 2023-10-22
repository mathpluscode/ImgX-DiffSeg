"""Variance schedule for diffusion models."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def get_beta_schedule(
    num_timesteps: int,
    beta_schedule: str,
    beta_start: float,
    beta_end: float,
) -> jnp.ndarray:
    """Get variance (beta) schedule for q(x_t | x_{t-1}).

    Args:
        num_timesteps: number of time steps in total, T.
        beta_schedule: schedule for beta.
        beta_start: beta for t=0.
        beta_end: beta for t=T-1.

    Returns:
        Shape (num_timesteps,) array of beta values, for t=0, ..., T-1.
        Values are in ascending order.

    Raises:
        ValueError: for unknown schedule.
    """
    if beta_schedule == "linear":
        return jnp.linspace(
            beta_start,
            beta_end,
            num_timesteps,
        )
    if beta_schedule == "quadradic":
        return (
            jnp.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_timesteps,
            )
            ** 2
        )
    if beta_schedule == "cosine":

        def f(t: float) -> float:
            """Eq 17 in https://arxiv.org/abs/2102.09672.

            Args:
                t: time step with values in [0, 1].

            Returns:
                Cumulative product of alpha.
            """
            return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2

        betas = [0.0]
        alphas_cumprod_prev = 1.0
        for i in range(1, num_timesteps):
            t = i / (num_timesteps - 1)
            alphas_cumprod = f(t)
            beta = 1 - alphas_cumprod / alphas_cumprod_prev
            betas.append(beta)
        return jnp.array(betas) * (beta_end - beta_start) + beta_start

    if beta_schedule == "warmup10":
        num_timesteps_warmup = max(num_timesteps // 10, 1)
        betas_warmup = (
            jnp.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_timesteps_warmup,
            )
            ** 2
        )
        return jnp.concatenate(
            [
                betas_warmup,
                jnp.ones((num_timesteps - num_timesteps_warmup,)) * beta_end,
            ]
        )
    if beta_schedule == "warmup50":
        num_timesteps_warmup = max(num_timesteps // 2, 1)
        betas_warmup = (
            jnp.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_timesteps_warmup,
            )
            ** 2
        )
        return jnp.concatenate(
            [
                betas_warmup,
                jnp.ones((num_timesteps - num_timesteps_warmup,)) * beta_end,
            ]
        )
    raise ValueError(f"Unknown beta_schedule {beta_schedule}.")


def downsample_beta_schedule(
    betas: jnp.ndarray,
    num_timesteps: int,
    num_timesteps_to_keep: int,
) -> jnp.ndarray:
    """Down-sample beta schedule.

    After down-sampling, the first and last values of alphas_cumprod are kept.

    Args:
        betas: beta schedule, shape (num_timesteps,).
            Values are in ascending order.
        num_timesteps: number of time steps in total, T.
        num_timesteps_to_keep: number of time steps to keep.

    Returns:
        Down-sampled beta schedule, shape (num_timesteps_to_keep,).
    """
    if betas.shape != (num_timesteps,):
        raise ValueError(
            f"betas.shape ({betas.shape}) must be equal to (num_timesteps,)=({num_timesteps},)"
        )
    if num_timesteps_to_keep > num_timesteps:
        raise ValueError(
            f"num_timesteps_to_keep ({num_timesteps_to_keep}) "
            f"must be <= num_timesteps ({num_timesteps})"
        )
    if (num_timesteps - 1) % (num_timesteps_to_keep - 1) != 0:
        raise ValueError(
            f"num_timesteps-1={num_timesteps-1} can't be evenly divided by "
            f"num_timesteps_to_keep-1={num_timesteps_to_keep-1}."
        )
    if num_timesteps_to_keep < 2:
        raise ValueError(f"num_timesteps_to_keep ({num_timesteps_to_keep}) must be >= 2.")
    if num_timesteps_to_keep == num_timesteps:
        return betas
    step_scale = (num_timesteps - 1) // (num_timesteps_to_keep - 1)
    alphas_cumprod = jnp.cumprod(1.0 - betas)
    # (num_timesteps_to_keep,)
    alphas_cumprod = alphas_cumprod[::step_scale]

    # recompute betas
    # (num_timesteps_to_keep-1,)
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    # (num_timesteps_to_keep,)
    alphas = jnp.concatenate([alphas_cumprod[:1], alphas])
    betas = 1 - alphas
    return betas
