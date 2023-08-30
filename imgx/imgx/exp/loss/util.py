"""Utility functions for loss."""
import jax.numpy as jnp


def aggregate_batch_scalars(
    scalars: dict[str, jnp.ndarray]
) -> dict[str, jnp.ndarray]:
    """Aggregate batch scalars.

    Args:
        scalars: dict of metrics, each of shape (batch,).

    Returns:
        aggregated metrics.
    """
    agg_scalars = {}
    for key, value in scalars.items():
        agg_scalars[f"mean_{key}"] = jnp.nanmean(value)
        agg_scalars[f"min_{key}"] = jnp.nanmin(value)
        agg_scalars[f"max_{key}"] = jnp.nanmax(value)
    return agg_scalars


def aggregate_batch_scalars_for_diffusion(
    scalars: dict[str, jnp.ndarray],
    t_index: jnp.ndarray,
    num_timesteps: int,
) -> dict[str, jnp.ndarray]:
    """Aggregate batch scalars for diffusion.

    Args:
        scalars: dict of metrics, each of shape (batch,).
        t_index: time of shape (batch, ...), values in [0, num_timesteps).
        num_timesteps: number of timesteps.

    Returns:
        aggregated metrics.
    """
    mask_t_min = t_index == 0  # almost noise-free
    mask_t_max = t_index == (num_timesteps - 1)  # all noise
    agg_scalars = {}
    for key, value in scalars.items():
        agg_scalars[f"mean_{key}_t_min"] = jnp.nanmean(value, where=mask_t_min)
        agg_scalars[f"mean_{key}_t_max"] = jnp.nanmean(value, where=mask_t_max)
    return agg_scalars
