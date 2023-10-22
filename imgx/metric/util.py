"""Util functions."""
from __future__ import annotations

import chex
import jax.numpy as jnp


def aggregate_metrics(metrics: dict[str, chex.ArrayTree]) -> dict[str, chex.ArrayTree]:
    """Calculate aggregated metrics.

    Args:
        metrics: metric dict.

    Returns:
        aggregated metric dict.
    """
    agg_metrics = {}
    for key, value in metrics.items():
        agg_metrics[f"mean_{key}"] = jnp.nanmean(value)
        agg_metrics[f"min_{key}"] = jnp.nanmin(value)
        agg_metrics[f"max_{key}"] = jnp.nanmax(value)
    return agg_metrics


def aggregate_metrics_for_diffusion(
    metrics: dict[str, jnp.ndarray],
    t_index: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Aggregate metrics for diffusion.

    Args:
        metrics: dict of metrics, each of shape (batch,).
        t_index: time of shape (batch, ...), values in [0, num_timesteps).

    Returns:
        aggregated metrics.
    """
    mask_t_min = t_index == jnp.min(t_index)
    mask_t_max = t_index == jnp.max(t_index)
    metrics_diff = {}
    for key, value in metrics.items():
        metrics_diff[f"{key}_t_min"] = jnp.nanmean(value, where=mask_t_min)
        metrics_diff[f"{key}_t_max"] = jnp.nanmean(value, where=mask_t_max)
    return metrics_diff


def aggregate_pmap_metrics(metrics: dict[str, chex.ArrayTree]) -> dict[str, chex.ArrayTree]:
    """Aggregate metrics across replicates.

    Args:
        metrics: metric dict from pmap.

    Returns:
        aggregated metric dict.
    """
    min_metrics = {}
    max_metrics = {}
    mean_metrics = {}
    for k in metrics:
        if k.startswith("min_"):
            min_metrics[k] = jnp.nanmin(metrics[k])
        elif k.startswith("max_"):
            max_metrics[k] = jnp.nanmax(metrics[k])
        else:
            mean_metrics[k] = jnp.nanmean(metrics[k])
    return {
        **min_metrics,
        **max_metrics,
        **mean_metrics,
    }


def flatten_diffusion_metrics(metrics: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """Flatten metrics dict for diffusion models.

    Args:
        metrics: dict of metrics, each value of shape (batch, num_steps).

    Returns:
        metrics: dict of metrics, each value of shape (batch, ).
    """
    metrics_flatten = {}
    for k, v in metrics.items():
        for i in range(v.shape[-1]):
            metrics_flatten[f"{k}_step_{i}"] = v[..., i]
        metrics_flatten[k] = v[..., -1]
    return metrics_flatten
