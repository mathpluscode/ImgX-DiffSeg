"""Factory functions."""
from __future__ import annotations

from typing import Callable

from hydra.utils import instantiate
from jax import numpy as jnp
from omegaconf import DictConfig

from imgx.diffusion.time_sampler import TimeSampler
from imgx.exp.loss.diffusion import diffusion_segmentation_loss
from imgx.exp.loss.segmentation import segmentation_loss
from imgx.exp.mixed_precision import set_mixed_precision_policy
from imgx_datasets import INFO_MAP


def build_loss_fn(
    config: DictConfig,
) -> Callable[
    [dict[str, jnp.ndarray]], tuple[jnp.ndarray, dict[str, jnp.ndarray]]
]:
    """Build model from config.

    Args:
        config: entire config.

    Returns:
        Loss function.

    Raises:
        ValueError: if config is wrong or not supported.
    """
    task_config = config.task
    model_cls_name = (
        config.task.model._target_  # pylint: disable=protected-access
    )
    model_cls_name = model_cls_name.split(".")[-1]
    set_mixed_precision_policy(
        use_mp=config.mixed_precision, model_cls_name=model_cls_name
    )

    # number of classes including background
    dataset_info = INFO_MAP[config.data["name"]]

    if task_config["name"] == "segmentation":

        def seg_loss_fn(
            input_dict: dict[str, jnp.ndarray]
        ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
            model = instantiate(config.task.model)
            return segmentation_loss(
                input_dict=input_dict,
                dataset_info=dataset_info,
                model=model,
                loss_config=config.loss,
            )

        return seg_loss_fn
    if task_config["name"] == "diffusion_segmentation":

        def diffusion_loss_fn(
            input_dict: dict[str, jnp.ndarray]
        ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
            diffusion_model = instantiate(
                task_config.diffusion,
                model=config.task.model,  # hydra will instantiate the model
                classes_are_exclusive=dataset_info.classes_are_exclusive,
            )
            time_sampler = TimeSampler(
                num_timesteps=task_config.diffusion.num_timesteps,
                uniform_time_sampling=task_config.uniform_time_sampling,
            )
            return diffusion_segmentation_loss(
                input_dict=input_dict,
                seg_dm=diffusion_model,
                time_sampler=time_sampler,
                dataset_info=dataset_info,
                loss_config=config.loss,
                diff_config=task_config,
            )

        return diffusion_loss_fn
    raise ValueError(f"Unknown task {task_config['name']}.")
