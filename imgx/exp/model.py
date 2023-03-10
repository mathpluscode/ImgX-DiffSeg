"""Module for building models."""
import haiku as hk
from omegaconf import DictConfig

from imgx.datasets import IMAGE_SHAPE_MAP, NUM_CLASSES_MAP
from imgx.diffusion.gaussian_diffusion import (
    DiffusionBetaSchedule,
    DiffusionModelOutputType,
    DiffusionModelVarianceType,
    DiffusionSpace,
    GaussianDiffusion,
)
from imgx.model import Unet3d, Unet3dSlice, Unet3dSliceTime, Unet3dTime


def build_vision_model(
    data_config: DictConfig,
    task_config: DictConfig,
    model_config: DictConfig,
) -> hk.Module:
    """Build model from config.

    Args:
        data_config: have in_shape and out_channels.
        task_config: have task name and configs.
        model_config: have model name attribute.

    Returns:
        Model.

    Raises:
        ValueError: if config is wrong or not supported.
    """
    if model_config.name not in model_config:
        raise ValueError(f"Missing configuration for {model_config.name}.")
    dataset_name = data_config["name"]
    image_shape = IMAGE_SHAPE_MAP[dataset_name]
    num_classes = NUM_CLASSES_MAP[dataset_name]
    if task_config.name == "segmentation":
        # TODO use enum
        in_channels = 1  # forward will expand dimension
        out_channels = num_classes
    elif task_config.name == "diffusion":
        # diffusion model takes the image and a noised mask/logits as input
        in_channels = 1 + num_classes
        # diffusion model may output variance per class
        out_channels = num_classes
        model_var_type = DiffusionModelVarianceType[
            task_config["diffusion"]["model_var_type"].upper()
        ]
        if model_var_type in [
            DiffusionModelVarianceType.LEARNED,
            DiffusionModelVarianceType.LEARNED_RANGE,
        ]:
            out_channels *= 2
    else:
        raise ValueError(f"Unknown task {task_config.name}.")

    total_config = {
        "remat": model_config.remat,
        "in_shape": image_shape,
        "in_channels": in_channels,
        "out_channels": out_channels,
        **model_config[model_config.name],
    }
    if model_config.name == "unet3d":
        return Unet3d(**total_config)
    if model_config.name == "unet3d_slice":
        return Unet3dSlice(**total_config)
    if model_config.name == "unet3d_time":
        num_timesteps = task_config["diffusion"]["num_timesteps"]
        return Unet3dTime(num_timesteps=num_timesteps, **total_config)
    if model_config.name == "unet3d_slice_time":
        num_timesteps = task_config["diffusion"]["num_timesteps"]
        return Unet3dSliceTime(num_timesteps=num_timesteps, **total_config)
    raise ValueError(f"Unknown model {model_config.name}.")


def build_diffusion_model(
    model: hk.Module,
    diffusion_config: DictConfig,
) -> GaussianDiffusion:
    """Build diffusion model from config and vision model.

    Args:
        model: the model used in diffusion.
        diffusion_config: config for diffusion setting.

    Returns:
        A GaussianDiffusion model.
    """
    num_timesteps = diffusion_config["num_timesteps"]
    num_timesteps_beta = diffusion_config["num_timesteps_beta"]
    beta_config = diffusion_config["beta"].copy()
    beta_config["beta_schedule"] = DiffusionBetaSchedule[
        beta_config["beta_schedule"].upper()
    ]
    model_out_type = DiffusionModelOutputType[
        diffusion_config["model_out_type"].upper()
    ]
    model_var_type = DiffusionModelVarianceType[
        diffusion_config["model_var_type"].upper()
    ]
    x_space = DiffusionSpace[diffusion_config["x_space"].upper()]
    x_limit = diffusion_config["x_limit"]
    use_ddim = diffusion_config["use_ddim"]
    return GaussianDiffusion(
        model=model,
        num_timesteps=num_timesteps,
        num_timesteps_beta=num_timesteps_beta,
        model_out_type=model_out_type,
        model_var_type=model_var_type,
        x_space=x_space,
        x_limit=x_limit,
        use_ddim=use_ddim,
        **beta_config,
    )
