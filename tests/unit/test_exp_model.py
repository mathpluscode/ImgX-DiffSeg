"""Test mixed precision related functions in factory."""
import haiku as hk
import pytest
from omegaconf import DictConfig

from imgx.exp.model import build_vision_model
from imgx.model import MODEL_CLS_NAME_TO_CONFIG_NAME, SUPPORTED_VISION_MODELS

DUMMY_TASK_CONFIG = {
    "name": "segmentation",
    "diffusion": {
        "num_timesteps": 4,
        "num_timesteps_sample": 20,
        "beta": {
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "beta_end": 0.02,
        },
        "model_out_type": "x_start",
        "model_var_type": "fixed_large",
        "x_space": "scaled_probs",
        "x_limit": 0.0,
    },
}
DUMMY_MODEL_CONFIG = {
    "remat": False,
    "unet3d": {
        "num_channels": [1, 2, 4],
    },
    "unet3d_slice": {
        "num_channels": [1, 2, 4],
    },
    "unet3d_time": {
        "num_channels": [1, 2, 4],
    },
    "unet3d_slice_time": {
        "num_channels": [1, 2, 4],
    },
}


@hk.testing.transform_and_run()
@pytest.mark.parametrize(
    "model_class",
    SUPPORTED_VISION_MODELS,
    ids=SUPPORTED_VISION_MODELS,
)
def test_build_vision_model(model_class: str) -> None:
    """Test all supported models.

    Args:
        model_class: name of model class.
    """
    data_config = {
        "name": "male_pelvic_mr",
    }
    data_config = DictConfig(data_config)
    model_config = DictConfig(DUMMY_MODEL_CONFIG)
    task_config = DictConfig(DUMMY_TASK_CONFIG)
    if model_class.endswith("_time"):
        task_config["name"] = "diffusion"
    model_config["name"] = MODEL_CLS_NAME_TO_CONFIG_NAME[model_class]
    build_vision_model(
        data_config=data_config,
        task_config=task_config,
        model_config=model_config,
    )
