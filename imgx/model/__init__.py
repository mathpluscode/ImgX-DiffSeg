"""Package for models."""
from imgx.model.unet_3d import Unet3d  # noqa: F401
from imgx.model.unet_3d_slice import Unet3dSlice  # noqa: F401
from imgx.model.unet_3d_slice_time import Unet3dSliceTime  # noqa: F401
from imgx.model.unet_3d_time import Unet3dTime  # noqa: F401

SUPPORTED_VISION_MODELS = [
    "Unet3d",
    "Unet3dSlice",
    "Unet3dTime",
    "Unet3dSliceTime",
]

# TODO uniform name
MODEL_CLS_NAME_TO_CONFIG_NAME = {
    "Unet3d": "unet3d",
    "Unet3dSlice": "unet3d_slice",
    "Unet3dTime": "unet3d_time",
    "Unet3dSliceTime": "unet3d_slice_time",
}
CONFIG_NAME_TO_MODEL_CLS_NAME = {
    v: k for k, v in MODEL_CLS_NAME_TO_CONFIG_NAME.items()
}
__all__ = SUPPORTED_VISION_MODELS
