"""Package for models."""
from imgx.model.unet.unet import Unet  # noqa: F401

SUPPORTED_VISION_MODELS = [
    "Unet",
]

__all__ = SUPPORTED_VISION_MODELS
