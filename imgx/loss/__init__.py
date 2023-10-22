"""Package for loss functions."""
from imgx.loss.cross_entropy import cross_entropy, focal_loss
from imgx.loss.dice import dice_loss

__all__ = [
    "cross_entropy",
    "focal_loss",
    "dice_loss",
]
