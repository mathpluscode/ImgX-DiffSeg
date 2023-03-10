"""Package for loss functions."""
from imgx.loss.cross_entropy import mean_cross_entropy, mean_focal_loss
from imgx.loss.dice import mean_dice_loss

__all__ = [
    "mean_cross_entropy",
    "mean_focal_loss",
    "mean_dice_loss",
]
