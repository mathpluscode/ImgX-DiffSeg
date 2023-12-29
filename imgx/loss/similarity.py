"""Image similarity loss functions."""
import jax.numpy as jnp

from imgx import EPS
from imgx.metric import nrmsd, psnr


def psnr_loss(
    image1: jnp.ndarray,
    image2: jnp.ndarray,
    value_range: float = 1.0,
    eps: float = EPS,
) -> jnp.ndarray:
    """Peak signal-to-noise ratio (PSNR) loss.

    Args:
        image1: image of shape (batch, ..., channels).
        image2: image of shape (batch, ..., channels).
        value_range: value range of input images.
        eps: epsilon, if two images are identical, MSE=0.

    Returns:
        PSNR loss of shape (batch,).
    """
    return -psnr(
        image1=image1,
        image2=image2,
        value_range=value_range,
        eps=eps,
    )


def nrmsd_loss(
    image_pred: jnp.ndarray,
    image_true: jnp.ndarray,
    eps: float = EPS,
) -> jnp.ndarray:
    """Normalized root-mean-square-deviation (NRMSD) loss.

    Args:
        image_pred: predicted image of shape (batch, ..., channels).
        image_true: ground truth image of shape (batch, ..., channels).
        eps: epsilon, if two images are identical, MSE=0.

    Returns:
        NRMSD loss of shape (batch,).
    """
    return nrmsd(
        image_pred=image_pred,
        image_true=image_true,
        eps=eps,
    )
