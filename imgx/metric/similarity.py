"""Image similarity metrics."""

import jax.numpy as jnp

from imgx import EPS
from imgx.metric.smoothing import get_conv


def ssim(
    image1: jnp.ndarray,
    image2: jnp.ndarray,
    max_val: float = 1.0,
    kernel_sigma: float = 1.5,
    kernel_size: int = 11,
    kernel_type: str = "gaussian",
    k1: float = 0.01,
    k2: float = 0.03,
) -> jnp.ndarray:
    """Calculate Structural similarity index metric (SSIM).

    https://en.wikipedia.org/wiki/Structural_similarity
    https://github.com/Project-MONAI/MONAI/blob/ccd32ca5e9e84562d2f388b45b6724b5c77c1f57/monai/metrics/regression.py#L240

    SSIM is calculated per window. The window size is specified by `window_size`.
    The window is convolved with a Gaussian kernel specified by `kernel_sigma`.

    SSIM as loss is not stable and may cause NaNs gradients.
    https://github.com/tensorflow/tensorflow/issues/50400
    https://github.com/tensorflow/tensorflow/issues/57353

    Args:
        image1: image of shape (batch, ..., channels).
        image2: image of shape (batch, ..., channels).
        max_val: maximum value of input images, minimum is assumed to be zero.
        kernel_sigma: sigma for Gaussian kernel.
        kernel_size: size for kernel.
        kernel_type: type of kernel, "gaussian" or "uniform".
        k1: stability constant for luminance.
        k2: stability constant for contrast.

    Returns:
        SSIM of shape (batch,).
    """
    num_spatial_dims = image1.ndim - 2
    conv = get_conv(
        num_spatial_dims=num_spatial_dims,
        kernel_sigma=kernel_sigma,
        kernel_size=kernel_size,
        kernel_type=kernel_type,
        padding="VALID",
    )

    # (batch, ..., channels)
    # the spatial dims are reduced by the conv
    mean1 = conv(image1)
    mean2 = conv(image2)
    mean11 = conv(image1 * image1)
    mean12 = conv(image1 * image2)
    mean22 = conv(image2 * image2)
    var1 = mean11 - mean1 * mean1
    var2 = mean22 - mean2 * mean2
    covar12 = mean12 - mean1 * mean2

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numerator = (2 * mean1 * mean2 + c1) * (2 * covar12 + c2)
    denominator = (mean1**2 + mean2**2 + c1) * (var1 + var2 + c2)

    # (batch,)
    return jnp.mean(numerator / denominator, axis=tuple(range(1, image1.ndim)))


def psnr(
    image1: jnp.ndarray,
    image2: jnp.ndarray,
    value_range: float = 1.0,
    eps: float = EPS,
) -> jnp.ndarray:
    """Calculate Peak signal-to-noise ratio (PSNR) metric.

    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    https://github.com/Project-MONAI/MONAI/blob/ccd32ca5e9e84562d2f388b45b6724b5c77c1f57/monai/metrics/regression.py#L186

    Args:
        image1: image of shape (batch, ..., channels).
        image2: image of shape (batch, ..., channels).
        value_range: value range of input images.
        eps: epsilon, if two images are identical, MSE=0.

    Returns:
        PSNR of shape (batch,).
    """
    mse = jnp.mean((image1 - image2) ** 2, axis=range(1, image1.ndim))
    mse = jnp.maximum(mse, eps)
    return 20 * jnp.log10(value_range) - 10 * jnp.log10(mse)


def nrmsd(
    image_pred: jnp.ndarray,
    image_true: jnp.ndarray,
    eps: float = EPS,
) -> jnp.ndarray:
    """Calculate normalized root-mean-square-deviation (NRMSD) metric.

    The normalization is performed by the mean of ground truth image.
    https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Args:
        image_pred: predicted image of shape (batch, ..., channels).
        image_true: ground truth image of shape (batch, ..., channels).
        eps: epsilon, if two images are identical, MSE=0.

    Returns:
        NRMSD of shape (batch,).
    """
    mse = jnp.mean((image_pred - image_true) ** 2, axis=range(1, image_true.ndim))
    rmsd = jnp.sqrt(mse)
    denominator = jnp.maximum(jnp.mean(image_true, axis=range(1, image_true.ndim)), eps)
    return rmsd / denominator
