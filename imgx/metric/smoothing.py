"""Label/image smoothing functions."""
from typing import Callable

import jax
from jax import lax
from jax import numpy as jnp


def gaussian_kernel(
    num_spatial_dims: int,
    kernel_sigma: float,
    kernel_size: int,
) -> jnp.ndarray:
    """Gaussian kernel for convolution.

    Args:
        num_spatial_dims: number of spatial dimensions.
        kernel_sigma: sigma for Gaussian kernel.
        kernel_size: size for Gaussian kernel.

    Returns:
        Gaussian kernel of shape (window_size, ..., window_size), ndim=num_spatial_dims.
    """
    if kernel_size // 2 == 0:
        raise ValueError(f"kernel_size = {kernel_size} must be odd.")
    # (kernel_size,)
    dist = jnp.arange((1 - kernel_size) / 2, (1 + kernel_size) / 2)
    kernel_1d = jnp.exp(-jnp.power(dist / kernel_sigma, 2) / 2)
    kernel_1d /= kernel_1d.sum()
    if num_spatial_dims == 1:
        return kernel_1d

    kernel_nd = jnp.ones((kernel_size,) * num_spatial_dims)
    for i in range(num_spatial_dims):
        kernel_nd *= jnp.expand_dims(kernel_1d, axis=[j for j in range(num_spatial_dims) if j != i])
    return kernel_nd


def get_conv(
    num_spatial_dims: int,
    kernel_sigma: float,
    kernel_size: int,
    kernel_type: str,
    padding: str,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get Gaussian convolution function.

    The function performs convolution on the spatial dimensions for each feature channel.

    Args:
        num_spatial_dims: number of spatial dimensions.
        kernel_sigma: sigma for Gaussian kernel.
        kernel_size: size for Gaussian kernel.
        kernel_type: type of kernel, "gaussian" or "uniform".
        padding: padding type, "SAME" or "VALID".

    Returns:
        Gaussian convolution function that takes input of shape (batch, ..., channels).
    """
    if num_spatial_dims > 4:
        raise ValueError(f"num_spatial_dims = {num_spatial_dims} must be <= 4.")
    if kernel_type == "gaussian":
        # (window_size, ..., window_size), ndim=num_spatial_dims
        kernel = gaussian_kernel(
            num_spatial_dims,
            kernel_sigma=kernel_sigma,
            kernel_size=kernel_size,
        )
    elif kernel_type == "uniform":
        kernel = jnp.ones((kernel_size,) * num_spatial_dims) / kernel_size**num_spatial_dims
    else:
        raise ValueError(f"kernel_type = {kernel_type} must be 'gaussian' or 'uniform'.")
    # (1,1,window_size, ..., window_size), ndim=num_spatial_dims+2
    kernel = kernel[None, None, ...]

    spatial_dilation = "WHDT"[:num_spatial_dims]
    lhs_spec = f"N{spatial_dilation}C"
    rhs_spec = f"OI{spatial_dilation}"
    out_spec = f"N{spatial_dilation}C"

    def conv(x: jnp.ndarray) -> jnp.ndarray:
        """Convolution function.

        Args:
            x: (batch, ...)

        Returns:
            (batch, ...)
        """
        return lax.conv_general_dilated(
            lhs=x[..., None],
            rhs=kernel.astype(x.dtype),
            dimension_numbers=(lhs_spec, rhs_spec, out_spec),
            window_strides=(1,) * num_spatial_dims,
            padding=padding,
        )[..., 0]

    return jax.vmap(conv, in_axes=-1, out_axes=-1)


def smooth_label(
    mask: jnp.ndarray, classes_are_exclusive: bool, label_smoothing: float = 0.0
) -> jnp.ndarray:
    """Label smoothing.

    If classes are exclusive, the even distribution has p = 1 / num_classes per class.
    If classes are not exclusive, the even distribution has p = 0.5 per class.

    Args:
        mask: probabilities per class, (batch, ..., num_classes).
        classes_are_exclusive: if False, each element can be assigned to multiple classes.
        label_smoothing: label smoothing factor between 0 and 1, 0.0 means no smoothing.

    Returns:
        Label smoothed mask.
    """
    p_even = lax.select(classes_are_exclusive, 1.0 / mask.shape[-1], 0.5)
    return mask * (1 - label_smoothing) + p_even * label_smoothing


def gaussian_smooth_label(
    mask: jnp.ndarray,
    classes_are_exclusive: bool,
    kernel_sigma: float = 1.5,
    kernel_size: int = 3,
) -> jnp.ndarray:
    """Label smoothing using Gaussian kernels.

    The smoothing is performed on the spatial dimensions per class.
    https://arxiv.org/abs/2104.05788

    Args:
        mask: probabilities per class, (batch, ..., num_classes).
        classes_are_exclusive: if False, each element can be assigned to multiple classes.
        kernel_sigma: sigma for Gaussian kernel.
        kernel_size: size for kernel.

    Returns:
        Label smoothed mask.
    """
    mask = get_conv(
        num_spatial_dims=mask.ndim - 2,
        kernel_sigma=kernel_sigma,
        kernel_size=kernel_size,
        kernel_type="gaussian",
        padding="SAME",
    )(mask)
    # if classes are exclusive, the sum should be 1
    mask = lax.cond(
        classes_are_exclusive,
        lambda x: x / jnp.sum(x, axis=-1, keepdims=True),
        lambda x: x,
        mask,
    )
    return mask
