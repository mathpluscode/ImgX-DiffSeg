"""Util functions for image.

Some are adapted from
https://github.com/google-research/scenic/blob/03735eb81f64fd1241c4efdb946ea6de3d326fe1/scenic/dataset_lib/dataset_utils.py
"""
import functools
import queue
import threading
from typing import Any, Callable, Dict, Generator, Iterable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from absl import logging

from imgx import IMAGE


def maybe_pad_batch(
    batch: Dict[str, chex.ArrayTree],
    is_train: bool,
    batch_size: int,
    batch_dim: int = 0,
) -> Dict[str, chex.ArrayTree]:
    """Zero pad the batch on the right to the batch_size.

    All leave tensors in the batch pytree will be padded. This function expects
    the root structure of the batch pytree to be a dictionary and returns a
    dictionary with the same structure (and substructures), additionally with
    the key 'batch_mask' added to the root dict, with 1.0 indicating indices
    which are true data and 0.0 indicating a padded index. `batch_mask` will
    be used for calculating the weighted cross entropy, or weighted accuracy.
    Note that in this codebase, we assume we drop the last partial batch from
    the training set, so if the batch is from the training set
    (i.e. `train=True`), or when the batch is from the test/validation set,
    but it is a complete batch, we *modify* the batch dict by adding an array of
    ones as the `batch_mask` of all examples in the batch. Otherwise, we create
    a new dict that has the padded patch and its corresponding `batch_mask`
    array. Note that batch_mask can be also used as the label mask
    (not input mask), for task that are pixel/token level. This is simply done
    by applying the mask we make for padding the partial batches on top of
    the existing label mask.

    Args:
          batch: A dictionary containing a pytree. If `inputs_key` is not
            set, we use the first leave to get the current batch size.
            Otherwise, the tensor mapped with `inputs_key`
            at the root dictionary is used.
          is_train: if the batch is from the training data. In that case,
            we drop the last (incomplete) batch and thus don't do any padding.
          batch_size: All arrays in the dict will be padded to have first
            dimension equal to desired_batch_size.
          batch_dim: Batch dimension. The default is 0, but it can be different
            if a sharded batch is given.

    Returns:
        A dictionary mapping the same keys to the padded batches.
          Additionally, we add a key representing weights, to indicate how
          the batch was padded.

    Raises:
        ValueError: if configs are conflicting.
    """
    sample_tensor = batch[IMAGE]
    batch_pad = batch_size - sample_tensor.shape[batch_dim]

    if is_train and batch_pad != 0:
        raise ValueError(
            "In this codebase, we assumed that we always drop the "
            "last partial batch of the train set. Please use "
            "` drop_remainder=True` for the training set."
        )

    # Most batches do not need padding, so we quickly return to avoid slowdown.
    if is_train or batch_pad == 0:
        return batch

    def zero_pad(array: np.ndarray) -> np.ndarray:
        pad_with = (
            [(0, 0)] * batch_dim
            + [(0, batch_pad)]
            + [(0, 0)] * (array.ndim - batch_dim - 1)
        )
        return np.pad(array, pad_with, mode="constant")

    padded_batch = jax.tree_map(zero_pad, batch)
    return padded_batch


def unpad(
    pytree: chex.ArrayTree,
    num_samples: int,
) -> chex.ArrayTree:
    """Remove padded data for all arrays in the pytree.

    We assume that all arrays in the pytree have the same leading dimension.

    Args:
        pytree: A pytree of arrays to be sharded.
        num_samples: number of samples to keep.

    Returns:
      Data without padding
    """

    def _unpad_array(x: jnp.ndarray) -> jnp.ndarray:
        return x[:num_samples, ...]

    return jax.tree_map(_unpad_array, pytree)


def tf_to_numpy(batch: Dict) -> np.ndarray:
    """Convert an input batch from tf Tensors to numpy arrays.

    Args:
        batch: A dictionary that has items in a batch: image and labels.

    Returns:
        Numpy arrays of the given tf Tensors.
    """

    def convert_data(x: tf.Tensor) -> np.ndarray:
        """Use _numpy() for zero-copy conversion between TF and NumPy.

        Args:
            x: tf tensor.

        Returns:
            numpy array.
        """
        return x._numpy()  # pylint: disable=protected-access

    return jax.tree_map(convert_data, batch)


def get_center_pad_shape(
    current_shape: Tuple[int, ...], target_shape: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Get pad sizes for sitk.ConstantPad.

    The padding is added symmetrically.

    Args:
        current_shape: current shape of the image.
        target_shape: target shape of the image.

    Returns:
        pad_lower: shape to pad on the lower side.
        pad_upper: shape to pad on the upper side.
    """
    pad_lower = []
    pad_upper = []
    for i, size_i in enumerate(current_shape):
        pad_i = max(target_shape[i] - size_i, 0)
        pad_lower_i = pad_i // 2
        pad_upper_i = pad_i - pad_lower_i
        pad_lower.append(pad_lower_i)
        pad_upper.append(pad_upper_i)
    return tuple(pad_lower), tuple(pad_upper)


def get_center_crop_shape(
    current_shape: Tuple[int, ...], target_shape: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Get crop sizes for sitk.Crop.

    The crop is performed symmetrically.

    Args:
        current_shape: current shape of the image.
        target_shape: target shape of the image.

    Returns:
        crop_lower: shape to pad on the lower side.
        crop_upper: shape to pad on the upper side.
    """
    crop_lower = []
    crop_upper = []
    for i, size_i in enumerate(current_shape):
        crop_i = max(size_i - target_shape[i], 0)
        crop_lower_i = crop_i // 2
        crop_upper_i = crop_i - crop_lower_i
        crop_lower.append(crop_lower_i)
        crop_upper.append(crop_upper_i)
    return tuple(crop_lower), tuple(crop_upper)


def try_to_get_center_crop_shape(
    label_min: int, label_max: int, current_length: int, target_length: int
) -> Tuple[int, int]:
    """Try to crop at the center of label, 1D.

    Args:
        label_min: label index minimum, inclusive.
        label_max: label index maximum, exclusive.
        current_length: current image length.
        target_length: target image length.

    Returns:
        crop_lower: shape to pad on the lower side.
        crop_upper: shape to pad on the upper side.

    Raises:
        ValueError: if label min max is out of range.
    """
    if label_min < 0 or label_max > current_length:
        raise ValueError("Label index out of range.")

    if current_length <= target_length:
        # no need of crop
        return 0, 0
    # attend to perform crop centered at label center
    label_center = (label_max - 1 + label_min) / 2.0
    bbox_lower = int(np.ceil(label_center - target_length / 2.0))
    bbox_upper = bbox_lower + target_length
    # if lower is negative, then have to shift the window to right
    bbox_lower = max(bbox_lower, 0)
    # if upper is too large, then have to shift the window to left
    if bbox_upper > current_length:
        bbox_lower -= bbox_upper - current_length
    # calculate crop
    crop_lower = bbox_lower  # bbox index starts at 0
    crop_upper = current_length - target_length - crop_lower
    return crop_lower, crop_upper


def get_center_crop_shape_from_bbox(
    bbox_min: Tuple[int, ...],
    bbox_max: Tuple[int, ...],
    current_shape: Tuple[int, ...],
    target_shape: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Get crop sizes for sitk.Crop from label bounding box.

    The crop is not necessarily performed symmetrically.

    Args:
        bbox_min: [start_in_1st_spatial_dim, ...], inclusive, starts at zero.
        bbox_max: [end_in_1st_spatial_dim, ...], exclusive, starts at zero.
        current_shape: current shape of the image.
        target_shape: target shape of the image.

    Returns:
        crop_lower: shape to pad on the lower side.
        crop_upper: shape to pad on the upper side.
    """
    crop_lower = []
    crop_upper = []
    for i, current_length in enumerate(current_shape):
        crop_lower_i, crop_upper_i = try_to_get_center_crop_shape(
            label_min=bbox_min[i],
            label_max=bbox_max[i],
            current_length=current_length,
            target_length=target_shape[i],
        )
        crop_lower.append(crop_lower_i)
        crop_upper.append(crop_upper_i)
    return tuple(crop_lower), tuple(crop_upper)


def get_foreground_range(label: tf.Tensor) -> tf.Tensor:
    """Get the foreground range for a given label.

    This function is not defined in jax for augmentation because,
    nonzero is not jittable as the number of nonzero elements is unknown.

    Args:
        label: shape (d1, ..., dn), here n = ndim below.

    Returns:
        shape (ndim, 2), for each dimension, it's [min, max].
    """
    # (ndim, num_nonzero)
    nonzero_indices = tnp.stack(tnp.nonzero(label))
    # (ndim, 2)
    return tnp.stack(
        [tnp.min(nonzero_indices, axis=-1), tnp.max(nonzero_indices, axis=-1)],
        axis=-1,
    )


def get_function_name(function: Callable[..., Any]) -> str:
    """Get name of any function.

    Args:
        function: function to query.

    Returns:
        function name.
    """
    if isinstance(function, functools.partial):
        return f"partial({function.func.__name__})"
    return function.__name__


def py_prefetch(
    iterable_function: Callable[[], Iterable[chex.ArrayTree]],
    buffer_size: int = 5,
) -> Generator[chex.ArrayTree, None, None]:
    """Performs prefetching of elements from an iterable in a separate thread.

    Args:
        iterable_function: A python function that when called with no arguments
            returns an iterable. This is used to build a fresh iterable for each
            thread (crucial if working with tensorflow datasets because
            `tf.graph` objects are thread local).
        buffer_size (int): Number of elements to keep in the prefetch buffer.

    Yields:
        Prefetched elements from the original iterable.

    Raises:
        ValueError: if the buffer_size <= 1.
            Any error thrown by the iterable_function. Note this is not
            raised inside the producer, but after it finishes executing.
    """
    if buffer_size <= 1:
        raise ValueError("the buffer_size should be > 1")

    buffer: queue.Queue = queue.Queue(maxsize=(buffer_size - 1))
    producer_error = []
    end = object()

    def producer() -> None:
        """Enqueues items from iterable on a given thread."""
        try:
            # Build a new iterable for each thread. This is crucial if
            # working with tensorflow datasets
            # because tf.graph objects are thread local.
            iterable = iterable_function()
            for item in iterable:
                buffer.put(item)
        except Exception as err:  # pylint: disable=broad-except
            logging.exception(
                "Error in producer thread for %s",
                get_function_name(iterable_function),
            )
            producer_error.append(err)
        finally:
            buffer.put(end)

    threading.Thread(target=producer, daemon=True).start()

    # Consumer.
    while True:
        value = buffer.get()
        if value is end:
            break
        yield value

    if producer_error:
        raise producer_error[0]
