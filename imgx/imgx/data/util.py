"""Utility functions."""
from __future__ import annotations

import chex
import jax
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from jax import numpy as jnp

from imgx_datasets.constant import IMAGE


def maybe_pad_batch(
    batch: dict[str, chex.ArrayTree],
    is_train: bool,
    batch_size: int,
    batch_dim: int = 0,
) -> dict[str, chex.ArrayTree]:
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


def tf_to_numpy(batch: dict) -> np.ndarray:
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
