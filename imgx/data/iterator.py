"""Dataset related classes and functions."""
from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterator
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
from omegaconf import DictConfig

from imgx.data.util import get_foreground_range, maybe_pad_batch, tf_to_numpy
from imgx.device import shard
from imgx.optim import get_half_precision_dtype
from imgx_datasets.constant import (
    FOREGROUND_RANGE,
    IMAGE,
    LABEL,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALID_SPLIT,
)

DatasetIterator = namedtuple(
    "DatasetIterator",
    [
        "train_iter",
        "valid_iter",
        "test_iter",
        # valid_iter is repeated, needs to know how many batches to eval
        "num_valid_steps",
        # for simplicity, better to know how many batches to eval
        "num_test_steps",
    ],
)


def create_image_label_dict_from_dict(
    x: dict[str, tf.Tensor],
) -> dict[str, tf.Tensor]:
    """Create a dict from inputs.

    Args:
        x: dict having image, label, and other tensors.

    Returns:
        Dict having image and label.
    """
    return {
        IMAGE: x[IMAGE],
        LABEL: x[LABEL],
    }


def add_foreground_range_in_dict(
    x: dict[str, tf.Tensor],
) -> dict[str, tf.Tensor]:
    """Add FOREGROUND_RANGE in input dict.

    Args:
        x: dict having some attributes.

    Returns:
        Dict having FOREGROUND_RANGE.
    """
    return {
        FOREGROUND_RANGE: get_foreground_range(x[LABEL]),
        **x,
    }


def load_split_from_image_tfds_builder(
    builder: tfds.core.DatasetBuilder,
    batch_size: int,
    split: str,
    shuffle_buffer_size: int | None = None,
    shuffle_seed: int = 0,
    max_num_samples: int = -1,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[tf.data.Dataset, int]:
    """Loads a split from a TensorFlow Dataset compatible builder.

    https://github.com/google-research/scenic/blob/main/scenic/dataset_lib/dataset_utils.py

    Args:
        builder: A TFDS compatible dataset builder.
        batch_size: The batch size returned by the data pipeline.
        split: Name of  the split to be loaded.
        shuffle_buffer_size: Size of the tf.data.dataset shuffle buffer.
        shuffle_seed: Seed for shuffling the training data.
        max_num_samples: maximum number of samples to consider.
        dtype: data type for images.

    Returns:
        - A repeated dataset.
        - Number of steps after batch if the dataset is not repeated,
          returns -1 for training.
    """
    is_train = split == TRAIN_SPLIT
    # Prepare arguments.
    shuffle_buffer_size = shuffle_buffer_size or (8 * batch_size)

    # Download data.
    builder.download_and_prepare()

    # Each host is responsible for a fixed subset of data.
    if is_train:
        split = tfds.even_splits(split, jax.process_count())[jax.process_index()]
    dataset = builder.as_dataset(
        split=split,
    )

    # Shrink data set if required
    if max_num_samples > 0:
        logging.info(f"Taking first {max_num_samples} data samples for split {split}.")
        dataset = dataset.take(max_num_samples)

    # Caching.
    dataset = dataset.cache()

    num_steps = -1  # not set for training
    if is_train:
        # First repeat then batch.
        dataset = dataset.repeat()
        # Augmentation should be done after repeat for true randomness.
        # remove uid and calculate foreground range (deterministic)
        dataset = dataset.map(
            create_image_label_dict_from_dict,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.map(
            add_foreground_range_in_dict,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        # Shuffle after augmentation to avoid loading non-augmented images into
        # buffer.
        dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        # First batch then repeat.
        dataset = dataset.batch(batch_size, drop_remainder=False)
        num_steps = tf.data.experimental.cardinality(dataset).numpy()
        if split == VALID_SPLIT:
            # repeat dataset for validation
            dataset = dataset.repeat()

    # NOTE: You may be tempted to move the casting earlier on in the pipeline,
    # but for bf16 some operations will end up silently placed on the TPU and
    # this causes stalls while TF and JAX battle for the accelerator.
    if dtype != jnp.float32:

        def cast_fn(batch: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
            batch[IMAGE] = tf.cast(batch[IMAGE], tf.dtypes.as_dtype(dtype))
            return batch

        dataset = dataset.map(cast_fn)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, num_steps


def get_image_iterator(
    builder: tfds.core.DatasetBuilder,
    split: str,
    is_train: bool,
    batch_size_per_replica: int,
    num_replicas: int,
    shuffle_seed: int,
    max_num_samples: int,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[Iterator, int]:
    """Returns iterator from builder.

    Args:
        builder: data set builder.
        split: split name.
        is_train: if the split is for training.
        batch_size_per_replica: Number of samples consumed per model per step.
        num_replicas: number of model replicas.
        shuffle_seed: Seed for shuffling the training data.
        max_num_samples: maximum number of samples in iterator.
        dtype: data type for images.

    Returns:
        - Batch iterator.
        - Number of steps after batch if the dataset is not repeated,
          returns -1 for training.
    """
    batch_size = batch_size_per_replica * num_replicas
    dataset, num_steps = load_split_from_image_tfds_builder(
        builder=builder,
        batch_size=batch_size,
        split=split,
        shuffle_seed=shuffle_seed,
        max_num_samples=max_num_samples,
        dtype=dtype,
    )
    maybe_pad_batches = partial(maybe_pad_batch, is_train=is_train, batch_size=batch_size)

    dataset_iter = iter(dataset)
    dataset_iter = map(tf_to_numpy, dataset_iter)
    dataset_iter = map(maybe_pad_batches, dataset_iter)

    shard_batches = partial(shard, num_replicas=num_replicas)
    dataset_iter = map(shard_batches, dataset_iter)
    return dataset_iter, num_steps


def get_image_tfds_dataset(
    dataset_name: str,
    config: DictConfig,
) -> DatasetIterator:
    """Returns generators for the dataset train, valid, and test sets.

    TODO: not necessary to init all iterators at once.

    Args:
        dataset_name: Data set name.
        config: entire config.

    Returns:
        A Dataset() which includes train_iter, valid_iter, and test_iter.
    """
    num_devices = jax.local_device_count()
    batch_size_per_replica = config.data.trainer.batch_size_per_replica
    num_devices_per_replica = config.data.trainer.num_devices_per_replica
    num_replicas = num_devices // num_devices_per_replica
    shuffle_seed = config.seed
    max_num_samples_per_split = config.data.loader.max_num_samples_per_split

    dtype = get_half_precision_dtype(config.half_precision)

    builder = tfds.builder(dataset_name)
    train_iter, _ = get_image_iterator(
        builder=builder,
        split=TRAIN_SPLIT,
        is_train=True,
        batch_size_per_replica=batch_size_per_replica,
        num_replicas=num_replicas,
        shuffle_seed=shuffle_seed,
        max_num_samples=max_num_samples_per_split,
        dtype=dtype,
    )
    valid_iter, num_valid_steps = get_image_iterator(
        builder=builder,
        split=VALID_SPLIT,
        is_train=False,
        batch_size_per_replica=batch_size_per_replica,
        num_replicas=num_replicas,
        shuffle_seed=shuffle_seed,
        max_num_samples=max_num_samples_per_split,
        dtype=dtype,
    )
    test_iter, num_test_steps = get_image_iterator(
        builder=builder,
        split=TEST_SPLIT,
        is_train=False,
        batch_size_per_replica=batch_size_per_replica,
        num_replicas=num_replicas,
        shuffle_seed=shuffle_seed,
        max_num_samples=max_num_samples_per_split,
        dtype=dtype,
    )
    return DatasetIterator(
        train_iter=train_iter,
        valid_iter=valid_iter,
        test_iter=test_iter,
        num_valid_steps=num_valid_steps,
        num_test_steps=num_test_steps,
    )
