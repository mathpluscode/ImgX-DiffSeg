"""Dataset related classes and functions."""
from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterator
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import tensorflow_datasets as tfds
from absl import logging
from omegaconf import DictConfig

from imgx.data.util import get_foreground_range, maybe_pad_batch, tf_to_numpy
from imgx.datasets.constant import (
    FOREGROUND_RANGE,
    IMAGE,
    LABEL,
    TEST_SPLIT,
    TRAIN_SPLIT,
    UID,
    VALID_SPLIT,
)
from imgx.device import shard
from imgx.train_state import get_half_precision_dtype

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


def remove_uid_from_dict(
    batch: dict[str, tf.Tensor],
) -> dict[str, tf.Tensor]:
    """Create a dict from inputs.

    Args:
        batch: dict potentially having uid.

    Returns:
        Dict not having uid.
    """
    return {k: v for k, v in batch.items() if k != UID}


def add_foreground_range(
    batch: dict[str, tf.Tensor],
) -> dict[str, tf.Tensor]:
    """Add FOREGROUND_RANGE in input dict if there are labels.

    Args:
        batch: dict maybe having label.

    Returns:
        Dict having FOREGROUND_RANGE of shape (ndim, 2).
    """
    foreground_ranges = []
    for k, v in batch.items():
        if LABEL in k:
            # (ndim, 2)
            foreground_ranges.append(get_foreground_range(v))
    if len(foreground_ranges) == 0:
        # no labels
        return batch
    # (num_labels, ndim, 2)
    foreground_range = tnp.stack(foreground_ranges, axis=0)
    # (ndim, 2)
    foreground_range = tnp.stack(
        [tnp.min(foreground_range[:, :, 0], axis=0), tnp.max(foreground_range[:, :, 1], axis=0)],
        axis=-1,
    )
    return {
        FOREGROUND_RANGE: foreground_range,
        **batch,
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
    shuffle_buffer_size = shuffle_buffer_size or (8 * batch_size)

    # download data
    builder.download_and_prepare()

    # each host is responsible for a fixed subset of data
    if is_train:
        split = tfds.even_splits(split, jax.process_count())[jax.process_index()]
    dataset = builder.as_dataset(
        split=split,
    )

    # shrink data set if required
    if max_num_samples > 0:
        logging.info(f"Taking first {max_num_samples} data samples for split {split}.")
        dataset = dataset.take(max_num_samples)

    # caching
    dataset = dataset.cache()

    num_steps = -1  # not set for training
    if is_train:
        # first repeat then batch
        dataset = dataset.repeat()
        # augmentation should be done after repeat for true randomness
        # remove uid and calculate foreground range (deterministic)
        dataset = dataset.map(
            remove_uid_from_dict,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.map(
            add_foreground_range,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        # shuffle after augmentation to avoid loading non-augmented images into buffer
        dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        # first batch then repeat
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
            for k in batch:
                if IMAGE in k:
                    batch[k] = tf.cast(batch[k], tf.dtypes.as_dtype(dtype))
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
) -> tuple[Iterator[dict[str, np.ndarray]], int]:
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
