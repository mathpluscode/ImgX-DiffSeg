"""Experiment interface."""
from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import tensorflow as tf
from absl import logging
from flax import jax_utils
from omegaconf import DictConfig

from imgx.data.iterator import get_image_tfds_dataset
from imgx.metric.util import aggregate_pmap_metrics
from imgx.train_state import TrainState
from imgx_datasets import INFO_MAP


class Experiment:
    """Experiment for supervised training."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes experiment.

        Args:
            config: experiment config.
        """
        # Do not use accelerators in data pipeline.
        try:
            tf.config.set_visible_devices([], device_type="GPU")
            tf.config.set_visible_devices([], device_type="TPU")
        except RuntimeError:
            logging.error(
                f"Failed to set visible devices, data set may be using GPU/TPUs. "
                f"Visible GPU devices: {tf.config.get_visible_devices('GPU')}. "
                f"Visible TPU devices: {tf.config.get_visible_devices('TPU')}."
            )

        # save config
        self.config = config

        # init data loaders and networks
        self.dataset_info = INFO_MAP[self.config.data.name]
        self.dataset = get_image_tfds_dataset(
            dataset_name=self.config.data.name,
            config=self.config,
        )
        self.train_iter = self.dataset.train_iter
        self.valid_iter = self.dataset.valid_iter
        self.test_iter = self.dataset.test_iter
        platform = jax.local_devices()[0].platform
        if platform not in ["cpu", "tpu"]:
            self.train_iter = jax_utils.prefetch_to_device(self.train_iter, 2)
            self.valid_iter = jax_utils.prefetch_to_device(self.valid_iter, 2)
            self.test_iter = jax_utils.prefetch_to_device(self.test_iter, 2)

        self.p_train_step = None  # To be defined in train_init
        self.p_eval_step = None  # To be defined in train_init

    def train_init(
        self, ckpt_dir: Path | None = None, step: int | None = None
    ) -> tuple[TrainState, int]:
        """Initialize data loader, loss, networks for training.

        Args:
            ckpt_dir: checkpoint directory to restore from.
            step: checkpoint step to restore from, if None use the latest one.

        Returns:
            initialized training state.
        """
        raise NotImplementedError

    def train_step(
        self, train_state: TrainState, key: jax.Array
    ) -> tuple[TrainState, jax.Array, chex.ArrayTree]:
        """Perform a training step.

        Args:
            train_state: training state.
            key: random key.

        Returns:
            - new training state.
            - new random key.
            - metric dict.
        """
        batch = next(self.train_iter)
        train_state, key, metrics = self.p_train_step(  # pylint: disable=not-callable
            train_state,
            batch,
            key,
        )
        metrics = aggregate_pmap_metrics(metrics)
        metrics = jax.tree_map(lambda x: x.item(), metrics)  # tensor to values
        return train_state, key, metrics

    def eval_step(
        self,
        train_state: TrainState,
        key: jax.Array,
        split: str,
        out_dir: Path | None = None,
    ) -> tuple[jax.Array, chex.ArrayTree]:
        """Evaluation on entire validation/test data set.

        Args:
            train_state: training state.
            key: random key.
            split: split to evaluate.
            out_dir: output directory, if not None, predictions will be saved.

        Returns:
            random key.
            metric dict.
        """
        raise NotImplementedError

    def eval_batch(
        self,
        train_state: TrainState,
        key: jax.Array,
        batch: dict[str, jnp.ndarray],
        uids: list[str],
        device_cpu: jax.Device,
        out_dir: Path | None,
        reference_suffix: str = "mask_preprocessed",
        output_suffix: str = "mask_pred",
    ) -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jax.Array]:
        """Evaluate a batch.

        Args:
            train_state: training state.
            key: random key.
            batch: batch data without uid.
            uids: uids in the batch.
            device_cpu: cpu device.
            out_dir: output directory, if not None, predictions will be saved.
            reference_suffix: suffix of reference image.
            output_suffix: suffix of output image.

        Returns:
            metrics, each item has shape (num_shards*batch,).
            label_pred: predicted label, of shape (num_shards*batch, *spatial_shape).
            key: random key.
        """
        raise NotImplementedError
