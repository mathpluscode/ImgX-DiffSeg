"""Experiment interface."""
from __future__ import annotations

from pathlib import Path

import chex
import jax
import tensorflow as tf
from flax import jax_utils
from omegaconf import DictConfig

from imgx.data.iterator import get_image_tfds_dataset
from imgx.segmentation.train_state import TrainState
from imgx_datasets import INFO_MAP


class Experiment:
    """Experiment for supervised training."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes experiment.

        Args:
            config: experiment config.
        """
        # Do not use accelerators in data pipeline.
        tf.config.experimental.set_visible_devices([], device_type="GPU")
        tf.config.experimental.set_visible_devices([], device_type="TPU")

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
        self, train_state: TrainState, key: jax.random.PRNGKeyArray
    ) -> tuple[TrainState, jax.random.PRNGKeyArray, chex.ArrayTree]:
        """Perform a training step.

        Args:
            train_state: training state.
            key: random key.

        Returns:
            - new training state.
            - new random key.
            - metric dict.
        """
        raise NotImplementedError

    def eval_step(
        self,
        train_state: TrainState,
        key: jax.random.PRNGKeyArray,
        split: str,
        out_dir: Path | None = None,
    ) -> tuple[jax.random.PRNGKeyArray, chex.ArrayTree]:
        """Evaluation on entire validation data set.

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
