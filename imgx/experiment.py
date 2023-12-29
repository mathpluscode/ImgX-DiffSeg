"""Experiment interface."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import logging
from omegaconf import DictConfig

from imgx.datasets import INFO_MAP
from imgx.metric.util import merge_aggregated_metrics
from imgx.train_state import TrainState


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

        self.config = config
        self.dataset_info = INFO_MAP[self.config.data.name]
        self.p_train_step = None  # To be defined in train_init
        self.p_eval_step = None  # To be defined in train_init

    def train_init(
        self, batch: dict[str, jnp.ndarray], ckpt_dir: Path | None = None, step: int | None = None
    ) -> tuple[TrainState, int]:
        """Initialize data loader, loss, networks for training.

        Args:
            batch: training data.
            ckpt_dir: checkpoint directory to restore from.
            step: checkpoint step to restore from, if None use the latest one.

        Returns:
            initialized training state.
        """
        raise NotImplementedError

    def train_step(
        self, train_state: TrainState, batch: dict[str, jnp.ndarray], key: jax.Array
    ) -> tuple[TrainState, chex.ArrayTree]:
        """Perform a training step.

        Args:
            train_state: training state.
            batch: training data.
            key: random key.

        Returns:
            - new training state.
            - new random key.
            - metric dict.
        """
        # key is updated/fold inside pmap function
        # to ensure a different key is used per step
        train_state, metrics = self.p_train_step(  # pylint: disable=not-callable
            train_state,
            batch,
            key,
        )
        metrics = merge_aggregated_metrics(metrics)
        metrics = jax.tree_map(lambda x: x.item(), metrics)  # tensor to values
        return train_state, metrics

    def eval_step(
        self,
        train_state: TrainState,
        iterator: Iterator[dict[str, jnp.ndarray]],
        num_steps: int,
        key: jax.Array,
        out_dir: Path | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Evaluation on entire validation/test data set.

        Args:
            train_state: training state.
            iterator: data iterator.
            num_steps: number of steps for evaluation.
            key: random key.
            out_dir: output directory, if not None, predictions will be saved.

        Returns:
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
    ) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Evaluate a batch.

        Args:
            train_state: training state.
            key: random key.
            batch: batch data without uid.
            uids: uids in the batch, potentially including padded samples.
            device_cpu: cpu device.

        Returns:
            uids: uids in the batch, excluding padded samples.
            metrics: each item has shape (num_samples,).
            prediction dict: each item has shape (num_samples, ...).
        """
        raise NotImplementedError
