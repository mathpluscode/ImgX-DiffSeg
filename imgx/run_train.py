"""Script to launch training."""
import json
from pathlib import Path

import hydra
import jax
import wandb
from absl import logging
from flax.training import common_utils
from flax.training.early_stopping import EarlyStopping
from omegaconf import DictConfig, OmegaConf

from imgx.config import flatten_dict
from imgx.diffusion_segmentation.experiment import DiffusionSegmentationExperiment
from imgx.experiment import Experiment
from imgx.segmentation.experiment import SegmentationExperiment
from imgx.train_state import save_checkpoint
from imgx_datasets import INFO_MAP
from imgx_datasets.constant import VALID_SPLIT

logging.set_verbosity(logging.INFO)


def set_debug_config(config: DictConfig) -> DictConfig:
    """Modify config for debugging purpose.

    Args:
        config: original config.

    Returns:
        modified config.
    """
    # reduce all model size
    # due to the attention, deeper model reduces the memory usage
    config.task.model.num_channels = (1, 2, 4, 4)

    # make training shorter
    n_devices = jax.local_device_count()
    config.data.loader.max_num_samples_per_split = 11
    config.data.trainer.batch_size_per_replica = 2
    config.data.trainer.batch_size = n_devices * config.data.trainer.batch_size_per_replica
    config.data.trainer.max_num_samples = 256

    # make logging more frequent
    config.logging.log_freq = 1
    config.logging.save_freq = 4

    # stop early
    config.task.early_stopping.patience = 5
    config.task.early_stopping.min_delta = 0.1
    return config


def process_config(config: DictConfig) -> DictConfig:
    """Modify attributes based on config.

    Args:
        config: original config.

    Returns:
        modified config.
    """
    if config.data.trainer.num_devices_per_replica != 1:
        raise ValueError("Distributed training not supported.")
    if config.task.early_stopping.mode not in ["min", "max"]:
        raise ValueError("Early stopping mode must be min or max.")

    dataset_info = INFO_MAP[config.data.name]

    # adjust model num_spatial_dims
    if dataset_info.ndim < config.task.model.num_spatial_dims:
        # model is 3D but dataset is 2D
        # as by default model is 3D
        config.task.model.num_spatial_dims = dataset_info.ndim

    # set model output channels
    out_channels = dataset_info.num_classes
    if config.task.name == "diffusion_segmentation":
        # diffusion model may output variance per class
        diff_config = config.task["diffusion"]
        model_var_type = diff_config["model_var_type"]
        if model_var_type in ["learned", "learned_range"]:
            out_channels *= 2
    config.task.model.out_channels = out_channels

    return config


def get_batch_size_per_step(config: DictConfig) -> int:
    """Return the actual number of samples per step.

    Args:
        config: total config.

    Returns:
        Number of samples across all devices.
    """
    num_devices = jax.local_device_count()
    num_devices_per_replica = config.data.trainer.num_devices_per_replica
    num_models = num_devices // num_devices_per_replica
    batch_size = config.data.trainer.batch_size_per_replica * num_models
    return batch_size


def build_experiment(config: DictConfig) -> Experiment:
    """Build experiment based on config.

    Args:
        config: total config.

    Returns:
        Experiment instance.
    """
    if config.task.name == "segmentation":
        return SegmentationExperiment(config=config)
    if config.task.name == "diffusion_segmentation":
        return DiffusionSegmentationExperiment(config=config)
    raise ValueError(f"Task {config.task.name} not supported.")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(  # pylint:disable=too-many-statements
    config: DictConfig,
) -> None:
    """Main function.

    Args:
        config: config loaded from yaml.
    """
    logging.info(f"Local devices are: {jax.local_devices()}")

    # update config
    if config.debug:
        config = set_debug_config(config)
    config = process_config(config)
    logging.info(OmegaConf.to_yaml(config))

    # init wandb
    wandb_run = wandb.init(
        project=config.logging.wandb.project,
        entity=config.logging.wandb.entity,
        config=flatten_dict(dict(config)),
    )
    files_dir = Path(wandb_run.settings.files_dir)
    # backup config
    OmegaConf.save(config=config, f=files_dir / "config_backup.yaml")
    ckpt_dir = files_dir / "ckpt"

    # init model
    run = build_experiment(config=config)
    train_state, step_offset = run.train_init()
    key = jax.random.PRNGKey(config.seed)
    key = common_utils.shard_prng_key(key)  # each replica has a different key

    logging.info(
        f"Start training with early stopping on {config.task.early_stopping.metric} "
        f"and patience = {config.task.early_stopping.patience}."
    )
    batch_size_per_step = get_batch_size_per_step(config)
    max_num_steps = config.data.trainer.max_num_samples // batch_size_per_step
    early_stop = EarlyStopping(
        min_delta=config.task.early_stopping.min_delta, patience=config.task.early_stopping.patience
    )
    for i in range(1 + step_offset, 1 + max_num_steps):
        train_state, key, train_metrics = run.train_step(train_state, key)
        train_metrics = {"train_" + k: v for k, v in train_metrics.items()}
        metrics = {
            "num_samples": i * batch_size_per_step,
            **train_metrics,
        }

        # save checkpoint if needed
        to_save_ckpt = (i > 0) and ((i % config.logging.save_freq == 0) or (i == max_num_steps))
        if to_save_ckpt:
            ckpt_path = save_checkpoint(
                train_state=train_state,
                ckpt_dir=ckpt_dir,
                # when early stop, it's patience+1 ckpt
                keep=config.task.early_stopping.patience + 1,
            )
            key, val_metrics = run.eval_step(train_state=train_state, key=key, split=VALID_SPLIT)
            out_dir = Path(ckpt_path)
            if config.task.name == "diffusion_segmentation":
                out_dir = out_dir / config.task.sampler.name
                out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "mean_metrics.json", "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, sort_keys=True, indent=4)

            # update early stopping
            early_stop_metric = val_metrics[config.task.early_stopping.metric]
            if config.task.early_stopping.mode == "max":
                early_stop_metric = -early_stop_metric
            _, early_stop = early_stop.update(early_stop_metric)
            logging.info(
                f"Early stop updated {i}: "
                f"should_stop={early_stop.should_stop}, "
                f"best_metric({config.task.early_stopping.metric})={early_stop.best_metric:.4f}, "
                f"patience_count={early_stop.patience_count}, "
                f"min_delta={early_stop.min_delta}, "
                f"patience={early_stop.patience}."
            )

            # update metrics
            # only add prefix after saving to json
            val_metrics = {"valid_" + k: v for k, v in val_metrics.items()}
            metrics = {
                **metrics,
                **val_metrics,
            }
            metrics_str = {k: v if isinstance(v, int) else f"{v:.2e}" for k, v in metrics.items()}
            logging.info(f"Batch {i}: {metrics_str}")

        # log metrics
        if config.logging.wandb.project and (i % config.logging.log_freq == 0):
            wandb.log(metrics)

        # early stopping
        if early_stop.should_stop:
            logging.info(
                f"Met early stopping criteria with {config.task.early_stopping.metric} = "
                f"{early_stop.best_metric} and patience {early_stop.patience_count}, breaking..."
            )
            break


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
