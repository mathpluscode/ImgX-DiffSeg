"""Script to launch training."""
from pathlib import Path

import hydra
import jax
import wandb
from absl import logging
from omegaconf import DictConfig, OmegaConf

from imgx.config import flatten_dict
from imgx.device import broadcast_to_local_devices
from imgx.exp import Experiment
from imgx.exp.train_state import save_ckpt
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
    config.data.trainer.batch_size = (
        n_devices * config.data.trainer.batch_size_per_replica
    )
    config.data.trainer.max_num_samples = 64

    # make logging more frequent
    config.logging.log_freq = 1
    config.logging.save_freq = 4
    return config


def process_config(config: DictConfig) -> DictConfig:
    """Modify attributes based on config.

    Args:
        config: original config.

    Returns:
        modified config.
    """
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
        diff_cls_name = diff_config["_target_"].split(".")[-1]
        if diff_cls_name == "GaussianSegmentationDiffusion":
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(  # pylint:disable=too-many-statements
    config: DictConfig,
) -> None:
    """Main function.

    Args:
        config: config loaded from yaml.
    """
    # update config
    if config.debug:
        config = set_debug_config(config)
    config = process_config(config)
    logging.info(OmegaConf.to_yaml(config))

    # init wandb
    files_dir = None
    if config.logging.wandb.project:
        wandb_run = wandb.init(
            project=config.logging.wandb.project,
            entity=config.logging.wandb.entity,
            config=flatten_dict(dict(config)),
        )
        files_dir = Path(wandb_run.settings.files_dir)
        # backup config
        OmegaConf.save(config=config, f=files_dir / "config_backup.yaml")

    # init devices
    devices = jax.local_devices()
    if config.data.trainer.num_devices_per_replica != 1:
        raise ValueError("Distributed training not supported.")
    logging.info(f"Local devices are: {devices}")

    # init exp
    run = Experiment(config=config)
    train_state = run.train_init()
    run.eval_init()

    logging.info("Start training.")
    rng = jax.random.PRNGKey(config.seed)
    rng = broadcast_to_local_devices(rng)
    batch_size_per_step = get_batch_size_per_step(config)
    max_num_steps = config.data.trainer.max_num_samples // batch_size_per_step
    for i in range(1, 1 + max_num_steps):
        # train step
        train_state, train_scalars = run.train_step(
            train_state=train_state,
        )
        train_scalars = {"train_" + k: v for k, v in train_scalars.items()}
        scalars = {
            "num_samples": i * batch_size_per_step,
            **train_scalars,
        }

        # save checkpoint and eval if needed
        ckpt_dir = None
        if (
            (i > 0)
            and ((i % config.logging.save_freq == 0) or (i == max_num_steps))
            and (files_dir is not None)
        ):
            ckpt_dir = files_dir / "ckpt" / f"batch_{i}"

        if config.eval and (ckpt_dir is not None):
            # when diffusion save under sampler name
            out_dir = ckpt_dir
            if config.task.name == "diffusion_segmentation":
                out_dir = ckpt_dir / config.task.sampler.name

            val_scalars = run.eval_step(
                split=VALID_SPLIT,
                params=train_state.params,
                state=train_state.network_state,
                rng=rng,
                out_dir=out_dir,
                save_predictions=False,
            )
            val_scalars = {"valid_" + k: v for k, v in val_scalars.items()}
            scalars = {
                **scalars,
                **val_scalars,
            }

        # log metrics and print
        if i % config.logging.log_freq == 0:
            # log to wandb
            if config.logging.wandb.project:
                wandb.log(scalars)

            # print to stdout
            scalars = {
                k: v if isinstance(v, int) else f"{v:.2e}"
                for k, v in scalars.items()
            }
            logging.info(f"Batch {i}: {scalars}")

        # save checkpoint and metrics
        if ckpt_dir is not None:
            save_ckpt(
                train_state=train_state,
                ckpt_dir=ckpt_dir,
            )
            # backup config every time
            OmegaConf.save(config=config, f=ckpt_dir / "config.yaml")
            logging.info(f"Checkpoint saved at {ckpt_dir}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
