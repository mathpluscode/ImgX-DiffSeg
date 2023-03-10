"""Script to launch training."""
from pathlib import Path

import hydra
import jax
import wandb
from absl import logging
from omegaconf import DictConfig, OmegaConf

from imgx import VALID_SPLIT
from imgx.config import flatten_dict
from imgx.exp import Experiment
from imgx.exp.train_state import get_eval_params_and_state, save_ckpt

logging.set_verbosity(logging.INFO)


def set_debug_config(config: DictConfig) -> DictConfig:
    """Modify config for debugging purpose.

    Args:
        config: original config.

    Returns:
        modified config.

    Raises:
        ValueError: if data set is unknown.
    """
    # reduce all model size
    config.model.unet3d.num_channels = (1, 2, 4, 8)
    config.model.unet3d_slice.num_channels = (1, 2, 4, 8)
    config.model.unet3d_time.num_channels = (1, 2, 4, 8)
    config.model.unet3d_slice_time.num_channels = (1, 2, 4, 8)

    # make training shorter
    n_devices = jax.local_device_count()
    config.data.max_num_samples = 11
    config.training.batch_size_per_replica = 2
    config.training.batch_size = (
        n_devices * config.training.batch_size_per_replica
    )
    config.training.max_num_samples = 100

    # make logging more frequent
    config.logging.eval_freq = 1
    config.logging.save_freq = 4
    return config


def get_batch_size_per_step(config: DictConfig) -> int:
    """Return the actual number of samples per step.

    Args:
        config: total config.

    Returns:
        Number of samples across all devices.
    """
    if "batch_size_per_replica" not in config["training"]:
        logging.warning("Batch size per step is not accurate.")
        return 1
    num_devices = jax.local_device_count()
    num_devices_per_replica = config["training"]["num_devices_per_replica"]
    num_models = num_devices // num_devices_per_replica
    batch_size = config["training"]["batch_size_per_replica"] * num_models
    return batch_size


@hydra.main(
    version_base=None, config_path="conf", config_name="config_segmentation"
)
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
    if config.training.num_devices_per_replica != 1:
        raise ValueError("Distributed training not supported.")
    logging.info(f"Local devices are: {devices}")

    # init exp
    run = Experiment(config=config)
    train_state = run.train_init()
    run.eval_init()

    logging.info("Start training.")
    batch_size_per_step = get_batch_size_per_step(config)
    max_num_steps = config.training.max_num_samples // batch_size_per_step
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

        to_save_ckpt = (
            (i > 0)
            and (i % config.logging.save_freq == 0)
            and (files_dir is not None)
        )
        # evaluate if saving ckpt or time to evaluate
        to_eval = to_save_ckpt or (i % config.logging.eval_freq == 0)
        if to_save_ckpt and (files_dir is not None):
            ckpt_dir = files_dir / "ckpt" / f"batch_{i}"
        else:
            ckpt_dir = None

        if to_eval and config.eval:
            # TODO on TPU evaluation causes OOM
            params, state = get_eval_params_and_state(train_state)
            val_scalars = run.eval_step(
                split=VALID_SPLIT,
                params=params,
                state=state,
                rng=jax.random.PRNGKey(config.seed),
                out_dir=ckpt_dir,
                save_predictions=False,
            )
            val_scalars = {"valid_" + k: v for k, v in val_scalars.items()}
            scalars = {
                **scalars,
                **val_scalars,
            }
        if config.logging.wandb.project:
            wandb.log(scalars)
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
