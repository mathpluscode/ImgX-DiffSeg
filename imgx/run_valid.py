"""Script to launch evaluation on validation tests."""
import argparse
import json
from pathlib import Path

import jax
from absl import logging
from flax.training import common_utils
from omegaconf import DictConfig, OmegaConf

from imgx.run_train import build_experiment
from imgx_datasets.constant import VALID_SPLIT

logging.set_verbosity(logging.INFO)


def get_checkpoint_steps(
    log_dir: Path,
) -> list[int]:
    """Get the steps of all available checkpoints.

    Args:
        log_dir: Directory of entire log.

    Returns:
        A list of available steps.

    Raises:
        ValueError: if any file not found.
    """
    ckpt_dir = log_dir / "files" / "ckpt"
    steps = []
    for step_dir in ckpt_dir.glob("checkpoint_*/"):
        if not step_dir.is_dir():
            continue
        ckpt_path = step_dir / "checkpoint"
        if not ckpt_path.exists():
            continue
        steps.append(int(step_dir.stem.split("_")[-1]))
    return sorted(steps, reverse=True)


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log_dir",
        type=Path,
        help="Folder of wandb.",
        default=None,
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        help="Number of sampling steps for diffusion_segmentation.",
        default=-1,
    )
    parser.add_argument(
        "--sampler",
        type=str,
        help="Sampling algorithm for diffusion_segmentation.",
        default="",
        choices=["", "DDPM", "DDIM"],
    )
    args = parser.parse_args()

    return args


def load_and_parse_config(
    log_dir: Path,
    num_timesteps: int,
    sampler: str,
) -> DictConfig:
    """Load and parse config.

    Args:
        log_dir: Directory of entire log.
        num_timesteps: Number of sampling steps for diffusion_segmentation.
        sampler: Sampling algorithm for diffusion_segmentation.

    Returns:
        Loaded config.
    """
    config = OmegaConf.load(log_dir / "files" / "config_backup.yaml")
    if config.task.name == "diffusion_segmentation":
        if num_timesteps <= 0:
            raise ValueError("num_timesteps required for diffusion.")
        config.task.sampler.num_inference_timesteps = num_timesteps
        logging.info(f"Sampling {num_timesteps} steps.")
        if not sampler:
            raise ValueError("sampler required for diffusion.")
        config.task.sampler.name = sampler
        logging.info(f"Using sampler {sampler}.")
    return config


def main() -> None:
    """Main function."""
    args = parse_args()
    logging.info(f"Local devices are: {jax.local_devices()}")

    # load config
    config = load_and_parse_config(
        log_dir=args.log_dir, num_timesteps=args.num_timesteps, sampler=args.sampler
    )

    # find all available checkpoints
    steps = get_checkpoint_steps(log_dir=args.log_dir)

    # evaluate
    ckpt_dir = args.log_dir / "files" / "ckpt"
    run = build_experiment(config=config)
    for step in steps:
        logging.info(f"Starting valid split evaluation for step {step}.")

        # load checkpoint
        train_state, _ = run.train_init(ckpt_dir=ckpt_dir, step=step)

        # evaluation
        key = jax.random.PRNGKey(config.seed)
        key = common_utils.shard_prng_key(key)  # each replica has a different key
        _, val_metrics = run.eval_step(train_state=train_state, key=key, split=VALID_SPLIT)

        # save metrics
        out_dir = ckpt_dir / f"checkpoint_{step}"
        if config.task.name == "diffusion_segmentation":
            out_dir = out_dir / config.task.sampler.name
            out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "mean_metrics.json", "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
