"""Script to launch evaluation on validation tests."""
import argparse
from pathlib import Path

import jax
from absl import logging
from omegaconf import OmegaConf

from imgx.device import broadcast_to_local_devices
from imgx.exp import Experiment
from imgx.exp.train_state import get_eval_params_and_state_from_ckpt
from imgx_datasets.constant import VALID_SPLIT

logging.set_verbosity(logging.INFO)


def get_checkpoint_dirs(
    log_dir: Path,
) -> list[Path]:
    """Get the directory of all checkpoints.

    Args:
        log_dir: Directory of entire log.

    Returns:
        A list of directories having arrays.npy and tree.pkl.

    Raises:
        ValueError: if any file not found.
    """
    ckpt_dir = log_dir / "files" / "ckpt"
    ckpt_dirs = []
    num_batches = []
    for ckpt_i_dir in ckpt_dir.glob("batch_*/"):
        if not ckpt_i_dir.is_dir():
            continue
        array_path = ckpt_i_dir / "arrays.npy"
        if not array_path.exists():
            continue
        tree_path = ckpt_i_dir / "tree.pkl"
        if not tree_path.exists():
            continue
        num_batches.append(int(ckpt_i_dir.stem.split("_")[-1]))
        ckpt_dirs.append(ckpt_i_dir)
    ckpt_dirs = [
        x[1] for x in sorted(zip(num_batches, ckpt_dirs), reverse=True)
    ]
    return ckpt_dirs


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


def main() -> None:
    """Main function."""
    args = parse_args()

    config = OmegaConf.load(args.log_dir / "files" / "config_backup.yaml")
    if config.task.name == "diffusion_segmentation":
        if args.num_timesteps <= 0:
            raise ValueError("num_timesteps required for diffusion.")
        config.task.sampler.num_inference_timesteps = args.num_timesteps
        logging.info(f"Sampling {args.num_timesteps} steps.")
        if not args.sampler:
            raise ValueError("sampler required for diffusion.")
        config.task.sampler.name = args.sampler
        logging.info(f"Using sampler {args.sampler}.")

    ckpt_dirs = get_checkpoint_dirs(
        log_dir=args.log_dir,
    )

    # init exp
    run = Experiment(config=config)
    run.eval_init()

    for ckpt_dir in ckpt_dirs:
        # output to different dirs for different sampler
        out_dir = ckpt_dir
        if config.task.name == "diffusion_segmentation":
            out_dir = ckpt_dir / args.sampler
        out_dir.mkdir(parents=True, exist_ok=True)

        # load checkpoint
        params, state = get_eval_params_and_state_from_ckpt(ckpt_dir=ckpt_dir)

        # inference
        logging.info(f"Starting valid split evaluation for {ckpt_dir}.")
        rng = jax.random.PRNGKey(config.seed)
        rng = broadcast_to_local_devices(rng)

        run.eval_step(
            split=VALID_SPLIT,
            params=params,
            state=state,
            rng=rng,
            out_dir=out_dir,
            save_predictions=False,
        )

        # clean up
        del params
        del state
        logging.info(f"Finished valid split evaluation for {ckpt_dir}.")


if __name__ == "__main__":
    main()
