"""Script to launch evaluation on validation tests."""
import argparse
from pathlib import Path
from typing import List

import jax
from absl import logging
from omegaconf import OmegaConf

from imgx import VALID_SPLIT
from imgx.device import broadcast_to_local_devices
from imgx.exp import Experiment
from imgx.exp.train_state import get_eval_params_and_state_from_ckpt

logging.set_verbosity(logging.INFO)


def get_checkpoint_dirs(
    log_dir: Path,
) -> List[Path]:
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
        help="Number of sampling steps for diffusion.",
        default=-1,
    )
    args = parser.parse_args()

    return args


def main() -> None:
    """Main function."""
    args = parse_args()

    config = OmegaConf.load(args.log_dir / "files" / "config_backup.yaml")
    if config.task.name == "diffusion":
        if args.num_timesteps <= 0:
            raise ValueError("num_timesteps required for diffusion.")
        config.task.diffusion.num_timesteps = args.num_timesteps

    ckpt_dirs = get_checkpoint_dirs(
        log_dir=args.log_dir,
    )

    # init exp
    run = Experiment(config=config)
    run.eval_init()

    for ckpt_dir in ckpt_dirs:
        # load checkpoint
        params, state = get_eval_params_and_state_from_ckpt(
            ckpt_dir=ckpt_dir,
            use_ema=config.training.ema.use,
        )
        # prevent any gradient related actions
        params = jax.lax.stop_gradient(params)
        state = jax.lax.stop_gradient(state)

        # inference
        logging.info(f"Starting valid split evaluation for {ckpt_dir}.")
        rng = jax.random.PRNGKey(config.seed)
        rng = broadcast_to_local_devices(rng)
        run.eval_step(
            split=VALID_SPLIT,
            params=params,
            state=state,
            rng=rng,
            out_dir=ckpt_dir,
            save_predictions=False,
        )

        # clean up
        del params
        del state


if __name__ == "__main__":
    main()
