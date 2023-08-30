"""Script to launch evaluation on test sets."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np
from absl import logging
from omegaconf import OmegaConf

from imgx.device import broadcast_to_local_devices
from imgx.exp import Experiment
from imgx.exp.train_state import get_eval_params_and_state_from_ckpt
from imgx_datasets.constant import TEST_SPLIT

logging.set_verbosity(logging.INFO)


def get_checkpoint_dir(
    log_dir: Path,
    num_batch: int,
    metric: str,
    max_metric: bool,
    sampler: str,
) -> Path:
    """Get the checkpoint directory.

    Args:
        log_dir: Directory of entire log.
        num_batch: number of batches to select checkpoint.
            -1 means select the latest one.
        metric: metric to maximise or minimise.
        max_metric: maximise the metric or not.
        sampler: sampler for diffusion.

    Returns:
        A directory having arrays.npy and tree.pkl.

    Raises:
        ValueError: if any file not found.
    """
    ckpt_dir = log_dir / "files" / "ckpt"
    if num_batch < 0:
        # take the one having the best metrics
        best_metric_scalar = -np.inf if max_metric else np.inf
        for ckpt_i_dir in ckpt_dir.glob("batch_*/"):
            if not ckpt_i_dir.is_dir():
                continue
            # load metric
            num_batch_i = int(ckpt_i_dir.stem.split("_")[-1])
            if not sampler:
                metric_path = ckpt_i_dir / "mean_metrics.json"
            else:
                metric_path = ckpt_i_dir / sampler / "mean_metrics.json"
            if not metric_path.exists():
                logging.warning(f"Metrics not found in {ckpt_i_dir}")
                continue

            with open(metric_path, encoding="utf-8") as f:
                scalars = json.load(f)
            if metric not in scalars:
                raise ValueError(f"Metrics {metric} not found in {ckpt_i_dir}")
            metric_scalar = scalars[metric]

            # use the ckpt if it's the first or its metric is better
            # if same performance, prefer being trained for longer
            if (
                (num_batch < 0)
                or (max_metric and (best_metric_scalar <= metric_scalar))
                or ((not max_metric) and (best_metric_scalar >= metric_scalar))
            ):
                best_metric_scalar = metric_scalar
                num_batch = num_batch_i
    if num_batch < 0:
        raise ValueError(f"Checkpoint not found under {ckpt_dir}")
    ckpt_dir = ckpt_dir / f"batch_{num_batch}"

    # sanity check
    if not ckpt_dir.exists():
        raise ValueError(f"Checkpoint directory {ckpt_dir} does not exist.")
    array_path = ckpt_dir / "arrays.npy"
    if not array_path.exists():
        raise ValueError(f"Checkpoint {array_path} does not exist.")
    tree_path = ckpt_dir / "tree.pkl"
    if not tree_path.exists():
        raise ValueError(f"Checkpoint {tree_path} does not exist.")

    return ckpt_dir


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
        "--num_batch",
        type=int,
        help="Number of batches for identify checkpoint.",
        default=-1,
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        help="Number of sampling steps for diffusion.",
        default=-1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed number, by default 0.",
        default=0,
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Metric to select template.",
        default="mean_binary_dice_score_without_background",
    )
    parser.add_argument("--max_metric", dest="max_metric", action="store_true")
    parser.add_argument("--min_metric", dest="max_metric", action="store_false")
    parser.set_defaults(max_metric=True)
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
    out_dir = args.log_dir / "files" / "test_evaluation"

    if config.task.name == "diffusion_segmentation":
        if args.num_timesteps <= 0:
            raise ValueError("num_timesteps required for diffusion.")
        config.task.sampler.num_inference_timesteps = args.num_timesteps
        logging.info(f"Sampling {args.num_timesteps} steps.")
        if not args.sampler:
            raise ValueError("sampler required for diffusion.")
        config.task.sampler.name = args.sampler
        logging.info(f"Using sampler {args.sampler}.")

    ckpt_dir = get_checkpoint_dir(
        log_dir=args.log_dir,
        num_batch=args.num_batch,
        metric=args.metric,
        max_metric=args.max_metric,
        sampler=args.sampler
        if config.task.name == "diffusion_segmentation"
        else "",
    )
    logging.info(f"Using checkpoint {ckpt_dir}.")

    # load checkpoint
    params, state = get_eval_params_and_state_from_ckpt(
        ckpt_dir=ckpt_dir,
    )

    # inference per seed
    seed = args.seed
    logging.info(f"Starting test split evaluation for seed {seed}.")

    out_dir_seed = out_dir / f"seed_{seed}"
    out_dir_seed.mkdir(parents=True, exist_ok=True)
    if config.task.name == "diffusion_segmentation":
        out_dir_seed = (
            out_dir_seed / f"sample_{args.num_timesteps}_steps" / args.sampler
        )

    # init exp
    # necessary as data set will be exhausted
    run = Experiment(config=config)
    run.eval_init()

    rng = jax.random.PRNGKey(seed)
    rng = broadcast_to_local_devices(rng)
    run.eval_step(
        split=TEST_SPLIT,
        params=params,
        state=state,
        rng=rng,
        out_dir=out_dir_seed,
        save_predictions=True,
    )
    logging.info(f"Finished test split evaluation for seed {seed}.")


if __name__ == "__main__":
    main()
