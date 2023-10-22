"""Script to launch evaluation on test sets."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np
from absl import logging
from flax.training import common_utils

from imgx.run_train import build_experiment
from imgx.run_valid import get_checkpoint_steps, load_and_parse_config
from imgx_datasets.constant import TEST_SPLIT

logging.set_verbosity(logging.INFO)


def get_best_checkpoint_step(
    log_dir: Path,
    metric: str,
    max_metric: bool,
    sampler: str,
) -> int:
    """Get the checkpoint directory.

    Args:
        log_dir: Directory of entire log.
        step: step for checkpoint, -1 means select the best one.
        metric: metric to maximise or minimise.
        max_metric: maximise the metric or not.
        sampler: sampler for diffusion.

    Returns:
        The best step.

    Raises:
        ValueError: if file not found.
    """
    ckpt_dir = log_dir / "files" / "ckpt"
    steps = get_checkpoint_steps(log_dir)
    best_metric_scalar = -np.inf if max_metric else np.inf
    best_step = -1
    for step in steps:
        step_dir = ckpt_dir / f"checkpoint_{step}"
        if not sampler:
            metric_path = step_dir / "mean_metrics.json"
        else:
            metric_path = step_dir / sampler / "mean_metrics.json"
        if not metric_path.exists():
            logging.warning(f"Metrics not found in {step_dir}")
            continue

        with open(metric_path, encoding="utf-8") as f:
            scalars = json.load(f)
        if metric not in scalars:
            raise ValueError(f"Metrics {metric} not found in {step_dir}")
        metric_scalar = scalars[metric]

        # use the ckpt if it's the first or its metric is better
        # if same performance, prefer being trained for longer
        if (
            (best_step < 0)
            or (max_metric and (best_metric_scalar <= metric_scalar))
            or ((not max_metric) and (best_metric_scalar >= metric_scalar))
        ):
            best_metric_scalar = metric_scalar
            best_step = step
    if best_step < 0:
        raise ValueError(f"Checkpoint not found under {ckpt_dir}")
    return best_step


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
        "--step",
        type=int,
        help="Step for identify checkpoint.",
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

    # load config
    config = load_and_parse_config(
        log_dir=args.log_dir, num_timesteps=args.num_timesteps, sampler=args.sampler
    )

    # prepare output directory
    out_dir = args.log_dir / "files" / "test_evaluation" / f"seed_{args.seed}"
    if config.task.name == "diffusion_segmentation":
        out_dir = out_dir / f"sample_{args.num_timesteps}_steps" / args.sampler
    out_dir.mkdir(parents=True, exist_ok=True)

    # find checkpoint
    step = args.step
    if step < 0:
        step = get_best_checkpoint_step(
            log_dir=args.log_dir,
            metric=args.metric,
            max_metric=args.max_metric,
            sampler=args.sampler if config.task.name == "diffusion_segmentation" else "",
        )
    logging.info(f"Using checkpoint at step {step}.")

    # load checkpoint
    logging.info(f"Starting test split evaluation for seed {args.seed}.")
    ckpt_dir = args.log_dir / "files" / "ckpt"
    run = build_experiment(config=config)
    train_state, _ = run.train_init(ckpt_dir=ckpt_dir, step=step)

    key = jax.random.PRNGKey(args.seed)
    key = common_utils.shard_prng_key(key)  # each replica has a different key

    _, test_metrics = run.eval_step(
        train_state=train_state,
        key=key,
        split=TEST_SPLIT,
        out_dir=out_dir,
    )
    with open(out_dir / "mean_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
