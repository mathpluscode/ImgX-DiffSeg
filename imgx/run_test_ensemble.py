"""Script to launch ensemble on test set results."""
import argparse
import json
from collections import defaultdict
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import SimpleITK as sitk  # noqa: N813
from absl import logging
from omegaconf import OmegaConf
from tqdm import tqdm

from imgx.datasets import (
    DIR_TFDS_PROCESSED_MAP,
    IMAGE_SPACING_MAP,
    NUM_CLASSES_MAP,
)
from imgx.exp.eval import (
    get_jit_segmentation_metrics,
    get_non_jit_segmentation_metrics_per_step,
)

logging.set_verbosity(logging.INFO)


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log_dir",
        type=Path,
        help="Folder of wandb.",
        default=None,
    )
    args = parser.parse_args()

    return args


def vote_ensemble(test_dir: Path, dir_tfds: Path, num_classes: int) -> None:
    """Ensemble prediction via voting.

    Args:
        test_dir: path having predictions.
        dir_tfds: path of tfds data, having ground truth.
        num_classes: number of classes in labels.
    """
    # get seed dirs and sort by seeds
    lst_seed_dir = sorted(
        test_dir.glob("seed_*/"), key=lambda x: int(x.stem.split("_")[-1])
    )
    num_seeds = len(lst_seed_dir)

    # map relative_path to list of full path, corresponding to seeds
    path_dict = defaultdict(list)
    for seed_dir in lst_seed_dir:
        for x in seed_dir.glob("**/*.nii.gz"):
            rel_path = x.relative_to(seed_dir)
            path_dict[rel_path].append(x)

    # vote to ensemble
    logging.info("Calculating ensemble predictions.")
    for rel_path, pred_paths in path_dict.items():
        # a list of shape (D, W, H)
        mask_preds = [
            sitk.GetArrayFromImage(sitk.ReadImage(x)) for x in pred_paths
        ]
        # (D, W, H, num_classes, num_seeds)
        mask_onehot = jax.nn.one_hot(
            jnp.stack(mask_preds, axis=-1), num_classes=num_classes, axis=-2
        )
        # vote (D, W, H)
        mask_pred = jnp.argmax(jnp.sum(mask_onehot, axis=-1), axis=-1).astype(
            "uint16"
        )

        # copy meta data
        uid = pred_paths[0].stem.split("_")[0]
        volume_mask_true = sitk.ReadImage(
            dir_tfds / f"{uid}_mask_preprocessed.nii.gz"
        )
        volume_mask_pred = sitk.GetImageFromArray(mask_pred)
        volume_mask_pred.CopyInformation(volume_mask_true)

        # save
        out_path = test_dir / f"ensemble_{num_seeds}" / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(
            image=volume_mask_pred,
            fileName=out_path,
            useCompression=True,
        )


def evaluate_ensemble_prediction(
    dir_path: Path, dir_tfds: Path, num_classes: int, spacing: jnp.ndarray
) -> None:
    """Evaluate the saved predictions from ensemble.

    Args:
        dir_path: path having predictions.
        dir_tfds: path of tfds data, having ground truth.
        num_classes: number of classes in labels.
        spacing: spacing for voxels.
    """
    num_steps = int(dir_path.name.split("_")[1])
    uids = [
        x.name.split("_")[0] for x in (dir_path / "step_0").glob("*.nii.gz")
    ]
    lst_df_scalar = []
    for uid in tqdm(uids, total=len(uids)):
        # (D, W, H)
        mask_true = sitk.GetArrayFromImage(
            sitk.ReadImage(dir_tfds / f"{uid}_mask_preprocessed.nii.gz")
        )
        # (D, W, H, num_classes)
        mask_true = jax.nn.one_hot(mask_true, num_classes=num_classes, axis=-1)
        # (1, W, H, D, num_classes)
        mask_true = jnp.transpose(mask_true, axes=(2, 1, 0, 3))[None, ...]

        pred_paths = [
            dir_path / f"step_{i}" / f"{uid}_mask_pred.nii.gz"
            for i in range(num_steps)
        ]
        # a list of shape (D, W, H)
        mask_preds = [
            sitk.GetArrayFromImage(sitk.ReadImage(x)) for x in pred_paths
        ]
        # (D, W, H, num_classes, num_steps)
        mask_pred = jax.nn.one_hot(
            jnp.stack(mask_preds, axis=-1), num_classes=num_classes, axis=-2
        )
        # (1, W, H, D, num_classes, num_steps)
        mask_pred = jnp.transpose(mask_pred, axes=(2, 1, 0, 3, 4))[None, ...]

        # metrics
        scalars_jit = jax.vmap(
            partial(
                get_jit_segmentation_metrics,
                mask_true=mask_true,
                spacing=spacing,
            ),
            in_axes=-1,
            out_axes=-1,
        )(mask_pred)
        scalars_nonjit = get_non_jit_segmentation_metrics_per_step(
            mask_pred=mask_pred,
            mask_true=mask_true,
            spacing=spacing,
        )
        scalars = {**scalars_jit, **scalars_nonjit}

        # flatten per step
        scalars_flatten = {}
        for k, v in scalars.items():
            for i in range(v.shape[-1]):
                scalars_flatten[f"{k}_step_{i}"] = v[..., i]
            scalars_flatten[k] = v[..., -1]
        scalars = scalars_flatten
        scalars = jax.tree_map(lambda x: np.asarray(x).tolist(), scalars)
        scalars["uid"] = [uid]
        lst_df_scalar.append(pd.DataFrame(scalars))

    # assemble metrics
    df_scalar = pd.concat(lst_df_scalar)
    df_scalar = df_scalar.sort_values("uid")
    df_scalar.to_csv(dir_path / "metrics_per_sample.csv", index=False)

    # average over samples in the dataset
    scalars = df_scalar.drop("uid", axis=1).mean().to_dict()
    scalars = {"test_" + k: v for k, v in scalars.items()}
    scalars["num_images_in_total"] = len(df_scalar)
    with open(dir_path / "mean_metrics.json", "w", encoding="utf-8") as f:
        json.dump(scalars, f, sort_keys=True, indent=4)


def main() -> None:  # pylint:disable=R0915
    """Main function."""
    args = parse_args()

    config = OmegaConf.load(args.log_dir / "files" / "config_backup.yaml")
    if config.task.name != "diffusion":
        raise ValueError("Ensemble is only for diffusion.")

    data_config = config.data
    dir_tfds = DIR_TFDS_PROCESSED_MAP[data_config.name]
    spacing = jnp.array(IMAGE_SPACING_MAP[data_config.name])
    num_classes = NUM_CLASSES_MAP[data_config["name"]]

    test_dir = args.log_dir / "files" / "test_evaluation"

    # no ensemble if 1 seed only
    lst_seed_dir = sorted(
        test_dir.glob("seed_*/"), key=lambda x: int(x.stem.split("_")[-1])
    )
    if len(lst_seed_dir) == 1:
        logging.info("Ensemble not performed as there is one seed only.")
        return

    # ensemble
    vote_ensemble(test_dir=test_dir, dir_tfds=dir_tfds, num_classes=num_classes)
    # evaluate
    for dir_path in test_dir.glob("ensemble_*/sample_*_steps"):
        logging.info(f"Evaluating ensemble predictions metrics for {dir_path}.")
        evaluate_ensemble_prediction(
            dir_path=dir_path,
            dir_tfds=dir_tfds,
            num_classes=num_classes,
            spacing=spacing,
        )


if __name__ == "__main__":
    main()
