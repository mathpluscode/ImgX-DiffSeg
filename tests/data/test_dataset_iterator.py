"""Test image data iterators."""
from typing import Tuple

import chex
import haiku as hk
import jax
import numpy as np
import SimpleITK as sitk  # noqa: N813
from absl.testing import parameterized
from chex._src import fake
from omegaconf import DictConfig

from imgx import IMAGE, LABEL, UID
from imgx.datasets import (
    AMOS_CT,
    DIR_TFDS_PROCESSED_MAP,
    IMAGE_SHAPE_MAP,
    MALE_PELVIC_MR,
    NUM_CLASSES_MAP,
    Dataset,
)
from imgx.datasets.iterator import get_image_tfds_dataset


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestImageIterator(chex.TestCase):
    """Test image iterators."""

    @chex.variants(without_jit=True, with_device=True, without_device=True)
    @parameterized.named_parameters(
        ("AMOS CT", AMOS_CT),
        ("Male Pelvic MR", MALE_PELVIC_MR),
    )
    def test_output_shape_variants(
        self,
        dataset_name: str,
    ) -> None:
        """Test iterator output shape under different device variants.

        Dataset num_valid_steps is tested too.

        Args:
            dataset_name: dataset name.
        """
        num_devices_per_replica = jax.local_device_count()
        batch_size = 2
        batch_size_per_replica = 1
        max_num_samples = 3
        image_shape = IMAGE_SHAPE_MAP[dataset_name]
        config = DictConfig(
            {
                "seed": 0,
                "training": {
                    "num_devices_per_replica": 1,
                    "batch_size": batch_size,
                    "batch_size_per_replica": batch_size_per_replica,
                    "mixed_precision": {
                        "use": False,
                    },
                },
                "data": {
                    "max_num_samples": max_num_samples,
                    dataset_name: {
                        "data_augmentation": {
                            "max_rotation": [0.088, 0.088, 0.088],
                            "max_translation": [20, 20, 4],
                            "max_scaling": [0.15, 0.15, 0.15],
                        },
                    },
                },
            }
        )

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def get_batch() -> (
            Tuple[Dataset, chex.ArrayTree, chex.ArrayTree, chex.ArrayTree]
        ):
            """Get one batch for iterator.

            Returns:
                Training batch
                Validation batch
                Test batch.
            """
            ds = get_image_tfds_dataset(
                dataset_name=dataset_name,
                config=config,
            )
            return (
                ds,
                next(ds.train_iter),
                next(ds.valid_iter),
                next(ds.test_iter),
            )

        dataset, train_batch, valid_batch, test_batch = get_batch()
        assert dataset.num_valid_steps == int(
            np.ceil(max_num_samples / batch_size)
        )
        for i, batch in enumerate([train_batch, valid_batch, test_batch]):
            chex.assert_shape(
                batch[IMAGE],
                (
                    num_devices_per_replica,
                    batch_size // num_devices_per_replica,
                    *image_shape,
                ),
            )
            chex.assert_shape(
                batch[LABEL],
                (
                    num_devices_per_replica,
                    batch_size // num_devices_per_replica,
                    *image_shape,
                ),
            )
            if i == 0:
                assert UID not in batch
            else:
                chex.assert_shape(
                    batch[UID],
                    (
                        num_devices_per_replica,
                        batch_size // num_devices_per_replica,
                    ),
                )


class TestImageShape(chex.TestCase):
    """Test the data loader shapes."""

    @parameterized.named_parameters(
        ("AMOS CT", AMOS_CT),
        ("Male Pelvic MR", MALE_PELVIC_MR),
    )
    def test_shape(
        self,
        dataset_name: str,
    ) -> None:
        """Test the data loader shapes.

        Args:
            dataset_name: dataset name.
        """
        image_shape = IMAGE_SHAPE_MAP[dataset_name]
        num_devices_per_replica = jax.local_device_count()
        batch_size = 2
        batch_size_per_replica = 1
        assert batch_size % num_devices_per_replica == 0
        config = DictConfig(
            {
                "seed": 0,
                "training": {
                    "num_devices_per_replica": 1,
                    "batch_size": batch_size,
                    "batch_size_per_replica": batch_size_per_replica,
                    "mixed_precision": {
                        "use": False,
                    },
                },
                "data": {
                    "max_num_samples": 4,
                    "dataset_name": {
                        "data_augmentation": {
                            "max_rotation": [0.088, 0.088, 0.088],
                            "max_translation": [20, 20, 4],
                            "max_scaling": [0.15, 0.15, 0.15],
                        },
                    },
                },
            }
        )

        dataset = get_image_tfds_dataset(
            dataset_name,
            config,
        )

        batch_size_per_replica = batch_size // num_devices_per_replica

        for it in [dataset.train_iter, dataset.valid_iter, dataset.test_iter]:
            batch = next(it)
            chex.assert_shape(
                batch[IMAGE],
                (
                    num_devices_per_replica,
                    batch_size_per_replica,
                    *image_shape,
                ),
            )
            chex.assert_shape(
                batch[LABEL],
                (
                    num_devices_per_replica,
                    batch_size_per_replica,
                    *image_shape,
                ),
            )

    # in AMOS not all images have all labels, even without resampling
    @parameterized.named_parameters(
        ("Male Pelvic MR", MALE_PELVIC_MR),
    )
    def test_labels(
        self,
        dataset_name: str,
    ) -> None:
        """Test all mask labels have all classes.

        Args:
            dataset_name: dataset name.
        """
        mask_paths = list(
            DIR_TFDS_PROCESSED_MAP[dataset_name].glob(
                "*_mask_preprocessed.nii.gz"
            )
        )
        err_paths = []
        for path in mask_paths:
            volume = sitk.ReadImage(path)
            arr = sitk.GetArrayFromImage(volume)
            if np.unique(arr).size != NUM_CLASSES_MAP[dataset_name]:
                err_paths.append(path.name)
        if len(err_paths) > 0:
            raise ValueError(
                f"{err_paths} have less than {NUM_CLASSES_MAP[dataset_name]} "
                f"classes including background."
            )
