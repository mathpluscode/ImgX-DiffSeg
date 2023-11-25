"""Test image data iterators, requires building datasets first."""
from __future__ import annotations

import chex
import jax
import numpy as np
import pytest
import SimpleITK as sitk  # noqa: N813
from absl.testing import parameterized
from chex._src import fake
from omegaconf import DictConfig

from imgx.data.iterator import DatasetIterator, add_foreground_range, get_image_tfds_dataset
from imgx_datasets import AMOS_CT, BRATS2021_MR, INFO_MAP, MALE_PELVIC_MR, MUSCLE_US
from imgx_datasets.constant import FOREGROUND_RANGE, IMAGE, LABEL, UID


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


@pytest.mark.slow()
class TestImageIterator(chex.TestCase):
    """Test image iterators."""

    @chex.variants(without_jit=True, with_device=True, without_device=True)
    @parameterized.named_parameters(
        ("AMOS CT", AMOS_CT),
        ("Male Pelvic MR", MALE_PELVIC_MR),
        ("Muscle US", MUSCLE_US),
        ("BraTS2021 MR", BRATS2021_MR),
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
        max_num_samples_per_split = 3
        dataset_info = INFO_MAP[dataset_name]
        image_shape = dataset_info.input_image_shape
        label_shape = dataset_info.label_shape

        config = DictConfig(
            {
                "seed": 0,
                "half_precision": False,
                "data": {
                    "loader": {
                        "max_num_samples_per_split": max_num_samples_per_split,
                        "data_augmentation": {
                            "max_rotation": [0.088, 0.088, 0.088],
                            "max_translation": [20, 20, 4],
                            "max_scaling": [0.15, 0.15, 0.15],
                        },
                    },
                    "trainer": {
                        "num_devices_per_replica": 1,
                        "batch_size": batch_size,
                        "batch_size_per_replica": batch_size_per_replica,
                    },
                },
            }
        )

        def get_batch() -> tuple[DatasetIterator, chex.ArrayTree, chex.ArrayTree, chex.ArrayTree]:
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

        dataset, train_batch, valid_batch, test_batch = self.variant(get_batch)()
        assert dataset.num_valid_steps == int(np.ceil(max_num_samples_per_split / batch_size))
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
                    *label_shape,
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


@pytest.mark.slow()
class TestImageShape(chex.TestCase):
    """Test the data loader shapes."""

    @parameterized.named_parameters(
        ("AMOS CT", AMOS_CT),
        ("Male Pelvic MR", MALE_PELVIC_MR),
        ("Muscle US", MUSCLE_US),
        ("BraTS2021 MR", BRATS2021_MR),
    )
    def test_shape(
        self,
        dataset_name: str,
    ) -> None:
        """Test the data loader shapes.

        Args:
            dataset_name: dataset name.
        """
        dataset_info = INFO_MAP[dataset_name]
        image_shape = dataset_info.input_image_shape
        label_shape = dataset_info.label_shape
        num_devices_per_replica = jax.local_device_count()
        batch_size = 2
        batch_size_per_replica = 1
        max_num_samples_per_split = 3
        assert batch_size % num_devices_per_replica == 0
        config = DictConfig(
            {
                "seed": 0,
                "half_precision": False,
                "data": {
                    "loader": {
                        "max_num_samples_per_split": max_num_samples_per_split,
                        "data_augmentation": {
                            "max_rotation": [0.088, 0.088, 0.088],
                            "max_translation": [20, 20, 4],
                            "max_scaling": [0.15, 0.15, 0.15],
                        },
                    },
                    "trainer": {
                        "num_devices_per_replica": 1,
                        "batch_size": batch_size,
                        "batch_size_per_replica": batch_size_per_replica,
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
                    *label_shape,
                ),
            )

    # in AMOS not all images have all labels, even without resampling
    # in BraTS2021 not all images have all labels, even without resampling
    @parameterized.named_parameters(
        ("Male Pelvic MR", MALE_PELVIC_MR),
        ("Muscle US", MUSCLE_US),
    )
    def test_labels(
        self,
        dataset_name: str,
    ) -> None:
        """Test all mask labels have all classes.

        Args:
            dataset_name: dataset name.
        """
        dataset_info = INFO_MAP[dataset_name]
        dir_processed = dataset_info.tfds_preprocessed_dir
        num_classes = dataset_info.num_classes
        assert dir_processed is not None
        mask_paths = list(dir_processed.glob("*_mask_preprocessed.nii.gz"))
        err_paths = []
        for path in mask_paths:
            volume = sitk.ReadImage(path)
            arr = sitk.GetArrayFromImage(volume)
            if np.unique(arr).size != num_classes:
                err_paths.append(path.name)
        if len(err_paths) > 0:
            raise ValueError(
                f"{err_paths} have less than {num_classes} classes including background."
            )


@pytest.mark.parametrize(
    ("batch", "expected"),
    [
        (
            {LABEL: np.array([0, 1, 2, 3])},
            np.array([[1, 3]]),
        ),
        (
            {LABEL: np.array([1, 2, 3, 0])},
            np.array([[0, 2]]),
        ),
        (
            {LABEL: np.array([1, 2, 3, 4])},
            np.array([[0, 3]]),
        ),
        (
            {LABEL: np.array([0, 1, 2, 3, 4, 0, 0])},
            np.array([[1, 4]]),
        ),
        (
            {LABEL: np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 0, 0, 0]])},
            np.array([[0, 1], [1, 3]]),
        ),
        (
            {
                LABEL + "_1": np.array([0, 1, 2, 3]),
                LABEL + "_2": np.array([1, 2, 3, 0]),
            },
            np.array([[0, 3]]),
        ),
        (
            {
                LABEL + "_1": np.array([0, 1, 2, 3]),
                LABEL + "_2": np.array([0, 2, 3, 0]),
            },
            np.array([[1, 3]]),
        ),
        (
            {},
            None,
        ),
    ],
    ids=[
        "1d-left",
        "1d-right",
        "1d-none",
        "1d-both",
        "2d",
        "1d two labels all foreground",
        "1d two labels some foreground",
        "no foreground",
    ],
)
def test_get_foreground_range(
    batch: dict[str, np.ndarray],
    expected: np.ndarray | None,
) -> None:
    """Test get_translation_range return values.

    Args:
        batch: batch may have labels.
        expected: expected range, if None means there is no labels in batch.
    """
    got = add_foreground_range(batch)
    if expected is None:
        assert FOREGROUND_RANGE not in got
    else:
        chex.assert_trees_all_equal(got[FOREGROUND_RANGE], expected)
