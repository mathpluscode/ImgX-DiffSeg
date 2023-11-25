"""Test experiments train, valid, and test.

mocker.patch, https://pytest-mock.readthedocs.io/en/latest/
"""
import shutil
from tempfile import TemporaryDirectory

import pytest
from pytest_mock import MockFixture

from imgx.run_test import main as run_test
from imgx.run_train import main as run_train
from imgx.run_valid import main as run_valid


@pytest.mark.integration()
@pytest.mark.parametrize(
    "dataset",
    ["muscle_us", "amos_ct"],
)
def test_segmentation_train_valid_test(mocker: MockFixture, dataset: str) -> None:
    """Test train, valid, and test.

    Args:
        mocker: mocker, a wrapper of unittest.mock.
        dataset: dataset name.
    """
    with TemporaryDirectory() as temp_dir:
        mocker.resetall()
        mocker.patch.dict("os.environ", {"WANDB_MODE": "offline"})
        mocker.patch(
            "sys.argv",
            ["pytest", "debug=true", "task=seg", f"data={dataset}", f"logging.root_dir={temp_dir}"],
        )
        run_train()  # pylint: disable=no-value-for-parameter
        mocker.patch(
            "sys.argv",
            ["pytest", f"--log_dir={temp_dir}/wandb/latest-run"],
        )
        run_valid()
        run_test()
        shutil.rmtree(temp_dir)


@pytest.mark.integration()
@pytest.mark.parametrize(
    "dataset",
    ["muscle_us", "amos_ct"],
)
def test_diffusion_segmentation_train_valid_test(mocker: MockFixture, dataset: str) -> None:
    """Test train, valid, and test.

    Args:
        mocker: mocker, a wrapper of unittest.mock.
        dataset: dataset name.
    """
    with TemporaryDirectory() as temp_dir:
        mocker.resetall()
        mocker.patch.dict("os.environ", {"WANDB_MODE": "offline"})
        mocker.patch(
            "sys.argv",
            [
                "pytest",
                "debug=true",
                "task=gaussian_diff_seg",
                f"data={dataset}",
                f"logging.root_dir={temp_dir}",
            ],
        )
        run_train()  # pylint: disable=no-value-for-parameter
        mocker.patch(
            "sys.argv",
            [
                "pytest",
                "--num_timesteps=2",
                "--sampler=DDPM",
                f"--log_dir={temp_dir}/wandb/latest-run",
            ],
        )
        run_valid()
        run_test()
        shutil.rmtree(temp_dir)
