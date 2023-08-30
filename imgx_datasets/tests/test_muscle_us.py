"""Test functions dedicated to muscle ultrasound data."""

from pathlib import Path

import chex
import numpy as np
import pytest

from imgx_datasets.image_io import load_2d_grayscale_image
from imgx_datasets.muscle_us.muscle_us_dataset_builder import (
    select_connected_component,
)


@pytest.mark.parametrize(
    ("mask", "threshold", "expected"),
    [
        (
            # "no component",
            np.zeros((8, 6), dtype=np.uint8),
            0.75,
            np.zeros((8, 6), dtype=np.uint8),
        ),
        (
            # "1 component",
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            0.75,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            # "2 components - 0.75",
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 1, 1],
                    [0, 1, 1, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            0.75,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            # "2 components - 0.6",
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 1, 1],
                    [0, 1, 1, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            0.6,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            # "2 components - 0.5",
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 1, 1],
                    [0, 1, 1, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            0.5,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
    ],
)
def test_select_connected_component_dummy_examples(
    mask: np.ndarray,
    threshold: float,
    expected: np.ndarray,
) -> None:
    """Test post-processed results.

    Args:
        mask: binary mask.
        threshold: threshold for connected component.
        expected: expected processed mask.
    """
    got = select_connected_component(mask, threshold)
    chex.assert_trees_all_equal(got, expected)


@pytest.mark.parametrize(
    ("input_name", "output_name", "threshold"),
    [
        (
            "BB_anon_425_2_mask_pred.png",
            "BB_anon_425_2_mask_pred_postprocessed.png",
            0.75,
        ),
        (
            "BB_anon_425_2_mask_pred.png",
            "BB_anon_425_2_mask_pred_postprocessed.png",
            0.5,
        ),
        (
            "GM_anon_780_3_mask_pred.png",
            "GM_anon_780_3_mask_pred_postprocessed.png",
            0.75,
        ),
        (
            "BB_anon_1789_3_mask_pred.png",
            "BB_anon_1789_3_mask_pred_postprocessed_0.5.png",
            0.5,
        ),
        (
            "BB_anon_1789_3_mask_pred.png",
            "BB_anon_1789_3_mask_pred_postprocessed_0.75.png",
            0.75,
        ),
    ],
)
def test_select_connected_component_real_examples(
    fixture_path: Path,
    input_name: str,
    output_name: str,
    threshold: float,
) -> None:
    """Test post-processed results.

    Args:
        fixture_path: path to fixture directory.
        input_name: name of input file.
        output_name: name of output file.
        threshold: threshold for connected component.
    """
    mask = load_2d_grayscale_image(fixture_path / input_name)
    expected = load_2d_grayscale_image(fixture_path / output_name)
    got = select_connected_component(mask, threshold)
    chex.assert_trees_all_equal(got, expected)
