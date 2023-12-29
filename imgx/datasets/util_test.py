"""Tests for image utils of datasets."""
import pytest
from chex._src import fake

from imgx.datasets.util import (
    get_center_crop_shape,
    get_center_crop_shape_from_bbox,
    get_center_pad_shape,
    try_to_get_center_crop_shape,
)


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


@pytest.mark.parametrize(
    ("current_shape", "target_shape", "expected_lower", "expected_upper"),
    [
        (
            (64, 44, 40),
            (64, 44, 40),
            (0, 0, 0),
            (0, 0, 0),
        ),
        (
            (64, 44, 40),
            (40, 44, 30),
            (0, 0, 0),
            (0, 0, 0),
        ),
        (
            (64, 44, 40),
            (64, 64, 40),
            (0, 10, 0),
            (0, 10, 0),
        ),
        (
            (63, 43, 39),
            (64, 64, 40),
            (0, 10, 0),
            (1, 11, 1),
        ),
        (
            (44, 40),
            (64, 40),
            (10, 0),
            (10, 0),
        ),
        (
            (43, 39),
            (64, 40),
            (10, 0),
            (11, 1),
        ),
    ],
    ids=[
        "3d_same",
        "3d_no_pad",
        "3d_even",
        "3d_odd",
        "2d_even",
        "2d_odd",
    ],
)
def test_get_center_pad_shape(
    current_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    expected_lower: tuple[int, ...],
    expected_upper: tuple[int, ...],
) -> None:
    """Test get_center_pad_shape.

    Args:
        current_shape: current shape of the image.
        target_shape: target shape of the image.
        expected_lower: shape to pad on the lower side.
        expected_upper: shape to pad on the upper side.
    """
    got_lower, got_upper = get_center_pad_shape(current_shape, target_shape)
    assert got_lower == expected_lower
    assert got_upper == expected_upper


@pytest.mark.parametrize(
    ("current_shape", "target_shape", "expected_lower", "expected_upper"),
    [
        (
            (64, 44, 40),
            (64, 44, 40),
            (0, 0, 0),
            (0, 0, 0),
        ),
        (
            (64, 44, 40),
            (64, 64, 40),
            (0, 0, 0),
            (0, 0, 0),
        ),
        (
            (64, 44, 40),
            (40, 44, 30),
            (12, 0, 5),
            (12, 0, 5),
        ),
        (
            (65, 45, 41),
            (40, 44, 30),
            (12, 0, 5),
            (13, 1, 6),
        ),
        (
            (64, 40),
            (40, 30),
            (12, 5),
            (12, 5),
        ),
        (
            (65, 41),
            (40, 30),
            (12, 5),
            (13, 6),
        ),
    ],
    ids=[
        "3d_same",
        "3d_no_crop",
        "3d_even",
        "3d_odd",
        "2d_even",
        "2d_odd",
    ],
)
def test_get_center_crop_shape(
    current_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    expected_lower: tuple[int, ...],
    expected_upper: tuple[int, ...],
) -> None:
    """Test get_center_crop_shape.

    Args:
        current_shape: current shape of the image.
        target_shape: target shape of the image.
        expected_lower: shape to crop on the lower side.
        expected_upper: shape to crop on the upper side.
    """
    got_lower, got_upper = get_center_crop_shape(current_shape, target_shape)
    assert got_lower == expected_lower
    assert got_upper == expected_upper


class TestTryCenterCrop:
    """Test try_to_get_center_crop_shape values and errors."""

    @pytest.mark.parametrize(
        (
            "label_min",
            "label_max",
            "current_length",
            "target_length",
            "expected_lower",
            "expected_upper",
        ),
        [
            (0, 5, 6, 6, 0, 0),
            (0, 5, 6, 7, 0, 0),
            (0, 5, 6, 4, 0, 2),
            (1, 5, 6, 4, 1, 1),
            (2, 6, 6, 4, 2, 0),
            (0, 3, 7, 4, 0, 3),
            (5, 7, 7, 4, 3, 0),
        ],
        ids=[
            "no_crop_same_length",
            "no_crop_too_short",
            "center_crop_no_shift_no_left_crop",
            "center_crop_no_shift_both_sides_crop",
            "center_crop_no_shift_no_right_crop",
            "shift_right",
            "shift_left",
        ],
    )
    def test_try_to_get_center_crop_shape(
        self,
        label_min: int,
        label_max: int,
        current_length: int,
        target_length: int,
        expected_lower: int,
        expected_upper: int,
    ) -> None:
        """Test try_to_get_center_crop_shape.

        Args:
            label_min: label index minimum, inclusive.
            label_max: label index maximum, exclusive.
            current_length: current image length.
            target_length: target image length.
            expected_lower: shape to crop on the lower side.
            expected_upper: shape to crop on the upper side.
        """
        got_lower, got_upper = try_to_get_center_crop_shape(
            label_min=label_min,
            label_max=label_max,
            current_length=current_length,
            target_length=target_length,
        )
        assert got_lower == expected_lower
        assert got_upper == expected_upper

    @pytest.mark.parametrize(
        ("label_min", "label_max", "current_length", "target_length"),
        [
            (-1, 5, 6, 6),
            (0, 7, 6, 7),
        ],
        ids=[
            "min_negative",
            "max_too_large",
        ],
    )
    def test_get_upsample_padding_configs_error(
        self,
        label_min: int,
        label_max: int,
        current_length: int,
        target_length: int,
    ) -> None:
        """Test try_to_get_center_crop_shape raising errors..

        Args:
            label_min: label index minimum, inclusive.
            label_max: label index maximum, exclusive.
            current_length: current image length.
            target_length: target image length.
        """
        with pytest.raises(ValueError) as err:  # noqa: PT011
            try_to_get_center_crop_shape(
                label_min=label_min,
                label_max=label_max,
                current_length=current_length,
                target_length=target_length,
            )
        assert "Label index out of range." in str(err.value)


@pytest.mark.parametrize(
    (
        "bbox_min",
        "bbox_max",
        "current_shape",
        "target_shape",
        "expected_lower",
        "expected_upper",
    ),
    [
        (
            (0, 0, 0),
            (64, 44, 40),
            (64, 44, 40),
            (64, 44, 40),
            (0, 0, 0),
            (0, 0, 0),
        ),
        (
            (0, 0, 0),
            (64, 44, 40),
            (64, 44, 40),
            (64, 64, 40),
            (0, 0, 0),
            (0, 0, 0),
        ),
        (
            (0, 30, 0),
            (20, 44, 40),
            (64, 44, 40),
            (20, 30, 30),
            (0, 14, 5),
            (44, 0, 5),
        ),
        (
            (0, 30, 0),
            (20, 44, 40),
            (65, 45, 41),
            (20, 30, 30),
            (0, 15, 5),
            (45, 0, 6),
        ),
    ],
    ids=[
        "3d_same",
        "3d_no_crop",
        "3d_even",
        "3d_odd",
    ],
)
def test_get_center_crop_shape_from_bbox(
    bbox_min: tuple[int, ...],
    bbox_max: tuple[int, ...],
    current_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    expected_lower: tuple[int, ...],
    expected_upper: tuple[int, ...],
) -> None:
    """Test get_center_crop_shape_from_bbox.

    Args:
        bbox_min: [start_in_1st_spatial_dim, ...], inclusive, starts at zero.
        bbox_max: [end_in_1st_spatial_dim, ...], exclusive, starts at zero.
        current_shape: current shape of the image.
        target_shape: target shape of the image.
        expected_lower: shape to crop on the lower side.
        expected_upper: shape to crop on the upper side.
    """
    got_lower, got_upper = get_center_crop_shape_from_bbox(
        bbox_min, bbox_max, current_shape, target_shape
    )
    assert got_lower == expected_lower
    assert got_upper == expected_upper
