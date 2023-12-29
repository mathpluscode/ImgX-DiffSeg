"""Tests for image preprocessing."""
from __future__ import annotations

import chex
import numpy as np
import SimpleITK as sitk  # noqa: N813
from absl.testing import parameterized
from chex._src import fake

from imgx.datasets.preprocess import (
    clip_and_normalise_intensity_3d,
    clip_and_normalise_intensity_4d,
    crop_4d,
    get_binary_mask_bounding_box,
    pad_4d,
    resample_3d,
    resample_4d,
    resample_clip_pad_3d,
    resample_clip_pad_4d,
)


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestBBox(chex.TestCase):
    """Test get_binary_mask_bounding_box."""

    @parameterized.named_parameters(
        (
            "1d-int",
            np.array([0, 1, 0, 1, 0]),
            np.array([1]),
            np.array([4]),
        ),
        (
            "1d-bool",
            np.array([False, True, False, True, False]),
            np.array([1]),
            np.array([4]),
        ),
        (
            "1d-all-true",
            np.array([True, True, True, True, True]),
            np.array([0]),
            np.array([5]),
        ),
        (
            "1d-all-false",
            np.array([False, False, False, False, False]),
            np.array([-1]),
            np.array([-1]),
        ),
        (
            "2d-1x5",
            np.array([[0, 1, 0, 1, 0]]),
            np.array([0, 1]),
            np.array([1, 4]),
        ),
        (
            "2d-2x5",
            np.array([[0, 1, 0, 1, 0], [1, 1, 0, 1, 0]]),
            np.array([0, 0]),
            np.array([2, 4]),
        ),
        (
            "2d-2x5-all-false",
            np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
            np.array([-1, -1]),
            np.array([-1, -1]),
        ),
    )
    def test_values(
        self,
        mask: np.ndarray,
        expected_bbox_min: np.ndarray,
        expected_bbox_max: np.ndarray,
    ) -> None:
        """Test dice loss values.

        Args:
            mask: binary mask with only spatial axes.
            expected_bbox_min: expected bounding box min, inclusive.
            expected_bbox_max: expected bounding box max, exclusive.
        """
        got_bbox_min, got_bbox_max = get_binary_mask_bounding_box(
            mask=mask,
        )
        chex.assert_trees_all_close(got_bbox_min, expected_bbox_min)
        chex.assert_trees_all_close(got_bbox_max, expected_bbox_max)


class TestCrop(chex.TestCase):
    """Test crop_4d."""

    @parameterized.product(
        (
            {
                "image_shape": (8, 10, 11, 12),
                "crop_lower": (0, 0, 0),
                "crop_upper": (0, 0, 0),
                "axis": 0,
                "expected_shape": (8, 10, 11, 12),
            },
            {
                "image_shape": (8, 10, 11, 12),
                "crop_lower": (1, 2, 0),
                "crop_upper": (0, 1, 2),
                "axis": 0,
                "expected_shape": (8, 9, 8, 10),
            },
            {
                "image_shape": (8, 10, 11, 12),
                "crop_lower": (1, 2, 0),
                "crop_upper": (0, 1, 2),
                "axis": 1,
                "expected_shape": (7, 10, 8, 10),
            },
        ),
    )
    def test_4d_shapes(
        self,
        image_shape: tuple[int, ...],
        crop_lower: tuple[int, ...],
        crop_upper: tuple[int, ...],
        axis: int,
        expected_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes."""
        x = np.ones(image_shape, dtype=np.float32)
        x = np.transpose(x, axes=[3, 2, 1, 0])
        volume = sitk.GetImageFromArray(x, isVector=False)  # axes are reversed
        out = crop_4d(volume, crop_lower, crop_upper, axis)
        out_shape = out.GetSize()
        assert out_shape == expected_shape


class TestPad(chex.TestCase):
    """Test pad_4d."""

    @parameterized.product(
        (
            {
                "image_shape": (8, 10, 11, 12),
                "pad_lower": (0, 0, 0),
                "pad_upper": (0, 0, 0),
                "axis": 0,
                "expected_shape": (8, 10, 11, 12),
            },
            {
                "image_shape": (8, 10, 11, 12),
                "pad_lower": (1, 2, 0),
                "pad_upper": (0, 1, 2),
                "axis": 0,
                "expected_shape": (8, 11, 14, 14),
            },
            {
                "image_shape": (8, 10, 11, 12),
                "pad_lower": (1, 2, 0),
                "pad_upper": (0, 1, 2),
                "axis": 1,
                "expected_shape": (9, 10, 14, 14),
            },
        ),
    )
    def test_4d_shapes(
        self,
        image_shape: tuple[int, ...],
        pad_lower: tuple[int, ...],
        pad_upper: tuple[int, ...],
        axis: int,
        expected_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes."""
        x = np.ones(image_shape, dtype=np.float32)
        x = np.transpose(x, axes=[3, 2, 1, 0])
        volume = sitk.GetImageFromArray(x, isVector=False)  # axes are reversed
        out = pad_4d(volume, pad_lower, pad_upper, axis)
        out_shape = out.GetSize()
        assert out_shape == expected_shape


class TestResample(chex.TestCase):
    """Test resample_3d and resample_4d."""

    @parameterized.product(
        (
            {
                "image_shape": (10, 11, 12),
                "target_spacing": (1.0, 2.0, 3.0),
                "expected_shape": (10, 11, 12),
            },
            {
                "image_shape": (10, 11, 12),
                "target_spacing": (1.0, 4.0, 3.0),
                "expected_shape": (10, 6, 12),
            },
            {
                "image_shape": (10, 11, 12),
                "target_spacing": (1.0, 2.0, 1.5),
                "expected_shape": (10, 11, 24),
            },
        ),
        is_label=[True, False],
    )
    def test_3d_shapes(
        self,
        image_shape: tuple[int, ...],
        is_label: bool,
        target_spacing: tuple[float, ...],
        expected_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes."""
        source_spacing = (1.0, 2.0, 3.0)
        dtype = np.uint8 if is_label else np.float32
        x = np.ones(image_shape, dtype=dtype)
        x = np.transpose(x, axes=[2, 1, 0])
        volume = sitk.GetImageFromArray(x)  # axes are reversed
        volume.SetSpacing(source_spacing)
        out = resample_3d(volume, is_label, target_spacing)
        out_shape = out.GetSize()
        assert out_shape == expected_shape

    @parameterized.product(
        (
            {
                "image_shape": (8, 10, 11, 12),
                "source_spacing": (1.0, 1.0, 2.0, 3.0),
                "target_spacing": (1.5, 2.5, 3.5),
                "axis": 0,
                "expected_shape": (8, 7, 9, 10),
            },
            {
                "image_shape": (8, 10, 11, 12),
                "source_spacing": (1.0, 1.0, 2.0, 3.0),
                "target_spacing": (1.5, 2.5, 3.5),
                "axis": 1,
                "expected_shape": (5, 10, 9, 10),
            },
            {
                "image_shape": (8, 10, 11, 12),
                "source_spacing": (1.0, 1.0, 2.0, 3.0),
                "target_spacing": (1.5, 2.5, 3.5),
                "axis": 2,
                "expected_shape": (5, 4, 11, 10),
            },
            {
                "image_shape": (8, 10, 11, 12),
                "source_spacing": (1.0, 1.0, 2.0, 3.0),
                "target_spacing": (1.5, 2.5, 3.5),
                "axis": 3,
                "expected_shape": (5, 4, 6, 12),
            },
            {
                "image_shape": (8, 10, 11, 12),
                "source_spacing": (1.0, 1.0, 2.0, 3.0),
                "target_spacing": (1.5, 2.5, 3.5),
                "axis": -4,
                "expected_shape": (8, 7, 9, 10),
            },
            {
                "image_shape": (8, 10, 11, 12),
                "source_spacing": (1.0, 1.0, 2.0, 3.0),
                "target_spacing": (1.5, 2.5, 3.5),
                "axis": -3,
                "expected_shape": (5, 10, 9, 10),
            },
            {
                "image_shape": (8, 10, 11, 12),
                "source_spacing": (1.0, 1.0, 2.0, 3.0),
                "target_spacing": (1.5, 2.5, 3.5),
                "axis": -2,
                "expected_shape": (5, 4, 11, 10),
            },
            {
                "image_shape": (8, 10, 11, 12),
                "source_spacing": (1.0, 1.0, 2.0, 3.0),
                "target_spacing": (1.5, 2.5, 3.5),
                "axis": -1,
                "expected_shape": (5, 4, 6, 12),
            },
        ),
        is_label=[True, False],
    )
    def test_4d_shapes(
        self,
        image_shape: tuple[int, ...],
        is_label: bool,
        source_spacing: tuple[float, ...],
        target_spacing: tuple[float, ...],
        axis: int,
        expected_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes."""
        dtype = np.uint8 if is_label else np.float32
        x = np.ones(image_shape, dtype=dtype)
        x = np.transpose(x, axes=[3, 2, 1, 0])
        volume = sitk.GetImageFromArray(x, isVector=False)  # axes are reversed
        volume.SetSpacing(source_spacing)
        out = resample_4d(volume, is_label, target_spacing, axis)
        out_shape = out.GetSize()
        assert out_shape == expected_shape


class TestResampleClipPad(chex.TestCase):
    """Test resample_clip_pad_3d and resample_clip_pad_4d."""

    @parameterized.product(
        (
            {
                "image_shape": (10, 11, 12),
                "target_spacing": (1.0, 2.0, 3.0),
                "target_shape": (9, 8, 12),
                "expected_shape": (10, 11, 12),
            },
            {
                "image_shape": (10, 11, 12),
                "target_spacing": (1.0, 4.0, 3.0),
                "target_shape": (9, 8, 12),
                "expected_shape": (10, 8, 12),
            },
            {
                "image_shape": (10, 11, 12),
                "target_spacing": (1.0, 2.0, 1.5),
                "target_shape": (9, 8, 12),
                "expected_shape": (10, 11, 24),
            },
        ),
        is_label=[True, False],
    )
    def test_3d_shapes(
        self,
        image_shape: tuple[int, ...],
        is_label: bool,
        target_spacing: tuple[float, ...],
        target_shape: tuple[int, ...],
        expected_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes."""
        source_spacing = (1.0, 2.0, 3.0)
        dtype = np.uint8 if is_label else np.float32
        x = np.ones(image_shape, dtype=dtype)
        x = np.transpose(x, axes=[2, 1, 0])
        volume = sitk.GetImageFromArray(x)  # axes are reversed
        volume.SetSpacing(source_spacing)
        out = resample_clip_pad_3d(volume, target_spacing, target_shape, is_label)
        out_shape = out.GetSize()
        assert out_shape == expected_shape

    @parameterized.product(
        (
            {
                "image_shape": (8, 10, 11, 12),
                "source_spacing": (1.0, 1.0, 2.0, 3.0),
                "target_spacing": (1.5, 2.5, 3.5),
                "target_shape": (9, 8, 12),
                "axis": 0,
                "expected_shape": (8, 9, 9, 12),
            },
            {
                "image_shape": (8, 10, 11, 12),
                "source_spacing": (1.0, 1.0, 2.0, 3.0),
                "target_spacing": (1.5, 2.5, 3.5),
                "target_shape": (9, 8, 12),
                "axis": 1,
                "expected_shape": (9, 10, 9, 12),
            },
            {
                "image_shape": (8, 10, 11, 12),
                "source_spacing": (1.0, 1.0, 2.0, 3.0),
                "target_spacing": (1.5, 2.5, 3.5),
                "target_shape": (9, 8, 12),
                "axis": 2,
                "expected_shape": (9, 8, 11, 12),
            },
            {
                "image_shape": (8, 10, 11, 12),
                "source_spacing": (1.0, 1.0, 2.0, 3.0),
                "target_spacing": (1.5, 2.5, 3.5),
                "target_shape": (9, 8, 12),
                "axis": 3,
                "expected_shape": (9, 8, 12, 12),
            },
        ),
        is_label=[True, False],
    )
    def test_4d_shapes(
        self,
        image_shape: tuple[int, ...],
        is_label: bool,
        source_spacing: tuple[float, ...],
        target_spacing: tuple[float, ...],
        axis: int,
        target_shape: tuple[int, ...],
        expected_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes."""
        dtype = np.uint8 if is_label else np.float32
        x = np.ones(image_shape, dtype=dtype)
        x = np.transpose(x, axes=[3, 2, 1, 0])
        volume = sitk.GetImageFromArray(x, isVector=False)  # axes are reversed
        volume.SetSpacing(source_spacing)
        out = resample_clip_pad_4d(volume, target_spacing, target_shape, is_label, axis)
        out_shape = out.GetSize()
        assert out_shape == expected_shape


class TestClipAndNormaliseIntensity(chex.TestCase):
    """Test clip_and_normalise_intensity_3d, clip_and_normalise_intensity_4d."""

    @parameterized.product(
        (
            {
                "image_shape": (10, 11, 12),
                "intensity_range": None,
            },
            {
                "image_shape": (10, 11, 12),
                "intensity_range": (0.2, 0.8),
            },
        ),
    )
    def test_3d_shapes(
        self,
        image_shape: tuple[int, ...],
        intensity_range: tuple[float, float] | None,
    ) -> None:
        """Test output shapes."""
        x = np.random.rand(*image_shape)
        x = np.transpose(x, axes=[2, 1, 0])
        volume = sitk.GetImageFromArray(x)  # axes are reversed
        out = clip_and_normalise_intensity_3d(volume, intensity_range)
        out_shape = out.GetSize()
        assert out_shape == image_shape

    @parameterized.product(
        (
            {
                "image_shape": (9, 10, 11, 12),
                "intensity_range": None,
            },
            {
                "image_shape": (9, 10, 11, 12),
                "intensity_range": (0.2, 0.8),
            },
        ),
        axis=[-4, -3, -2, -1, 0, 1, 2, 3],
    )
    def test_4d_shapes(
        self,
        image_shape: tuple[int, ...],
        intensity_range: tuple[float, float] | None,
        axis: int,
    ) -> None:
        """Test output shapes."""
        x = np.random.rand(*image_shape)
        x = np.transpose(x, axes=[3, 2, 1, 0])
        volume = sitk.GetImageFromArray(x, isVector=False)  # axes are reversed
        out = clip_and_normalise_intensity_4d(volume, intensity_range, axis)
        out_shape = out.GetSize()
        assert out_shape == image_shape
