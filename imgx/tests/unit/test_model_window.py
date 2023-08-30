"""Test Swin Transformer related classes and functions."""
from __future__ import annotations

from functools import partial

import chex
import jax
import numpy as np
import pytest
from absl.testing import parameterized
from chex._src import fake

from imgx.model.window import (
    get_window_mask,
    get_window_mask_index,
    get_window_shift_pad_shapes,
    window_partition,
    window_unpartition,
)


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestWindowPartition(chex.TestCase):
    """Test window related functions."""

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "multiple 2x2x2 windows",
            (4, 6, 8),
            (2, 2, 2),
            24,
        ),
        (
            "multiple 1x1x1 windows",
            (4, 6, 8),
            (1, 1, 1),
            192,
        ),
        (
            "one window",
            (4, 4, 4),
            (4, 4, 4),
            1,
        ),
        (
            "non-uniform 3d windows",
            (4, 6, 8),
            (2, 3, 4),
            8,
        ),
        (
            "non-uniform 2d windows",
            (4, 8),
            (2, 4),
            4,
        ),
        (
            "non-uniform 4d windows",
            (4, 9, 8, 5),
            (2, 3, 4, 5),
            12,
        ),
        (
            "non-even 4d windows",
            (4, 8, 7, 5),
            (2, 3, 4, 5),
            12,
        ),
    )
    def test_window_partition_shape(
        self,
        spatial_shape: tuple[int, ...],
        window_shape: tuple[int, ...],
        num_windows: int,
    ) -> None:
        """Test window_partition and window_unpartition output shapes.

        window_unpartition is also tested by reversing the output.

        Args:
            spatial_shape: (height, width, depth) for 3d.
            window_shape: (wh, ww, wd) for 3d.
            num_windows: number of windows.
        """
        batch = 2
        num_classes = 3
        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, shape=(batch, *spatial_shape, num_classes))
        got = self.variant(
            partial(window_partition, window_shape=window_shape)
        )(x=x)

        chex.assert_shape(
            got,
            (
                batch,
                num_windows,
                np.prod(window_shape),
                num_classes,
            ),
        )

        reversed_x = self.variant(
            partial(
                window_unpartition,
                window_shape=window_shape,
                spatial_shape=spatial_shape,
            )
        )(x=got)
        chex.assert_trees_all_equal(reversed_x, x)

    @chex.all_variants()
    def test_window_partition_value(
        self,
    ) -> None:
        """Test window_partition and window_unpartition output values.

        window_unpartition_3d is also tested by reversing the output.
        """
        window_shape = (1, 1, 1)
        spatial_shape = (2, 2, 2)
        x = np.array(range(8)).reshape(spatial_shape)[None, ..., None]
        got = self.variant(
            partial(window_partition, window_shape=window_shape)
        )(
            x=x,
        )
        expected = np.array(range(8))[None, :, None, None]
        chex.assert_trees_all_equal(got, expected)

        reversed_x = self.variant(
            partial(
                window_unpartition,
                window_shape=window_shape,
                spatial_shape=spatial_shape,
            )
        )(x=got)
        chex.assert_trees_all_equal(reversed_x, x)


class TestGetWindowMaskIndex(chex.TestCase):
    """Test get_window_mask_index."""

    # pmap requires at least one argument
    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, without_device=True
    )
    @parameterized.named_parameters(
        (
            "2d",
            (6, 7),
            (4, 3),
            (1, 2),
        ),
        (
            "3d",
            (6, 7, 8),
            (4, 3, 2),
            (1, 2, 1),
        ),
    )
    def test_get_window_mask_index(
        self,
        spatial_shape: tuple[int, ...],
        window_shape: tuple[int, ...],
        shift_shape: tuple[int, ...],
    ) -> None:
        """Test get_window_mask_index.

        Args:
            spatial_shape: 2 or 3d.
            window_shape: same shape as spatial_shape.
            shift_shape: same shape as spatial_shape.
        """
        assert len(spatial_shape) in [2, 3]
        expected = np.zeros(spatial_shape)
        if len(spatial_shape) == 2:
            h_slices = (
                slice(0, -window_shape[0]),
                slice(-window_shape[0], -shift_shape[0]),
                slice(-shift_shape[0], None),
            )
            w_slices = (
                slice(0, -window_shape[1]),
                slice(-window_shape[1], -shift_shape[1]),
                slice(-shift_shape[1], None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    expected[h, w] = cnt
                    cnt += 1
        else:
            h_slices = (
                slice(0, -window_shape[0]),
                slice(-window_shape[0], -shift_shape[0]),
                slice(-shift_shape[0], None),
            )
            w_slices = (
                slice(0, -window_shape[1]),
                slice(-window_shape[1], -shift_shape[1]),
                slice(-shift_shape[1], None),
            )
            d_slices = (
                slice(0, -window_shape[2]),
                slice(-window_shape[2], -shift_shape[2]),
                slice(-shift_shape[2], None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    for d in d_slices:
                        expected[h, w, d] = cnt
                        cnt += 1
        expected = expected.astype(np.int32)
        got = self.variant(
            partial(
                get_window_mask_index,
                spatial_shape=spatial_shape,
                window_shape=window_shape,
                shift_shape=shift_shape,
            )
        )()
        chex.assert_trees_all_equal(got, expected)


class TestGetWindowMask(chex.TestCase):
    """Test get_window_mask."""

    # pmap requires at least one argument
    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, without_device=True
    )
    @parameterized.named_parameters(
        (
            "2d",
            (6, 9),
            (2, 3),
            (1, 2),
        ),
        (
            "3d",
            (6, 8, 9),
            (2, 4, 3),
            (1, 2, 1),
        ),
    )
    def test_get_window_mask(
        self,
        spatial_shape: tuple[int, ...],
        window_shape: tuple[int, ...],
        shift_shape: tuple[int, ...],
    ) -> None:
        """Test get_window_mask.

        Args:
            spatial_shape: 2 or 3d.
            window_shape: same shape as spatial_shape.
            shift_shape: same shape as spatial_shape.
        """
        assert len(spatial_shape) in [2, 3]
        window_volume = np.prod(window_shape)
        num_windows = np.prod(
            tuple(ss // ws for ss, ws in zip(spatial_shape, window_shape))
        )
        got = self.variant(
            partial(
                get_window_mask,
                spatial_shape=spatial_shape,
                window_shape=window_shape,
                shift_shape=shift_shape,
            )
        )()
        chex.assert_shape(got, (num_windows, window_volume, window_volume))


@pytest.mark.parametrize(
    (
        "spatial_shape",
        "window_shape",
        "shift_shape",
        "expected_window_shape",
        "expected_spatial_padding",
        "expected_padded_spatial_shape",
        "expected_shift_shape",
        "expected_neg_shift_shape",
    ),
    [
        # 2D pad only
        (
            (5, 6),
            (2, 2),
            (1, 1),
            (2, 2),
            (1, 0),
            (6, 6),
            (1, 1),
            (-1, -1),
        ),
        # 2D no shift
        (
            (5, 6),
            (2, 2),
            (0, 0),
            (2, 2),
            (1, 0),
            (6, 6),
            (0, 0),
            (0, 0),
        ),
        # 2D irregular window and shift
        (
            (6, 6),
            (3, 4),
            (2, 3),
            (3, 4),
            (0, 2),
            (6, 8),
            (2, 3),
            (-2, -3),
        ),
        # 2D large window
        (
            (5, 3),
            (2, 4),
            (1, 3),
            (2, 3),
            (1, 0),
            (6, 3),
            (1, 0),
            (-1, 0),
        ),
        # 3D pad only
        (
            (5, 6, 7),
            (2, 2, 2),
            (1, 1, 1),
            (2, 2, 2),
            (1, 0, 1),
            (6, 6, 8),
            (1, 1, 1),
            (-1, -1, -1),
        ),
        # 3D no shift
        (
            (5, 6, 7),
            (2, 2, 2),
            (0, 0, 0),
            (2, 2, 2),
            (1, 0, 1),
            (6, 6, 8),
            (0, 0, 0),
            (0, 0, 0),
        ),
        # 3D irregular window and shift
        (
            (5, 6, 6),
            (2, 3, 4),
            (1, 2, 3),
            (2, 3, 4),
            (1, 0, 2),
            (6, 6, 8),
            (1, 2, 3),
            (-1, -2, -3),
        ),
        # 3D large window
        (
            (5, 6, 3),
            (2, 3, 4),
            (1, 2, 3),
            (2, 3, 3),
            (1, 0, 0),
            (6, 6, 3),
            (1, 2, 0),
            (-1, -2, 0),
        ),
    ],
)
def test_get_window_shift_pad_shapes(
    spatial_shape: tuple[int, int, int],
    window_shape: tuple[int, int, int],
    shift_shape: tuple[int, int, int],
    expected_window_shape: tuple[int, int, int],
    expected_spatial_padding: tuple[int, int, int],
    expected_padded_spatial_shape: tuple[int, int, int],
    expected_shift_shape: tuple[int, int, int],
    expected_neg_shift_shape: tuple[int, int, int],
) -> None:
    """Test get_window_shift_pad_shapes outputs.

    Args:
        spatial_shape: (w, h, d).
        window_shape: (ww, wh, wd).
        shift_shape: (sw, wh, wd).
        expected_window_shape: (ww, wh, wd).
        expected_spatial_padding: (pw, ph, pd)
        expected_padded_spatial_shape: (w, h, d).
        expected_shift_shape: (sw, wh, wd).
        expected_neg_shift_shape: -(sw, wh, wd).
    """
    (
        got_window_shape,
        got_spatial_padding,
        got_padded_spatial_shape,
        got_shift_shape,
        got_neg_shift_shape,
    ) = get_window_shift_pad_shapes(
        spatial_shape=spatial_shape,
        window_shape=window_shape,
        shift_shape=shift_shape,
    )
    assert got_window_shape == expected_window_shape
    assert got_spatial_padding == expected_spatial_padding
    assert got_padded_spatial_shape == expected_padded_spatial_shape
    assert got_shift_shape == expected_shift_shape
    assert got_neg_shift_shape == expected_neg_shift_shape
