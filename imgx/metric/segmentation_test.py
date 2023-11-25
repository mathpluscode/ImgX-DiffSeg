"""Test functions in imgx.metric.util."""


import chex
import jax
import numpy as np
from chex._src import fake

from imgx.metric.segmentation import (
    get_jit_segmentation_metrics,
    get_non_jit_segmentation_metrics,
    get_non_jit_segmentation_metrics_per_step,
)


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestGetSegmentationMetrics(chex.TestCase):
    """Test get_segmentation_metrics."""

    batch = 2
    num_classes = 3
    spatial_shape = (4, 5, 6)
    spacing = np.array((0.2, 0.5, 1.0))
    mask_shape = (batch, *spatial_shape, num_classes)

    @chex.all_variants()
    def test_jit_shapes(self) -> None:
        """Test shapes."""
        key = jax.random.PRNGKey(0)
        key_pred, key_true = jax.random.split(key)
        mask_pred = jax.random.uniform(key_pred, shape=self.mask_shape)
        mask_true = jax.random.uniform(key_true, shape=self.mask_shape)

        got = self.variant(get_jit_segmentation_metrics)(mask_pred, mask_true, self.spacing)
        for _, v in got.items():
            chex.assert_shape(v, (self.batch,))

    @chex.variants(without_jit=True, with_device=True, without_device=True)
    def test_nonjit_shapes(self) -> None:
        """Test shapes."""
        key = jax.random.PRNGKey(0)
        key_pred, key_true = jax.random.split(key)
        mask_pred = jax.random.uniform(key_pred, shape=self.mask_shape)
        mask_true = jax.random.uniform(key_true, shape=self.mask_shape)

        got = self.variant(get_non_jit_segmentation_metrics)(mask_pred, mask_true, self.spacing)
        for _, v in got.items():
            chex.assert_shape(v, (self.batch,))

    @chex.variants(without_jit=True, with_device=True, without_device=True)
    def test_nonjit_per_step_shapes(self) -> None:
        """Test shapes."""
        num_steps = 2
        key = jax.random.PRNGKey(0)
        key_pred, key_true = jax.random.split(key)
        mask_pred = jax.random.uniform(key_pred, shape=(*self.mask_shape, num_steps))
        mask_true = jax.random.uniform(key_true, shape=self.mask_shape)

        got = self.variant(get_non_jit_segmentation_metrics_per_step)(
            mask_pred, mask_true, self.spacing
        )
        for _, v in got.items():
            chex.assert_shape(v, (self.batch, num_steps))
