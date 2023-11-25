"""Test segmentation loss."""
from functools import partial

import chex
import jax.numpy as jnp
from absl.testing import parameterized
from chex._src import fake

from imgx.loss.segmentation import segmentation_loss
from imgx_datasets import INFO_MAP


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestSegmentationLoss(chex.TestCase):
    """Test segmentation_loss."""

    batch_size = 2

    @chex.all_variants()
    @parameterized.product(
        dataset_name=sorted(INFO_MAP.keys()),
        loss_config=[
            {
                "cross_entropy": 1.0,
                "dice": 1.0,
                "focal": 1.0,
            },
            {
                "dice": 1.0,
            },
        ],
    )
    def test_shape(
        self,
        dataset_name: str,
        loss_config: dict[str, float],
    ) -> None:
        """Test return shapes.

        Args:
            dataset_name: dataset name.
            loss_config: loss config.
        """
        dataset_info = INFO_MAP[dataset_name]
        shape = dataset_info.image_spatial_shape
        shape = tuple(max(x // 16, 2) for x in shape)  # reduce shape to speed up test
        logits = jnp.ones(
            (self.batch_size, *shape, dataset_info.num_classes),
            dtype=jnp.float32,
        )
        label = jnp.ones((self.batch_size, *shape), dtype=jnp.int32)

        got_loss_batch, got_metrics = self.variant(
            partial(
                segmentation_loss,
                dataset_info=dataset_info,
                loss_config=loss_config,
            )
        )(logits, label)
        chex.assert_shape(got_loss_batch, (self.batch_size,))
        for v in got_metrics.values():
            chex.assert_shape(v, (self.batch_size,))
