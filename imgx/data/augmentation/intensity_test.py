"""Test function for intensity data augmentation."""

import chex
import jax
from absl.testing import parameterized
from chex._src import fake

from imgx.data.augmentation.intensity import batch_random_adjust_gamma, batch_rescale_intensity
from imgx.datasets.constant import IMAGE


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestRandomAdjustGamma(chex.TestCase):
    """Test batch_random_adjust_gamma."""

    @chex.all_variants()
    @parameterized.product(
        image_shape=[(8, 12, 6), (8, 12)],
        max_log_gamma=[0.0, 0.3],
        batch_size=[4, 1],
    )
    def test_shapes(
        self,
        batch_size: int,
        max_log_gamma: float,
        image_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes.

        Args:
            batch_size: number of samples in batch.
            max_log_gamma: maximum log gamma.
            image_shape: image spatial shape.
        """
        key = jax.random.PRNGKey(0)
        image = jax.random.uniform(key=key, shape=(batch_size, *image_shape), minval=0, maxval=1)
        batch = {IMAGE: image}
        got = self.variant(batch_random_adjust_gamma)(
            key=key,
            batch=batch,
            max_log_gamma=max_log_gamma,
        )

        assert len(got) == 1
        chex.assert_shape(got[IMAGE], (batch_size, *image_shape))


class TestRescaleIntensity(chex.TestCase):
    """Test batch_rescale_intensity."""

    @chex.all_variants()
    @parameterized.product(
        image_shape=[(8, 12, 6), (8, 12)],
        v_min=[0.0, 0.3],
        v_max=[1.0, 0.5],
        batch_size=[4, 1],
    )
    def test_shapes(
        self,
        batch_size: int,
        v_min: float,
        v_max: float,
        image_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes.

        Args:
            batch_size: number of samples in batch.
            v_min: minimum intensity.
            v_max: maximum intensity.
            image_shape: image spatial shape.
        """
        key = jax.random.PRNGKey(0)
        image = jax.random.uniform(key=key, shape=(batch_size, *image_shape), minval=0, maxval=1)
        batch = {IMAGE: image}
        got = self.variant(batch_rescale_intensity)(
            key=key,
            batch=batch,
            v_min=v_min,
            v_max=v_max,
        )

        assert len(got) == 1
        chex.assert_shape(got[IMAGE], (batch_size, *image_shape))
