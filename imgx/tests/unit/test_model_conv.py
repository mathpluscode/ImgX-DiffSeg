"""Test convolutional layers."""
from __future__ import annotations

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from chex._src import fake

from imgx.model.conv import PatchEmbedding


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestPatchEmbedding(chex.TestCase):
    """Test PatchEmbedding."""

    model_size = 2
    batch = 2

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "3d - evenly divided rectangular patch",
            (4, 6, 8),
            (2, 3, 4),
        ),
        (
            "3d - non evenly divided cube patch",
            (4, 6, 8),
            (3, 3, 3),
        ),
        (
            "3d - non evenly divided rectangular patch",
            (4, 7, 8),
            (3, 2, 3),
        ),
        (
            "2d - evenly divided rectangular patch",
            (4, 6),
            (2, 3),
        ),
        (
            "2d - non evenly divided cube patch",
            (4, 7),
            (3, 4),
        ),
    )
    def test_shape(
        self,
        spatial_shape: tuple[int, ...],
        patch_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes.

        Args:
            spatial_shape: image shape.
            patch_shape: patch shape.
        """
        num_spatial_dims = len(spatial_shape)

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            image: jnp.ndarray,
        ) -> jnp.ndarray:
            """Forward function.

            Args:
                image: shape (batch, H, W, D, C).

            Returns:
                Network prediction.
            """
            return PatchEmbedding(
                patch_shape=patch_shape,
                model_size=self.model_size,
            )(x=image)

        patched_shape = tuple(
            (spatial_shape[i] + patch_shape[i] - 1) // patch_shape[i]
            for i in range(num_spatial_dims)
        )
        key = jax.random.PRNGKey(0)
        dummy_image = jax.random.uniform(
            key, shape=(self.batch, *spatial_shape, self.model_size)
        )
        out = forward(image=dummy_image)
        chex.assert_shape(out, (self.batch, *patched_shape, self.model_size))
