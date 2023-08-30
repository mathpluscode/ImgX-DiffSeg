"""Test Swin Transformer related classes and functions."""
from __future__ import annotations

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from chex._src import fake

from imgx.model.swin_vit import (
    SwinMultiHeadAttention,
    SwinTransformerEncoder,
    SwinTransformerEncoderLayer,
)


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestSwinMultiHeadAttention(chex.TestCase):
    """Test SwinMultiHeadAttention."""

    batch_size = 2
    model_size = 4
    num_heads = 2
    widening_factor = 2

    @chex.all_variants()
    @parameterized.named_parameters(
        ("2D - with shift", (4, 6), (1, 1), (2, 2)),
        ("2D - without shift", (4, 6), (0, 0), (2, 2)),
        ("2D - pad with shift", (5, 6), (1, 1), (2, 2)),
        (
            "2D - pad with irregular shift and window",
            (5, 7),
            (0, 2),
            (2, 4),
        ),
        (
            "2D - pad with irregular shift and large window",
            (6, 7),
            (1, 2),
            (3, 4),
        ),
        ("3D - with shift", (4, 6, 8), (1, 1, 1), (2, 2, 2)),
        ("3D - without shift", (4, 6, 8), (0, 0, 0), (2, 2, 2)),
        ("3D - pad with shift", (5, 6, 7), (1, 1, 1), (2, 2, 2)),
        (
            "3D - pad with irregular shift and window",
            (5, 6, 7),
            (0, 1, 2),
            (2, 3, 4),
        ),
        (
            "3D - pad with irregular shift and large window",
            (5, 6, 7),
            (0, 1, 2),
            (6, 3, 4),
        ),
    )
    def test_shape(
        self,
        spatial_shape: tuple[int, ...],
        shift_shape: tuple[int, ...],
        window_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes.

        Args:
            spatial_shape: spatial shape.
            shift_shape: int for each axis.
            window_shape: int for each axis.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            x: jnp.ndarray,
        ) -> jnp.ndarray:
            """Forward function.

            Args:
                x: shape (batch, height, width, depth, channel).

            Returns:
                Prediction.
            """
            transformer = SwinMultiHeadAttention(
                shift_shape=shift_shape,
                window_shape=window_shape,
                num_heads=self.num_heads,
                widening_factor=self.widening_factor,
            )
            return transformer(x)

        in_key = jax.random.PRNGKey(0)
        in_shape = (
            self.batch_size,
            *spatial_shape,
            self.model_size,
        )
        dummy_input = jax.random.uniform(in_key, shape=in_shape)
        out = forward(x=dummy_input)
        chex.assert_shape(out, in_shape)


class TestSwinTransformerEncoderLayer(chex.TestCase):
    """Test SwinTransformerEncoderLayer."""

    batch_size = 2
    model_size = 4
    num_heads = 2
    num_layers = 2
    widening_factor = 2

    @chex.all_variants()
    @parameterized.named_parameters(
        ("2D - non-uniform", (8, 16), (2, 2)),
        ("2D - uniform", (8, 8), (2, 2)),
        ("3D - non-uniform", (8, 12, 16), (2, 2, 2)),
        ("3D - uniform", (8, 8, 8), (2, 2, 2)),
    )
    def test_shape(
        self,
        spatial_shape: tuple[int, ...],
        window_shape: tuple[int, ...],
    ) -> None:
        """Test output shapes.

        Args:
            spatial_shape: spatial shape.
            window_shape: window shape.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            x: jnp.ndarray,
        ) -> jnp.ndarray:
            """Forward function.

            Args:
                x: shape (batch, height, width, depth, model_size).

            Returns:
                SwinTransformerEncoderLayer prediction.
            """
            transformer = SwinTransformerEncoderLayer(
                window_shape=window_shape,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                widening_factor=self.widening_factor,
            )
            return transformer(x)

        in_key = jax.random.PRNGKey(0)
        in_shape = (
            self.batch_size,
            *spatial_shape,
            self.model_size,
        )
        out_shape = (
            self.batch_size,
            *spatial_shape,
            self.model_size,
        )
        dummy_input = jax.random.uniform(in_key, shape=in_shape)
        out = forward(x=dummy_input)
        chex.assert_shape(out, out_shape)


class TestSwinTransformerEncoder(chex.TestCase):
    """Test SwinTransformerEncoder."""

    batch_size = 2
    in_channels = 3
    widening_factor = 2

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "3D - 2 layers - non-uniform image shape",
            2,
            2,
            (2, 4),
            (8, 10, 12),
            (2, 2, 2),
            (2, 2, 2),
            True,
            2,
        ),
        (
            "3D - 2 layers - non-uniform image shape - scale 3",
            2,
            2,
            (2, 4),
            (8, 10, 12),
            (2, 2, 2),
            (2, 2, 2),
            True,
            3,
        ),
        (
            "3D - 3 layers - non-uniform image shape",
            2,
            2,
            (2, 4, 8),
            (12, 14, 15),
            (2, 2, 3),
            (2, 2, 2),
            True,
            2,
        ),
        (
            "2D - 3 layers - non-uniform image shape",
            2,
            2,
            (1, 2, 4),
            (8, 12),
            (2, 3),
            (2, 2),
            True,
            2,
        ),
    )
    def test_shape(
        self,
        num_layers: int,
        num_heads: int,
        num_channels: tuple[int, ...],
        image_shape: tuple[int, ...],
        patch_shape: tuple[int, ...],
        window_shape: tuple[int, ...],
        add_position_embedding: bool,
        scale_factor: int,
    ) -> None:
        """Test output shapes.

        Args:
            num_layers: number of layers per Layer.
            num_heads: number of heads per Layer.
            num_channels: number of channels per Layer.
            image_shape: image shape before patching.
            patch_shape: int for each axis.
            window_shape: int for each axis.
            add_position_embedding: add embedding or not.
            scale_factor: down-sample/up-sample scales.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            x: jnp.ndarray,
        ) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
            """Forward function.

            Args:
                x: shape (batch, H, W, D, C).

            Returns:
                Prediction.
            """
            transformer = SwinTransformerEncoder(
                num_layers=num_layers,
                num_heads=num_heads,
                num_channels=num_channels,
                patch_shape=patch_shape,
                window_shape=window_shape,
                widening_factor=self.widening_factor,
                add_position_embedding=add_position_embedding,
                scale_factor=scale_factor,
            )
            return transformer(x)

        in_key = jax.random.PRNGKey(0)
        in_shape = (
            self.batch_size,
            *image_shape,
            self.in_channels,
        )
        dummy_input = jax.random.uniform(in_key, shape=in_shape)
        out, out_embs = forward(x=dummy_input)
        depth = len(num_channels)
        assert len(out_embs) == depth

        chex.assert_shape(
            out,
            out_embs[-1].shape,
        )

        emb_shape = tuple(s // ps for s, ps in zip(image_shape, patch_shape))
        for ch in num_channels:
            emb = out_embs.pop(0)
            chex.assert_shape(
                emb,
                (
                    self.batch_size,
                    *emb_shape,
                    ch,
                ),
            )
            # pad if the shape cannot be evenly divided by scale_factor
            emb_shape = tuple(
                es // scale_factor + (es % scale_factor > 0) for es in emb_shape
            )
