"""Test Swin Transformer related classes and functions."""
from __future__ import annotations

from functools import partial

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from absl.testing import parameterized
from chex._src import fake

from imgx.model.window_attention import (
    AxialWindowMultiHeadAttentionBlock,
    WindowMultiHeadAttention,
    WindowMultiHeadAttentionBlock,
    add_decomposed_relative_positional_bias,
    get_rel_pos,
)


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


def get_rel_pos_torch(
    q_size: int, k_size: int, rel_pos: torch.Tensor
) -> torch.Tensor:
    """Get relative positional embeddings.

    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292

    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos: relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(
            1, 0
        )
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(
        q_size / k_size, 1.0
    )

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos_torch(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> torch.Tensor:
    """Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.

    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

    Args:
        attn: attention map.
        q: query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h: relative position embeddings (Lh, C) for height axis.
        rel_pos_w: relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn: attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    rh = get_rel_pos_torch(q_h, k_h, rel_pos_h)
    rw = get_rel_pos_torch(q_w, k_w, rel_pos_w)

    batch, _, dim = q.shape
    r_q = q.reshape(batch, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, rw)

    attn = (
        attn.view(batch, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(batch, q_h * q_w, k_h * k_w)

    return attn


class TestGetRelPos(chex.TestCase):
    """Test get_rel_pos."""

    dim = 8

    @chex.all_variants()
    @parameterized.named_parameters(
        ("same size, no interp", 2, 2, 4),
        ("diff size, no interp", 2, 3, 6),
        ("same size, extrap", 2, 2, 3),
        ("diff size, extrap", 2, 3, 4),
        ("same size, interp", 2, 2, 10),
        ("diff size, interp", 2, 3, 10),
    )
    def test_shape(
        self,
        q_size: int,
        k_size: int,
        max_rel_dist: int,
    ) -> None:
        """Test output shapes."""

        rng = jax.random.PRNGKey(0)
        rel_pos = jax.random.uniform(rng, shape=(max_rel_dist, self.dim))
        got = self.variant(partial(get_rel_pos, q_size=q_size, k_size=k_size))(
            rel_pos=rel_pos
        )
        chex.assert_shape(got, (q_size, k_size, self.dim))

        got_torch = get_rel_pos_torch(
            q_size, k_size, torch.from_numpy(np.array(rel_pos))
        )
        chex.assert_trees_all_close(got, got_torch.numpy())


class TestAddDecomposedRelativePositionalBias(chex.TestCase):
    """Test add_decomposed_relative_positional_bias."""

    num_heads = 4
    key_size = 2

    @chex.all_variants()
    @parameterized.named_parameters(
        ("2d, 1 lead dim", (2,), (4, 5), (2, 3), (5, 6)),
        ("2d, 1 lead dim extrap", (2,), (4, 5), (2, 3), (2, 3)),
        ("2d, 2 lead dims", (2, 3), (4, 5), (2, 3), (5, 6)),
        ("3d, 1 lead dim", (2,), (2, 3, 4), (4, 5, 6), (2, 3, 4)),
        ("3d, 2 lead dims", (2, 3), (4, 5, 6), (2, 3, 4), (10, 11, 12)),
    )
    def test_shape(
        self,
        leading_dims: tuple[int, ...],
        query_window_shape: tuple[int, ...],
        key_window_shape: tuple[int, ...],
        pretrained_window_shape: tuple[int, ...],
    ) -> None:
        """Test output shape."""
        num_queries = np.prod(query_window_shape)
        num_keys = np.prod(key_window_shape)

        query = jax.random.uniform(
            jax.random.PRNGKey(0),
            shape=(*leading_dims, num_queries, self.num_heads, self.key_size),
        )
        attn_logits = jax.random.uniform(
            jax.random.PRNGKey(1),
            shape=(*leading_dims, self.num_heads, num_queries, num_keys),
        )
        rel_pos_list = [
            jax.random.uniform(
                jax.random.PRNGKey(2 + i), shape=(2 * s - 1, self.key_size)
            )
            for i, s in enumerate(pretrained_window_shape)
        ]
        got = self.variant(
            partial(
                add_decomposed_relative_positional_bias,
                query_window_shape=query_window_shape,
                key_window_shape=key_window_shape,
            )
        )(
            query=query,
            attn_logits=attn_logits,
            rel_pos_list=rel_pos_list,
        )
        chex.assert_shape(
            got, (*leading_dims, self.num_heads, num_queries, num_keys)
        )

        if len(leading_dims) == 1 and len(query_window_shape) == 2:
            # (batch*num_heads, num_queries, num_keys)
            attn_logits_torch = attn_logits.reshape(-1, num_queries, num_keys)

            # (batch, num_heads, num_queries, key_size)
            query_torch = jnp.moveaxis(query, 2, 1)
            # (batch*num_heads, num_queries, key_size)
            query_torch = query_torch.reshape(-1, num_queries, self.key_size)

            # (batch*num_heads, num_queries, num_keys)
            got_torch = add_decomposed_rel_pos_torch(
                attn=torch.from_numpy(np.array(attn_logits_torch)),
                q=torch.from_numpy(np.array(query_torch)),
                rel_pos_h=torch.from_numpy(np.array(rel_pos_list[0])),
                rel_pos_w=torch.from_numpy(np.array(rel_pos_list[1])),
                q_size=(query_window_shape[0], query_window_shape[1]),
                k_size=(key_window_shape[0], key_window_shape[1]),
            )
            got_torch = got_torch.reshape(
                *leading_dims, self.num_heads, num_queries, num_keys
            )
            chex.assert_trees_all_close(got, got_torch.numpy())


class TestWindowMultiHeadAttention(chex.TestCase):
    """Test WindowMultiHeadAttention."""

    num_heads = 2
    key_size = 4
    batch_size = 2
    num_windows = 5
    in_channels = 3

    @chex.all_variants()
    @parameterized.named_parameters(
        ("3D - uniform win", (2, 2, 2), (2, 2, 2), False),
        (
            "3D - non-uniform win",
            (2, 3, 4),
            (2, 3, 4),
            False,
        ),
        (
            "3D - uniform win - mask",
            (2, 2, 2),
            (2, 2, 2),
            True,
        ),
        (
            "3D - non-uniform win - mask",
            (2, 3, 4),
            (2, 3, 4),
            True,
        ),
        (
            "3D - uniform win - pre-train",
            (2, 2, 2),
            (2, 3, 4),
            False,
        ),
        (
            "2D - non-uniform win - mask",
            (3, 4),
            (3, 4),
            True,
        ),
        (
            "2D - non-uniform win - pre-train",
            (2, 3),
            (2, 2),
            False,
        ),
    )
    def test_shape(
        self,
        window_shape: tuple[int, ...],
        pretrained_window_shape: tuple[int, ...],
        input_mask: bool,
    ) -> None:
        """Test output shapes.

        Args:
            window_shape: (wh, ww, wd).
            pretrained_window_shape: (pwh, pww, pwd), used for normalization.
            input_mask: input mask or not.
        """
        window_volume = np.prod(window_shape)

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            x: jnp.ndarray,
            mask: jnp.ndarray | None,
            window_shape: tuple[int, ...],
        ) -> jnp.ndarray:
            """Forward function.

            Args:
                x: shape (batch, num_windows, window_volume, channel).
                mask: shape (num_windows, window_volume, window_volume).
                window_shape: window_volume = prod(window_shape).

            Returns:
                MHA prediction.
            """
            mha = WindowMultiHeadAttention(
                pretrained_window_shape=pretrained_window_shape,
                num_heads=self.num_heads,
                key_size=self.key_size,
                w_init=hk.initializers.VarianceScaling(),
            )
            return mha(x=x, mask=mask, window_shape=window_shape)

        model_size = self.key_size * self.num_heads
        in_key = jax.random.PRNGKey(0)
        dummy_input = jax.random.uniform(
            in_key,
            shape=(
                self.batch_size,
                self.num_windows,
                window_volume,
                self.in_channels,
            ),
        )
        dummy_mask = None
        if input_mask:
            mask_shape = (self.num_windows, window_volume, window_volume)
            in_key, mask_key = jax.random.split(in_key)
            dummy_mask = jax.random.uniform(mask_key, shape=mask_shape)
            dummy_mask = dummy_mask > jnp.mean(dummy_mask)
            dummy_mask = dummy_mask[None, :, None, :, :]
        out = forward(x=dummy_input, mask=dummy_mask, window_shape=window_shape)

        chex.assert_shape(
            out,
            (
                self.batch_size,
                self.num_windows,
                window_volume,
                model_size,
            ),
        )

    @parameterized.named_parameters(
        ("3D", (3, 2, 5)),
        ("2D", (3, 5)),
    )
    @hk.testing.transform_and_run
    def test_mask(
        self,
        window_shape: tuple[int, ...],
    ) -> None:
        """Test mask behaviour.

        Masked tokens should not impact other embeddings.

        Args:
            window_shape: window shape.
        """
        cutoff = 4
        window_volume = np.prod(window_shape)
        pretrained_window_shape = window_shape
        mha = WindowMultiHeadAttention(
            pretrained_window_shape=pretrained_window_shape,
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init=hk.initializers.VarianceScaling(),
        )

        in_key = jax.random.PRNGKey(0)
        in_shape = (
            self.batch_size,
            self.num_windows,
            window_volume,
            self.in_channels,
        )

        # build two embeddings, the first few windows are different
        dummy_input1 = jax.random.uniform(in_key, shape=in_shape)
        dummy_input2 = jnp.concatenate(
            [
                dummy_input1[..., :cutoff, :] * 2 + 10,
                dummy_input1[..., cutoff:, :],
            ],
            axis=-2,
        )
        # build a mask masking the first few windows
        dummy_mask = jnp.concatenate(
            [
                jnp.zeros(
                    shape=(self.num_windows, window_volume, cutoff),
                    dtype=jnp.bool_,
                ),
                jnp.ones(
                    shape=(
                        self.num_windows,
                        window_volume,
                        window_volume - cutoff,
                    ),
                    dtype=jnp.bool_,
                ),
            ],
            axis=-1,
        )
        dummy_mask = dummy_mask[None, :, None, :, :]
        out1 = mha(x=dummy_input1, mask=dummy_mask, window_shape=window_shape)
        out2 = mha(x=dummy_input2, mask=dummy_mask, window_shape=window_shape)
        # the non-masked tokens are not impacted
        chex.assert_trees_all_equal(
            out1[..., cutoff:, :], out2[..., cutoff:, :]
        )


class TestWindowMultiHeadAttentionBlock(chex.TestCase):
    """Test (Axial)WindowMultiHeadAttentionBlock."""

    batch_size = 2
    model_size = 4
    num_heads = 2
    widening_factor = 2

    @chex.all_variants()
    @parameterized.product(
        spatial_shape=[
            (8,),
            (8, 12),
            (8, 12, 16),
        ],
        window_size=[0, 2],
        block_cls=[
            WindowMultiHeadAttentionBlock,
            AxialWindowMultiHeadAttentionBlock,
        ],
    )
    def test_shape(
        self,
        spatial_shape: tuple[int, ...],
        window_size: int,
        block_cls: hk.Module,
    ) -> None:
        """Test output shapes.

        Args:
            spatial_shape: spatial shape.
            window_size: size of window.
            block_cls: class to test.
        """
        window_shape = (window_size,) * len(spatial_shape)

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
            return block_cls(
                window_shape=window_shape,
                num_heads=self.num_heads,
                widening_factor=self.widening_factor,
            )(x)

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
