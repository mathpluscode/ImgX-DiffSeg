"""Tests for attention related functions."""
from __future__ import annotations

from functools import partial

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from chex._src import fake

from imgx.model.efficient_attention import (
    dot_product_attention_with_key_value_chunk,
    dot_product_attention_with_qkv_chunks,
    summarize_chunk,
)


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


def hk_mha_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: jnp.ndarray | None,
) -> jnp.ndarray:
    """Multi-head attention from hk.MultiHeadAttention.

    Args:
        query: shape (batch, num_q, num_heads, qk_features).
        key: shape (batch, num_kv, num_heads, qk_features).
        value: shape (batch, num_kv, num_heads, v_features).
        mask: shape (batch, 1, num_q, num_kv).
            consistent with hk.MultiHeadAttention.

    Returns:
        shape (batch, num_q, num_heads, v_features).
    """
    qk_features = query.shape[-1]
    attn_logits = jnp.einsum("...thd,...Thd->...htT", query, key)
    attn_logits = attn_logits / jnp.sqrt(qk_features).astype(key.dtype)
    if mask is not None:
        if mask.ndim != attn_logits.ndim:
            raise ValueError(
                f"Mask dimensionality {mask.ndim} must "
                f"match logits dimensionality {attn_logits.ndim}."
            )
        attn_logits = jnp.where(mask, attn_logits, -1e30)
    attn_weights = jax.nn.softmax(attn_logits)
    return jnp.einsum("...htT,...Thd->...thd", attn_weights, value)


class TestSummarizeChunk(chex.TestCase):
    """Test the function summarize_chunk."""

    batch_size = 2
    num_heads = 4

    @chex.all_variants()
    @parameterized.product(
        num_q=[3, 4],
        num_kv=[3, 4],
        qk_features=[7, 8],
        v_features=[8],
        use_mask=[False, True],
    )
    def test_shapes(
        self,
        num_q: int,
        num_kv: int,
        qk_features: int,
        v_features: int,
        use_mask: bool,
    ) -> None:
        """Test output shapes under different device condition.

        Args:
            num_q: number of tokens for queries.
            num_kv: number of tokens for keys and values.
            qk_features: dimension of features for keys.
            v_features: dimension of features for values.
            use_mask: whether to use mask.
        """
        rng = jax.random.PRNGKey(0)
        rng_query, rng_key, rng_value, rng_mask = jax.random.split(rng, 4)
        query = jax.random.uniform(
            rng_query,
            shape=(self.batch_size, num_q, self.num_heads, qk_features),
        )
        key = jax.random.uniform(
            rng_key,
            shape=(self.batch_size, num_kv, self.num_heads, qk_features),
        )
        value = jax.random.uniform(
            rng_value,
            shape=(self.batch_size, num_kv, self.num_heads, v_features),
        )
        mask = None
        if use_mask:
            mask = jax.random.uniform(
                rng_mask,
                shape=(self.batch_size, 1, num_q, num_kv),
                minval=0,
                maxval=1,
            )
            mask = jnp.asarray(mask > jnp.mean(mask), dtype=jnp.float32)

        got_attn, got_attn_weights, got_max_logits = self.variant(
            partial(summarize_chunk, precision=jax.lax.Precision.DEFAULT),
        )(
            query=query,
            key=key,
            value=value,
            mask=mask,
        )

        # check shapes
        chex.assert_shape(got_attn, (self.batch_size, num_q, self.num_heads, v_features))
        chex.assert_shape(got_attn_weights, (self.batch_size, num_q, self.num_heads))
        chex.assert_shape(got_max_logits, (self.batch_size, num_q, self.num_heads))


class TestDotProductAttentionWithKVChunk(chex.TestCase):
    """Test the function dot_product_attention_with_key_value_chunk."""

    batch_size = 2
    num_heads = 4

    @chex.all_variants()
    @parameterized.product(
        num_kv=[4],
        key_chunk_size=[2, 3, 4],
        num_q=[3],
        qk_features=[8],
        v_features=[8],
        use_mask=[False, True],
    )
    def test_shapes(
        self,
        num_q: int,
        num_kv: int,
        qk_features: int,
        v_features: int,
        key_chunk_size: int,
        use_mask: bool,
    ) -> None:
        """Test output shapes under different device condition.

        Args:
            num_q: number of tokens for queries.
            num_kv: number of tokens for keys and values.
            qk_features: dimension of features for keys.
            v_features: dimension of features for values.
            key_chunk_size: number of key-value pairs to process at a time.
            use_mask: whether to use mask.
        """
        rng = jax.random.PRNGKey(0)
        rng_query, rng_key, rng_value, rng_mask = jax.random.split(rng, 4)
        query = jax.random.uniform(
            rng_query,
            shape=(self.batch_size, num_q, self.num_heads, qk_features),
        )
        key = jax.random.uniform(
            rng_key,
            shape=(self.batch_size, num_kv, self.num_heads, qk_features),
        )
        value = jax.random.uniform(
            rng_value,
            shape=(self.batch_size, num_kv, self.num_heads, v_features),
        )
        mask = None
        if use_mask:
            mask = jax.random.uniform(
                rng_mask,
                shape=(self.batch_size, 1, num_q, num_kv),
                minval=0,
                maxval=1,
            )
            mask = jnp.asarray(mask > jnp.mean(mask), dtype=jnp.float32)

        got = self.variant(
            partial(
                dot_product_attention_with_key_value_chunk,
                precision=jax.lax.Precision.DEFAULT,
                key_chunk_size=key_chunk_size,
            ),
        )(
            query=query,
            key=key,
            value=value,
            mask=mask,
        )

        # check shape
        chex.assert_shape(got, (self.batch_size, num_q, self.num_heads, v_features))

        # check values
        expected = hk_mha_attention(query, key, value, mask)
        chex.assert_tree_all_close(got, expected)


class TestDotProductAttentionWithQKVChunks(chex.TestCase):
    """Test the function dot_product_attention_with_qkv_chunks."""

    batch_size = 2
    num_heads = 4

    @chex.all_variants()
    @parameterized.product(
        num_kv=[4],
        key_chunk_size=[2, 3, 4],
        num_q=[4],
        query_chunk_size=[2, 3, 4],
        qk_features=[8],
        v_features=[8],
        use_mask=[False, True],
    )
    def test_shapes(
        self,
        num_q: int,
        num_kv: int,
        qk_features: int,
        v_features: int,
        query_chunk_size: int,
        key_chunk_size: int,
        use_mask: bool,
    ) -> None:
        """Test output shapes under different device condition.

        Args:
            num_q: number of tokens for queries.
            num_kv: number of tokens for keys and values.
            qk_features: dimension of features for keys.
            v_features: dimension of features for values.
            query_chunk_size: number of queries to process at a time.
            key_chunk_size: number of key-value pairs to process at a time.
            use_mask: whether to use mask.
        """
        rng = jax.random.PRNGKey(0)
        rng_query, rng_key, rng_value, rng_mask = jax.random.split(rng, 4)
        query = jax.random.uniform(
            rng_query,
            shape=(self.batch_size, num_q, self.num_heads, qk_features),
        )
        key = jax.random.uniform(
            rng_key,
            shape=(self.batch_size, num_kv, self.num_heads, qk_features),
        )
        value = jax.random.uniform(
            rng_value,
            shape=(self.batch_size, num_kv, self.num_heads, v_features),
        )
        mask = None
        if use_mask:
            mask = jax.random.uniform(
                rng_mask,
                shape=(self.batch_size, 1, num_q, num_kv),
                minval=0,
                maxval=1,
            )
            mask = jnp.asarray(mask > jnp.mean(mask), dtype=jnp.float32)

        got = self.variant(
            partial(
                dot_product_attention_with_qkv_chunks,
                query_chunk_size=query_chunk_size,
                key_chunk_size=key_chunk_size,
                precision=jax.lax.Precision.DEFAULT,
            ),
        )(
            query=query,
            key=key,
            value=value,
            mask=mask,
        )

        # check shape
        chex.assert_shape(got, (self.batch_size, num_q, self.num_heads, v_features))

        # check values with haiku implementation
        expected_hk = hk_mha_attention(query, key, value, mask)
        chex.assert_tree_all_close(got, expected_hk)

        # check values with flax implementation
        expected_flax = nn.dot_product_attention(query=query, key=key, value=value, mask=mask)
        chex.assert_tree_all_close(got, expected_flax)
