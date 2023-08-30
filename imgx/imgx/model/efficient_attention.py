"""Efficient attention implementation.

https://arxiv.org/abs/2112.05682
https://github.com/AminRezaei0x443/memory-efficient-attention
"""
from __future__ import annotations

from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp


def summarize_chunk(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: jnp.ndarray | None,
    precision: jax.lax.Precision,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Summarize a chunk of key-value pairs.

    Args:
        query: shape (batch, num_q, num_heads, qk_features).
        key: shape (batch, num_kv, num_heads, qk_features).
        value: shape (batch, num_kv, num_heads, v_features).
        mask: shape (batch, 1, num_q, num_kv).
            consistent with hk.MultiHeadAttention.
        precision: precision for the computation.

    Returns:
        attn: shape (batch, num_q, num_heads, v_features).
            value features weighted by attention scores for each query.
        attn_weights: shape (batch, num_q, num_heads).
            summed exp of attention logits for each query.
        max_logits: shape (batch, num_q, num_heads).
            maximum attention logits for each query.
    """
    # (batch, num_q, num_heads, num_kv)
    attn_logits = jnp.einsum(
        "...qhd,...khd->...qhk", query, key, precision=precision
    )
    if mask is not None:
        if mask.ndim != attn_logits.ndim:
            raise ValueError(
                f"Mask dimensionality {mask.ndim} must "
                f"match logits dimensionality {attn_logits.ndim}."
            )
        # (batch, num_q, num_kv)
        mask = jnp.squeeze(mask, axis=1)
        # (batch, num_q, 1, num_kv)
        mask = mask[:, :, None, :]
        # (batch, num_q, num_heads, num_kv)
        attn_logits = jnp.where(mask, attn_logits, -1e30)
    # (batch, num_q, num_heads, 1)
    max_logits = jnp.max(attn_logits, axis=-1, keepdims=True)
    max_logits = jax.lax.stop_gradient(max_logits)
    # (batch, num_q, num_heads, num_kv)
    attn_weights = jnp.exp(attn_logits - max_logits)
    # (batch, num_q, num_heads, v_features)
    attn = jnp.einsum(
        "...vhf,...qhv->...qhf", value, attn_weights, precision=precision
    )
    # (batch, num_q, num_heads)
    attn_weights = jnp.sum(attn_weights, axis=-1)
    # (batch, num_q, num_heads)
    max_logits = jnp.squeeze(max_logits, axis=-1)
    return attn, attn_weights, max_logits


def attention_with_key_value_chunk(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: jnp.ndarray | None,
    key_chunk_size: int,
    precision: jax.lax.Precision,
) -> jnp.ndarray:
    """Multi-head dot product attention with chunks on key and values.

    Args:
        query: shape (batch, num_q, num_heads, qk_features).
        key: shape (batch, num_kv, num_heads, qk_features).
        value: shape (batch, num_kv, num_heads, v_features).
        mask: shape (batch, 1, num_q, num_kv).
            consistent with hk.MultiHeadAttention.
        key_chunk_size: number of key-value pairs to process at a time.
        precision: precision for the computation.

    Returns:
        query, shape (batch, num_q, num_heads, v_features).
    """
    batch_size, num_kv, num_heads, qk_features = key.shape
    v_features = value.shape[-1]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(qk_features)  # normalize query
    if num_kv % key_chunk_size != 0:
        # jax.lax.dynamic_slice does not raise error when
        # slice is out of range, so we need to pad zeros to key and value

        # if mask is None, initialize mask to all True
        if mask is None:
            num_q = query.shape[1]
            mask = jnp.ones((batch_size, 1, num_q, num_kv), dtype=jnp.bool_)

        # pad key, value, and mask
        pad_size = key_chunk_size - num_kv % key_chunk_size
        key = jnp.pad(
            key,
            pad_width=((0, 0), (0, pad_size), (0, 0), (0, 0)),
        )
        value = jnp.pad(
            value,
            pad_width=((0, 0), (0, pad_size), (0, 0), (0, 0)),
        )
        mask = jnp.pad(
            mask,
            pad_width=((0, 0), (0, 0), (0, 0), (0, pad_size)),
        ).astype(jnp.bool_)

    def chunk_scanner(
        chunk_idx: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Scan a chunk of key-value pairs.

        Args:
            chunk_idx: index of the chunk to scan.

        Returns:
            attn: shape (batch, num_q, num_heads, v_features).
                value features weighted by attention scores for each query.
            attn_weights: shape (batch, num_q, num_heads).
                summed exp of attention logits for each query.
            max_logits: shape (batch, num_q, num_heads).
                maximum attention logits for each query.
        """
        key_chunk = jax.lax.dynamic_slice(
            key,
            start_indices=(0, chunk_idx, 0, 0),
            slice_sizes=(batch_size, key_chunk_size, num_heads, qk_features),
        )
        value_chunk = jax.lax.dynamic_slice(
            value,
            start_indices=(0, chunk_idx, 0, 0),
            slice_sizes=(batch_size, key_chunk_size, num_heads, v_features),
        )
        mask_chunk = None
        if mask is not None:
            num_q = query.shape[1]
            mask_chunk = jax.lax.dynamic_slice(
                mask,
                start_indices=(0, 0, 0, chunk_idx),
                slice_sizes=(batch_size, 1, num_q, key_chunk_size),
            )
        summarize_chunk_ckpt = jax.checkpoint(
            fun=partial(summarize_chunk, precision=precision),
            prevent_cse=False,
        )
        return summarize_chunk_ckpt(
            query=query,
            key=key_chunk,
            value=value_chunk,
            mask=mask_chunk,
        )

    # chunk_attn, (num_chunks, batch, num_q, num_heads, v_features)
    # chunk_attn_weights, (num_chunks, batch, num_q, num_heads)
    # chunk_max_logits, (num_chunks, batch, num_q, num_heads)
    num_kv_maybe_padded = key.shape[1]
    chunk_attn, chunk_attn_weights, chunk_max_logits = jax.lax.map(
        chunk_scanner, xs=jnp.arange(0, num_kv_maybe_padded, key_chunk_size)
    )

    # (1, batch, num_q, num_heads)
    global_max_logits = jnp.max(chunk_max_logits, axis=0, keepdims=True)
    # (num_chunks, batch, num_q, num_heads)
    exp_max_diffs = jnp.exp(chunk_max_logits - global_max_logits)
    chunk_attn *= exp_max_diffs[..., None]
    chunk_attn_weights *= exp_max_diffs

    # (batch, num_q, num_heads, v_features)
    global_attn = jnp.sum(chunk_attn, axis=0)
    # (batch, num_q, num_heads, 1)
    global_attn_weights = jnp.sum(chunk_attn_weights, axis=0)[..., None]
    return global_attn / global_attn_weights


def attention_with_qkv_chunks(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: jnp.ndarray | None,
    query_chunk_size: int,
    key_chunk_size: int,
    precision: jax.lax.Precision,
) -> jnp.ndarray:
    """Multi-head dot product attention with chunks on query.

    Args:
        query: shape (batch, num_q, num_heads, qk_features).
        key: shape (batch, num_kv, num_heads, qk_features).
        value: shape (batch, num_kv, num_heads, v_features).
        mask: shape (batch, 1, num_q, num_kv).
            consistent with hk.MultiHeadAttention.
        query_chunk_size: number of queries to process at a time.
        key_chunk_size: number of key-value pairs to process at a time.
        precision: precision for the computation.

    Returns:
        query, shape (batch, num_q, num_heads, v_features).
    """
    batch_size, num_q, num_heads, qk_features = query.shape
    v_features = value.shape[-1]
    query_chunk_size = min(query_chunk_size, num_q)
    if num_q % query_chunk_size != 0:
        # jax.lax.dynamic_slice does not raise error when
        # slice is out of range, so we pad the query

        # if mask is None, initialize mask to all True
        if mask is None:
            num_kv = key.shape[1]
            mask = jnp.ones((batch_size, 1, num_q, num_kv), dtype=jnp.bool_)

        # pad query and mask
        pad_size = query_chunk_size - num_q % query_chunk_size
        query = jnp.pad(
            query,
            pad_width=((0, 0), (0, pad_size), (0, 0), (0, 0)),
        )
        mask = jnp.pad(
            mask,
            pad_width=((0, 0), (0, 0), (0, pad_size), (0, 0)),
        ).astype(jnp.bool_)

    def chunk_scanner(
        chunk_idx: int,
    ) -> jnp.ndarray:
        """Scan a chunk of queries.

        Args:
            chunk_idx: index of the chunk to scan.

        Returns:
            attn: shape (batch, query_chunk_size, num_heads, v_features).
        """
        query_chunk = jax.lax.dynamic_slice(
            query,
            start_indices=(0, chunk_idx, 0, 0),
            slice_sizes=(batch_size, query_chunk_size, num_heads, qk_features),
        )
        mask_chunk = None
        if mask is not None:
            num_kv = key.shape[1]
            mask_chunk = jax.lax.dynamic_slice(
                mask,
                start_indices=(0, 0, chunk_idx, 0),
                slice_sizes=(batch_size, 1, query_chunk_size, num_kv),
            )
        return attention_with_key_value_chunk(
            query=query_chunk,
            key=key,
            value=value,
            mask=mask_chunk,
            key_chunk_size=key_chunk_size,
            precision=precision,
        )

    # (num_chunks, batch, query_chunk_size, num_heads, v_features)
    num_q_maybe_padded = query.shape[1]
    attn = jax.lax.map(
        chunk_scanner,
        xs=jnp.arange(0, num_q_maybe_padded, query_chunk_size),
    )
    # (batch, num_chunks, query_chunk_size, num_heads, v_features)
    attn = jnp.moveaxis(attn, 0, 1)
    # (batch, num_q_maybe_padded, num_heads, v_features)
    attn = attn.reshape(batch_size, num_q_maybe_padded, num_heads, v_features)
    # remove padding, (batch, num_q, num_heads, v_features)
    attn = attn[:, :num_q, ...]
    return attn


class EfficientMultiHeadAttention(hk.MultiHeadAttention):
    """Overload hk.MultiHeadAttention to use efficient attention."""

    def __init__(  # type: ignore[no-untyped-def]
        self,
        w_init: hk.initializers.Initializer,
        query_chunk_size: int = 1024,
        key_chunk_size: int = 1024,
        precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
        **kwargs,
    ) -> None:
        """Multi-head attention with efficient computation.

        Args:
            w_init: initializer for the weights.
            query_chunk_size: number of queries to process at a time.
            key_chunk_size: number of key-value pairs to process at a time.
            precision: precision for the computation.
            kwargs: additional arguments.
        """
        super().__init__(w_init=w_init, **kwargs)
        self.query_chunk_size = query_chunk_size
        self.key_chunk_size = key_chunk_size
        self.precision = precision

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more
        'batch-like' leading dimensions.

        Args:
            query: Embeddings sequence used to compute queries;
                shape [..., T', D_q].
            key: Embeddings sequence used to compute keys;
                shape [..., T, D_k].
            value: Embeddings sequence used to compute values;
                shape [..., T, D_v].
            mask: Optional mask applied to attention weights;
                shape [..., H=1, T', T].

        Returns:
            A new sequence of embeddings, consisting of a projection of the
                attention-weighted value projections; shape [..., T', D'].
        """
        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        *leading_dims, sequence_length, _ = query.shape
        projection = self._linear_projection

        # Compute key/query/values
        # (overload K/Q/V to denote the respective sizes).
        query_heads = projection(query, self.key_size, "query")  # [T', H, Q=K]
        key_heads = projection(key, self.key_size, "key")  # [T, H, K]
        value_heads = projection(value, self.value_size, "value")  # [T, H, V]

        # Compute attention weights.
        # START OVERLOAD
        attn = attention_with_qkv_chunks(
            query=query_heads,
            key=key_heads,
            value=value_heads,
            mask=mask,
            query_chunk_size=self.query_chunk_size,
            key_chunk_size=self.key_chunk_size,
            precision=self.precision,
        )  # [T', H, V]
        # END OVERLOAD

        attn = jnp.reshape(
            attn, (*leading_dims, sequence_length, -1)
        )  # [T', H*V]

        # Apply another projection to get the final embeddings.
        final_projection = hk.Linear(self.model_size, w_init=self.w_init)
        return final_projection(attn)  # [T', D']
