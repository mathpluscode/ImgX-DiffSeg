"""Window attention layers."""
from __future__ import annotations

import dataclasses
from typing import Callable

import haiku as hk
import jax.lax
import jax.nn
import numpy as np
from jax import numpy as jnp

from imgx.model.basic import MLP, layer_norm
from imgx.model.window import window_partition, window_unpartition


def get_rel_pos(q_size: int, k_size: int, rel_pos: jnp.ndarray) -> jnp.ndarray:
    """Get relative positional embeddings for one axis.

    Args:
        q_size: size of query q.
        k_size: size of key k.
        rel_pos: relative position embeddings (max_rel_dist_pretrained, d).

    Returns:
        Extracted positional embeddings according to relative positions,
            shape (q_size, k_size, d)
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    rel_pos_resized = rel_pos
    if rel_pos.shape[0] != max_rel_dist:
        # interpolate rel pos to (max_rel_dist, d)
        rel_pos_resized = jax.image.resize(
            image=rel_pos,
            shape=(max_rel_dist, rel_pos.shape[1]),
            method="linear",
            antialias=False,
        )

    # scale the coords with short length if shapes for q and k are different.
    # (q_size, 1), values in [0, max(k_size,q_size)-1] (both inclusive)
    q_coords = jnp.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    # (1, k_size), values in [0, max(k_size,q_size)-1] (both inclusive)
    k_coords = jnp.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    # (q_size, k_size), values in [0, 2*max(k_size,q_size)-2] (both inclusive)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(
        q_size / k_size, 1.0
    )
    # (q_size, k_size, d)
    return rel_pos_resized[relative_coords.astype(jnp.int32)]


def add_decomposed_relative_positional_bias(
    query: jnp.ndarray,
    attn_logits: jnp.ndarray,
    rel_pos_list: list[jnp.ndarray],
    query_window_shape: tuple[int, ...],
    key_window_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Calculate decomposed relative positional bias.

    https://arxiv.org/abs/2112.01526

    Args:
        query: query used for attention,
            shape (..., num_queries, num_heads, key_size)
            num_queries = query_window_volume.
        attn_logits: attention logits of shape
            (..., num_heads, num_queries, num_keys),
            num_queries = query_window_volume,
            num_keys = key_window_volume.
        rel_pos_list: list of relative positional bias per axis,
            for each axis of size sh in pretrained_window_shape,
            shape is (2*sh-1, key_size).
        query_window_shape: window shape for query.
        key_window_shape: window shape for key.

    Returns:
        attention logits of shape
            (..., num_heads, num_queries, num_keys).
    """
    *leading_axis, num_queries, num_heads, key_size = query.shape
    num_keys = attn_logits.shape[-1]
    if num_queries != np.prod(query_window_shape):
        raise ValueError(
            f"num_queries {num_queries} != "
            f"query_window_volume {np.prod(query_window_shape)}"
        )
    num_spacial_dims = len(query_window_shape)
    if num_spacial_dims > 5:
        raise ValueError(
            f"num_spacial_dims > 5 not supported, got {num_spacial_dims}."
        )
    axes_str = "abcef"[:num_spacial_dims]  # used for einsum

    # (..., *query_window_shape, num_heads, key_size)
    query = query.reshape(
        (*leading_axis, *query_window_shape, num_heads, key_size)
    )
    # (..., num_heads, *query_window_shape, *key_window_shape)
    attn_logits = attn_logits.reshape(
        (*leading_axis, num_heads, *query_window_shape, *key_window_shape)
    )

    # add relative positional bias for each axis
    num_spacial_dims = len(query_window_shape)
    for i in range(num_spacial_dims):
        # (sh_q, sh_k, key_size)
        # where sh_q = query_window_shape[i]
        # and sh_k = key_window_shape[i]
        rel_pos = get_rel_pos(
            q_size=query_window_shape[i],
            k_size=key_window_shape[i],
            rel_pos=rel_pos_list[i],
        )
        # (..., num_heads, *query_window_shape, sh_k)
        einsum_str = f"...{axes_str}hd,{axes_str[i]}kd->...h{axes_str}k"
        rel_pos_q = jnp.einsum(einsum_str, query, rel_pos)
        # (..., num_heads, *query_window_shape, *key_shape)
        # key_shape has size sh_k for axis i and 1 for other axes
        key_shape = tuple(
            1 if j != i else key_window_shape[i]
            for j in range(num_spacial_dims)
        )
        rel_pos_q = rel_pos_q.reshape(
            (*leading_axis, num_heads, *query_window_shape, *key_shape)
        )
        # (..., num_heads, *query_window_shape, *key_window_shape)
        attn_logits += rel_pos_q
    attn_logits = attn_logits.reshape(
        (*leading_axis, num_heads, num_queries, num_keys)
    )
    return attn_logits


@dataclasses.dataclass
class WindowMultiHeadAttention(hk.MultiHeadAttention):
    """Overload hk.MultiHeadAttention to use attention with positional bias.

    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        pretrained_window_shape: tuple[int, ...],
        w_init: hk.initializers.Initializer,
        **kwargs,
    ) -> None:
        """Multi-head attention with efficient computation.

        Args:
            window_shape: window shape.
            pretrained_window_shape: window shape of pretrained model.
            w_init: initializer for the weights.
            kwargs: additional arguments.
        """
        if w_init is None:
            w_init = hk.initializers.VarianceScaling()
        super().__init__(w_init=w_init, **kwargs)
        self.pretrained_window_shape = pretrained_window_shape

    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None,
        window_shape: tuple[int, ...],
    ) -> jnp.ndarray:
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more
        'batch-like' leading dimensions.

        Args:
            x: shape (..., window_volume, model_size).
            mask: Optional mask applied to attention weights;
                mask: shape (..., 1, window_volume, window_volume).
            window_shape: window shape, window_volume = prod(window_shape).

        Returns:
            A new sequence of embeddings, consisting of a projection of the
                attention-weighted value projections; shape [..., T', D'].
        """
        if x.shape[-2] != np.prod(window_shape):
            raise ValueError(
                f"Input shape {x.shape} does not match "
                f"window shape {window_shape}."
            )
        # query: shape [..., T', D_q] = [..., window_volume, model_size]
        # key: shape [..., T, D_k] = [..., window_volume, model_size]
        # value: shape [..., T, D_v] = [..., window_volume, model_size]
        query, key, value = x, x, x

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
        attn_logits = jnp.einsum(
            "...thd,...Thd->...htT", query_heads, key_heads
        )
        attn_logits = attn_logits / np.sqrt(self.key_size).astype(key.dtype)

        # START RELATIVE POSITIONAL BIAS
        # for each axis (2*sh-1, key_size)
        rel_pos_list = [
            hk.get_parameter(
                name=self.name + f"_rel_pos_axis_{i}",
                shape=(2 * sh - 1, self.key_size),
                init=hk.initializers.Constant(0.0),
            )
            for i, sh in enumerate(self.pretrained_window_shape)
        ]
        # [H, T', T], H=num_heads
        attn_logits = add_decomposed_relative_positional_bias(
            query=query_heads,
            attn_logits=attn_logits,
            rel_pos_list=rel_pos_list,
            query_window_shape=window_shape,
            key_window_shape=window_shape,
        )
        # END RELATIVE POSITIONAL BIAS

        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} "
                    f"must match logits dimensionality {attn_logits.ndim}."
                )
            attn_logits = jnp.where(mask, attn_logits, -1e30)
        attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = jnp.reshape(
            attn, (*leading_dims, sequence_length, -1)
        )  # [T', H*V]

        # Apply another projection to get the final embeddings.
        final_projection = hk.Linear(self.model_size, w_init=self.w_init)
        return final_projection(attn)  # [T', D']


@dataclasses.dataclass
class WindowMultiHeadAttentionBlock(hk.Module):
    """Window based multi-head attention in SAM.

    Norm - split into windows - mha - merge windows - skip.

    TODO: pretrained_window_shape = window_shape for now.

    Modification: for an axis, if the window size is larger than spatial size,
    window is shrunk to avoid unnecessary padding, and no need of shift.
    """

    window_shape: tuple[int, ...]
    num_heads: int
    widening_factor: int = 4
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu
    initializer: hk.initializers.Initializer | None = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: shape (batch, *spatial_shape, model_size).

        Returns:
            Array of shape (batch, *spatial_shape, model_size).
        """
        _, *spatial_shape, model_size = x.shape
        if (model_size > self.widening_factor) and (
            model_size % self.widening_factor != 0
        ):
            raise ValueError(
                f"Model size {model_size} should be evenly divided by"
                f"widening factor {self.widening_factor}."
            )
        # ensure window shape no larger than spatial shape
        window_shape = tuple(
            min(ss, ws) for ss, ws in zip(spatial_shape, self.window_shape)
        )
        if min(window_shape) == 0:
            # global attention
            window_shape = tuple(spatial_shape)

        # norm
        shortcut = x
        x = layer_norm(x)

        # split into windows
        # (batch, num_windows, window_volume, model_size)
        x = window_partition(x=x, window_shape=window_shape)

        # attention
        key_size = model_size
        if model_size >= self.widening_factor:
            key_size = model_size // self.widening_factor
        mha = WindowMultiHeadAttention(
            pretrained_window_shape=window_shape,
            num_heads=self.num_heads,
            key_size=key_size,
            model_size=model_size,
            w_init=self.initializer,
        )
        # (batch, num_windows, window_volume, model_size)
        x = mha(x=x, mask=None, window_shape=window_shape)

        # merge windows
        # (batch, *spatial_shape, model_size)
        x = window_unpartition(
            x=x,
            window_shape=window_shape,
            spatial_shape=spatial_shape,
        )

        # skip connection
        mlp = MLP(
            emb_size=model_size * self.widening_factor,
            output_size=model_size,
        )
        x = shortcut + x
        x += mlp(layer_norm(x))

        return x


@dataclasses.dataclass
class AxialWindowMultiHeadAttentionBlock(hk.Module):
    """Axial window based multi-head attention in SAM.

    Norm - split into windows - mha - merge windows - skip.

    TODO: pretrained_window_shape = window_shape for now.

    Modification: for an axis, if the window size is larger than spatial size,
    window is shrunk to avoid unnecessary padding, and no need of shift.
    """

    window_shape: tuple[int, ...]
    num_heads: int
    widening_factor: int = 4
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu
    initializer: hk.initializers.Initializer | None = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: shape (batch, *spatial_shape, model_size).

        Returns:
            Array of shape (batch, *spatial_shape, model_size).
        """
        _, *spatial_shape, model_size = x.shape

        for i, (ss, ws) in enumerate(zip(spatial_shape, self.window_shape)):
            # move axis i to the second last
            # (batch, *other_spatial_shape, ss, model_size)
            x = jnp.moveaxis(x, i + 1, -2)
            other_spatial_shape = x.shape[1:-2]

            # merge other spatial dimensions to batch
            # (extented_batch, ss, model_size)
            x = jnp.reshape(x, (-1, ss, model_size))

            # attention
            x = WindowMultiHeadAttentionBlock(
                window_shape=(ws,),
                num_heads=self.num_heads,
                widening_factor=self.widening_factor,
                activation=self.activation,
                initializer=self.initializer,
            )(x)

            # split batch back to other spatial dimensions
            # (batch, *other_spatial_shape, ss, model_size)
            x = jnp.reshape(x, (-1, *other_spatial_shape, ss, model_size))

            # move axis back
            # (batch, *spatial_shape, model_size)
            x = jnp.moveaxis(x, -2, i + 1)

        return x
