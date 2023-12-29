"""Generic transformer related functions.

https://github.com/deepmind/dm-haiku/blob/main/examples/transformer/model.py
https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from imgx.model.attention.efficient_attention import dot_product_attention_with_qkv_chunks
from imgx.model.basic import MLP


class TransformerEncoder(nn.Module):
    """A transformer encoder/decoder.

    The architecture is from
    https://github.com/google-deepmind/dm-haiku/blob/main/examples/transformer/model.py
    """

    num_heads: int
    num_layers: int = 1
    autoregressive: bool = False
    widening_factor: int = 4
    add_position_embedding: bool = False
    dropout: float = 0.0
    remat: bool = True  # if True also uses efficient attention
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        is_train: bool,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Transformer encoder forward pass.

        Args:
            is_train: whether in training mode.
            x: shape (batch, ..., model_size).
            mask: shape (batch, ...) or None.
                Tokens with False values are ignored.

        Returns:
            (batch, ..., model_size).
        """
        batch, *spatial_shape, channels = x.shape
        if len(spatial_shape) > 1:
            # x: shape(batch, seq_len, model_size).
            # mask: shape(batch, seq_len) or None.
            x = x.reshape((batch, -1, channels))
            if mask is not None:
                mask = mask.reshape((batch, -1))

        kernel_init = nn.initializers.variance_scaling(
            scale=2 / self.num_layers,
            mode="fan_in",
            distribution="truncated_normal",
        )

        # define classes
        attn_cls = nn.MultiHeadDotProductAttention
        attention_fn = nn.dot_product_attention
        if self.remat:
            attn_cls = nn.remat(attn_cls)
            attention_fn = dot_product_attention_with_qkv_chunks

        _, seq_len, model_size = x.shape

        if model_size % self.widening_factor != 0:
            raise ValueError(
                f"Model size {model_size} is not divisible by widening factor "
                f"{self.widening_factor}"
            )

        # compute mask if provided
        if mask is not None:
            if self.autoregressive:
                # compute causal mask for autoregressive sequence modelling.
                # (1, 1, seq_len, seq_len)
                causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len)))
                # (batch, 1, seq_len, seq_len)
                mask = mask[:, None, None, :] * causal_mask
            else:
                # (batch, 1, 1, seq_len) * (batch, 1, seq_len, 1)
                # -> (batch, 1, seq_len, seq_len)
                mask = mask[:, None, None, :] * mask[:, None, :, None]

        # embed the input tokens and positions.
        if self.add_position_embedding:
            positional_embeddings = self.param(
                "transformer_positional_embeddings",
                nn.initializers.truncated_normal(stddev=0.02),
                (1, seq_len, model_size),
            )
            x += positional_embeddings
            x = nn.Dropout(rate=self.dropout, deterministic=not is_train)(x)

        h = x
        for _ in range(self.num_layers):
            h_attn = nn.LayerNorm(dtype=self.dtype)(h)
            h_attn = attn_cls(
                num_heads=self.num_heads,
                # head_dim = qkv_features // num_heads
                qkv_features=model_size // self.widening_factor * self.num_heads,
                out_features=model_size,
                attention_fn=attention_fn,
                kernel_init=kernel_init,
                dtype=self.dtype,
            )(inputs_q=h_attn, inputs_k=h_attn, inputs_v=h_attn, mask=mask)
            h_attn = nn.Dropout(rate=self.dropout, deterministic=not is_train)(h_attn)
            h += h_attn

            h_dense = nn.LayerNorm(dtype=self.dtype)(h)
            h_dense = MLP(
                emb_size=model_size * self.widening_factor,
                output_size=model_size,
                dtype=self.dtype,
                kernel_init=kernel_init,
                remat=self.remat,
            )(h_dense)
            h_dense = nn.Dropout(rate=self.dropout, deterministic=not is_train)(h_dense)
            h += h_dense

        h = nn.LayerNorm(dtype=self.dtype)(h)

        if len(spatial_shape) > 1:
            # (batch, spatial_shape, model_size).
            h = h.reshape((batch, *spatial_shape, model_size))

        return h
