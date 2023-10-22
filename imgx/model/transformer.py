"""Generic transformer related functions."""
from __future__ import annotations

import chex
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from imgx.model.basic import MLP, LayerNorm, truncated_normal
from imgx.model.efficient_attention import dot_product_attention_with_qkv_chunks


class TransformerEncoder(nn.Module):
    """A transformer encoder/decoder.

    Removed dropout.

    Different from deepmind:
        - Feed forward layer has an extra dropout between two Linear layers.
        - Auto-regressive mask is optional.

    Different from google-research:
        - Initializer scale for MHA is smaller given more layers.
        - Auto-regressive mask is possible.

    Initialization settings is from
    https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py

    https://github.com/deepmind/dm-haiku/blob/main/examples/transformer/model.py
    https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
    """

    num_heads: int
    num_layers: int
    autoregressive: bool
    widening_factor: int = 4
    add_position_embedding: bool = True
    remat: bool = True  # if True also uses efficient attention
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
        """Transformer encoder forward pass.

        Args:
            x: shape (batch, seq_len, model_size).
            mask: shape (batch, seq_len) or None.
                Tokens with False values are ignored.

        Returns:
            - Transformed embeddings,  shape (batch, seq_len, model_size).
            - Hidden embeddings after each block.
        """
        chex.assert_rank(x, 3)
        kernel_init = nn.initializers.variance_scaling(
            scale=2 / self.num_layers,
            mode="fan_in",
            distribution="truncated_normal",
        )

        # define classes
        attn_cls = nn.MultiHeadDotProductAttention
        attention_fn = nn.dot_product_attention
        mlp_cls = MLP
        if self.remat:
            attn_cls = nn.remat(attn_cls)
            attention_fn = dot_product_attention_with_qkv_chunks
            mlp_cls = nn.remat(mlp_cls)

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
                truncated_normal(stddev=0.02),
                (1, seq_len, model_size),
            )
            x += positional_embeddings

        h = x
        hidden_embeddings = []
        for _ in range(self.num_layers):
            h_attn = attn_cls(
                num_heads=self.num_heads,
                # head_dim = qkv_features // num_heads
                qkv_features=model_size // self.widening_factor * self.num_heads,
                out_features=model_size,
                attention_fn=attention_fn,
                kernel_init=kernel_init,
                dtype=self.dtype,
            )(inputs_q=h, inputs_kv=h, mask=mask)
            h_attn = LayerNorm(dtype=self.dtype)(h_attn)
            h = h + h_attn

            h_dense = mlp_cls(
                emb_size=model_size * self.widening_factor,
                output_size=model_size,
                dtype=self.dtype,
                kernel_init=kernel_init,
            )(h)
            h_dense = LayerNorm(dtype=self.dtype)(h_dense)
            h = h + h_dense

            # save intermediate hidden embeddings
            hidden_embeddings.append(h)

        return LayerNorm(dtype=self.dtype)(h), hidden_embeddings
