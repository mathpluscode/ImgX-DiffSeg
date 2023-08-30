"""Generic transformer related functions."""
from __future__ import annotations

import dataclasses

import chex
import haiku as hk
import jax.numpy as jnp
import numpy as np

from imgx.model.basic import MLP, layer_norm
from imgx.model.efficient_attention import EfficientMultiHeadAttention


@dataclasses.dataclass
class TransformerEncoder(hk.Module):
    """A transformer encoder/decoder.

    Removed dropout.

    Different from deepmind:
        - Feed forward layer has an extra dropout between two Linear layers.
        - Auto-regressive mask is optional.

    Different from google-research:
        - Initializer scale for MHA is smaller given more layers.
        - Auto-regressive mask is possible.

    https://github.com/deepmind/dm-haiku/blob/main/examples/transformer/model.py
    https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
    """

    num_heads: int
    num_layers: int
    autoregressive: bool
    widening_factor: int = 4
    add_position_embedding: bool = True
    # reduces memory cost at cost of compute speed
    remat: bool = True
    use_efficient_attention: bool = True

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
        initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
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
                mask = mask[:, None, None, :]  # (batch, 1, 1, seq_len)
                causal_mask = np.tril(
                    np.ones((1, 1, seq_len, seq_len))
                )  # (1, 1, seq_len, seq_len)
                mask *= causal_mask  # (batch, 1, seq_len, seq_len)
            else:
                # (batch, 1, 1, seq_len) * (batch, 1, seq_len, 1)
                # -> (batch, 1, seq_len, seq_len)
                mask = mask[:, None, None, :] * mask[:, None, :, None]

        # embed the input tokens and positions.
        if self.add_position_embedding:
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            positional_embeddings = hk.get_parameter(
                name="positional_embeddings",
                shape=[1, seq_len, model_size],
                init=embed_init,
            )
            x += positional_embeddings

        h = x
        hidden_embeddings = []
        for _ in range(self.num_layers):
            # define blocks
            mha_cls = (
                EfficientMultiHeadAttention
                if self.use_efficient_attention
                else hk.MultiHeadAttention
            )
            attn_block = hk.Sequential(
                [
                    mha_cls(
                        num_heads=self.num_heads,
                        key_size=model_size // self.widening_factor,
                        model_size=model_size,
                        w_init=initializer,
                    ),
                    layer_norm,
                ]
            )
            attn_block = hk.remat(attn_block) if self.remat else attn_block
            ff_block = hk.Sequential(
                [
                    MLP(
                        emb_size=model_size * self.widening_factor,
                        output_size=model_size,
                        initializer=initializer,
                    ),
                    layer_norm,
                ]
            )
            ff_block = hk.remat(ff_block) if self.remat else ff_block

            # forward
            h_attn = attn_block(h, h, h, mask=mask)
            h = h + h_attn
            h_dense = ff_block(h)
            h = h + h_dense

            # save intermediate hidden embeddings
            hidden_embeddings.append(h)

        return layer_norm(h), hidden_embeddings
