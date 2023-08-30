"""Test Transformer related classes and functions."""


import chex
import haiku as hk
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from chex._src import fake

from imgx.model.transformer import TransformerEncoder


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestTransformerEncoder(chex.TestCase):
    """Test TransformerEncoder."""

    num_heads: int = 8
    num_layers: int = 3
    widening_factor: int = 4

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "encoder-with-pe",
            (2, 3, 8),
            False,
            True,
        ),
        (
            "encoder-without-pe",
            (2, 3, 8),
            False,
            False,
        ),
        (
            "decoder",
            (2, 3, 8),
            True,
            True,
        ),
    )
    def test_shape(
        self,
        in_shape: tuple[int, int, int],
        autoregressive: bool,
        add_position_embedding: bool,
    ) -> None:
        """Test TransformerEncoder output shapes.

        Args:
            in_shape: input tensor shape (batch, seq_len, emb_dim).
            autoregressive: True for decoder, False for encoder.
            add_position_embedding: use positional embedding or not.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            emb: jnp.ndarray,
            mask: jnp.ndarray,
        ) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
            """Forward function.

            Args:
                emb: input embedding.
                mask: input mask.

            Returns:
                Network prediction.
            """
            transformer = TransformerEncoder(
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                autoregressive=autoregressive,
                widening_factor=self.widening_factor,
                add_position_embedding=add_position_embedding,
            )
            return transformer(x=emb, mask=mask)

        emb_key = jax.random.PRNGKey(0)
        emb_key, mask_key = jax.random.split(emb_key)
        dummy_emb = jax.random.uniform(emb_key, shape=in_shape)
        dummy_mask = jax.random.uniform(mask_key, shape=in_shape[:-1])
        dummy_mask = dummy_mask > jnp.mean(dummy_mask)
        out, hidden_embeddings = forward(
            emb=dummy_emb,
            mask=dummy_mask,
        )
        chex.assert_shape(out, dummy_emb.shape)
        assert len(hidden_embeddings) == self.num_layers
        for emb in hidden_embeddings:
            chex.assert_shape(emb, dummy_emb.shape)

    @hk.testing.transform_and_run
    def test_encoder_mask(
        self,
    ) -> None:
        """Test TransformerEncoder encoder mask behaviour.

        Masked tokens should not impact other embeddings.
        """
        batch = 2
        seq_len = 4
        emb_dim = 8
        cutoff = 2
        assert seq_len > 2
        transformer = TransformerEncoder(
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            autoregressive=False,
            widening_factor=self.widening_factor,
        )
        emb_key = jax.random.PRNGKey(0)
        # build two embeddings, the first few tokens are different
        dummy_emb1 = jax.random.uniform(
            emb_key, shape=(batch, seq_len, emb_dim)
        )
        dummy_emb2 = jnp.concatenate(
            [
                dummy_emb1[:, :cutoff, :] * 2 + 10,
                dummy_emb1[:, cutoff:, :],
            ],
            axis=1,
        )
        # build a mask masking the first two tokens
        dummy_mask = jnp.concatenate(
            [
                jnp.zeros(shape=(batch, cutoff), dtype=jnp.bool_),
                jnp.ones(shape=(batch, seq_len - cutoff), dtype=jnp.bool_),
            ],
            axis=1,
        )  # (batch, seq_len)
        out1, _ = transformer(
            x=dummy_emb1,
            mask=dummy_mask,
        )
        out2, _ = transformer(
            x=dummy_emb2,
            mask=dummy_mask,
        )
        # the non-masked tokens are not impacted
        chex.assert_trees_all_equal(out1[:, cutoff:, :], out2[:, cutoff:, :])

    @hk.testing.transform_and_run
    def test_decoder(
        self,
    ) -> None:
        """Test TransformerEncoder decoder's autoregressive behaviour.

        If changing a certain tokens, only the following tokens are impacted.
        """
        batch = 2
        seq_len = 5
        model_size = 8
        cutoff = 3
        assert seq_len > 2
        transformer = TransformerEncoder(
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            autoregressive=True,
            widening_factor=self.widening_factor,
        )
        emb_key = jax.random.PRNGKey(0)
        # build two embeddings, the token at index cutoff is different
        dummy_emb1 = jax.random.uniform(
            emb_key, shape=(batch, seq_len, model_size)
        )
        dummy_emb2 = jnp.concatenate(
            [
                dummy_emb1[:, :cutoff, :],
                dummy_emb1[:, cutoff : (cutoff + 1), :] * 2 + 10,
                dummy_emb1[:, (cutoff + 1) :, :],
            ],
            axis=1,
        )
        # mask do not mask any tokens
        dummy_mask = jnp.ones(shape=(batch, seq_len), dtype=jnp.bool_)
        out1, _ = transformer(
            x=dummy_emb1,
            mask=dummy_mask,
        )
        out2, _ = transformer(
            x=dummy_emb2,
            mask=dummy_mask,
        )
        # the first few tokens are not impacted
        chex.assert_trees_all_equal(out1[:, :cutoff, :], out2[:, :cutoff, :])
        # each of the following tokens are impacted
        # (batch, seq_len-cutoff)
        diff = jnp.linalg.norm(
            out1[:, cutoff:, :] - out2[:, cutoff:, :], axis=2
        )
        assert jnp.sum(diff > 0).item() == (seq_len - cutoff) * batch
