"""Test Transformer related classes and functions."""


import chex
import jax
import jax.numpy as jnp
import optax
from absl.testing import parameterized
from chex._src import fake

from imgx.model.attention.transformer import TransformerEncoder


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestTransformerEncoder(chex.TestCase):
    """Test TransformerEncoder."""

    num_heads: int = 2
    num_layers: int = 3
    widening_factor: int = 4

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "encoder-with-pe remat",
            (2, 3, 8),
            False,
            True,
            True,
        ),
        (
            "encoder-with-pe",
            (2, 3, 8),
            False,
            True,
            False,
        ),
        (
            "encoder-multi-dim-with-pe",
            (2, 3, 4, 8),
            False,
            True,
            False,
        ),
        (
            "encoder-without-pe",
            (2, 3, 8),
            False,
            False,
            True,
        ),
        (
            "decoder",
            (2, 3, 8),
            True,
            True,
            True,
        ),
    )
    def test_shapes(
        self,
        in_shape: tuple[int, int, int],
        autoregressive: bool,
        add_position_embedding: bool,
        remat: bool,
    ) -> None:
        """Test TransformerEncoder output shapes.

        Args:
            in_shape: input tensor shape (batch, seq_len, emb_dim).
            autoregressive: True for decoder, False for encoder.
            add_position_embedding: use positional embedding or not.
            remat: remat networks or not.
        """
        rng = {"params": jax.random.PRNGKey(0)}
        transformer = TransformerEncoder(
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            autoregressive=autoregressive,
            widening_factor=self.widening_factor,
            add_position_embedding=add_position_embedding,
            remat=remat,
        )
        emb = jnp.ones(in_shape)
        mask = jnp.ones(in_shape[:-1], dtype=jnp.bool_)
        is_train = True
        out, _ = self.variant(transformer.init_with_output, static_argnums=(1,))(
            rng, is_train, emb, mask
        )
        chex.assert_shape(out, in_shape)

    @chex.all_variants()
    @parameterized.product(
        dropout=[0.0, 0.1],
        is_train=[True, False],
    )
    def test_encoder_mask(
        self,
        dropout: float,
        is_train: bool,
    ) -> None:
        """Test TransformerEncoder encoder mask behaviour.

        Masked tokens should not impact other embeddings.

        Args:
            dropout: dropout rate.
            is_train: whether in training mode.
        """
        rng = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}
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
            dropout=dropout,
        )
        # build two embeddings, the first few tokens are different
        emb1 = jax.random.uniform(jax.random.PRNGKey(0), shape=(batch, seq_len, emb_dim))
        emb2 = jnp.concatenate(
            [
                emb1[:, :cutoff, :] * 2 + 10,
                emb1[:, cutoff:, :],
            ],
            axis=1,
        )
        # build a mask masking the first two tokens
        mask = jnp.concatenate(
            [
                jnp.zeros(shape=(batch, cutoff), dtype=jnp.bool_),
                jnp.ones(shape=(batch, seq_len - cutoff), dtype=jnp.bool_),
            ],
            axis=1,
        )  # (batch, seq_len)
        out1, _ = self.variant(transformer.init_with_output, static_argnums=(1,))(
            rng, is_train, emb1, mask
        )
        out2, _ = self.variant(transformer.init_with_output, static_argnums=(1,))(
            rng, is_train, emb2, mask
        )
        # the non-masked tokens are not impacted
        chex.assert_trees_all_equal(out1[:, cutoff:, :], out2[:, cutoff:, :])

    @chex.all_variants()
    def test_decoder(
        self,
    ) -> None:
        """Test TransformerEncoder decoder's autoregressive behaviour.

        If changing a certain tokens, only the following tokens are impacted.
        """
        rng = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}
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
        # build two embeddings, the token at index cutoff is different
        emb1 = jax.random.uniform(jax.random.PRNGKey(0), shape=(batch, seq_len, model_size))
        emb2 = jnp.concatenate(
            [
                emb1[:, :cutoff, :],
                emb1[:, cutoff : (cutoff + 1), :] * 2 + 10,
                emb1[:, (cutoff + 1) :, :],
            ],
            axis=1,
        )
        # mask do not mask any tokens
        mask = jnp.ones(shape=(batch, seq_len), dtype=jnp.bool_)
        is_train = True
        out1, _ = self.variant(transformer.init_with_output, static_argnums=(1,))(
            rng, is_train, emb1, mask
        )
        out2, _ = self.variant(transformer.init_with_output, static_argnums=(1,))(
            rng, is_train, emb2, mask
        )
        # the first few tokens are not impacted
        chex.assert_trees_all_equal(out1[:, :cutoff, :], out2[:, :cutoff, :])
        # each of the following tokens are impacted
        # (batch, seq_len-cutoff)
        diff = jnp.linalg.norm(out1[:, cutoff:, :] - out2[:, cutoff:, :], axis=2)
        assert jnp.sum(diff > 0).item() == (seq_len - cutoff) * batch

    @parameterized.named_parameters(
        (
            "encoder-with-pe",
            (2, 3, 8),
            False,
            True,
            2236,
            13.435797,
        ),
        (
            "encoder-without-pe",
            (2, 3, 8),
            False,
            False,
            2212,
            13.435564,
        ),
        (
            "decoder",
            (2, 3, 8),
            True,
            True,
            2236,
            13.435797,
        ),
    )
    def test_params_count(
        self,
        in_shape: tuple[int, int, int],
        autoregressive: bool,
        add_position_embedding: bool,
        expected_params_count: int,
        expected_params_norm: float,
    ) -> None:
        """Test TransformerEncoder parameter numbers.

        Args:
            in_shape: input tensor shape (batch, seq_len, emb_dim).
            autoregressive: True for decoder, False for encoder.
            add_position_embedding: use positional embedding or not.
            expected_params_count: expected number of parameters.
            expected_params_norm: expected parameters norm.
        """
        rng = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}
        transformer = TransformerEncoder(
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            autoregressive=autoregressive,
            widening_factor=self.widening_factor,
            add_position_embedding=add_position_embedding,
            remat=True,
        )
        is_train = True
        emb = jnp.ones(in_shape)
        mask = jnp.ones(in_shape[:-1], dtype=jnp.bool_)
        _, variables = transformer.init_with_output(rng, is_train, emb, mask)

        params = variables["params"]
        got_params_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        assert got_params_count == expected_params_count

        got_params_norm = optax.global_norm(params)
        assert got_params_norm == expected_params_norm
