"""Test Gaussian diffusion related classes and functions."""


import chex
import haiku as hk
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from chex._src import fake

from imgx.diffusion.gaussian.gaussian_diffusion import GaussianDiffusion
from imgx.diffusion.gaussian.gaussian_diffusion_segmentation import (
    GaussianDiffusionSegmentation,
)
from imgx.diffusion.gaussian.sampler import (
    DDIMSegmentationSampler,
    DDPMSegmentationSampler,
)


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestGaussianDiffusion(chex.TestCase):
    """Test the class GaussianDiffusion."""

    batch_size = 2

    # unet
    in_channels = 1
    num_classes = 2
    num_channels = (1, 2)

    num_timesteps = 5
    num_timesteps_beta = 1001
    beta_schedule = "QUADRADIC"
    beta_start = 0.0001
    beta_end = 0.02

    @chex.variants(without_jit=True)
    def test_attributes(
        self,
    ) -> None:
        """Test attribute shape."""

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward() -> GaussianDiffusion:
            diffusion = GaussianDiffusion(
                model=hk.Module(),
                num_timesteps=self.num_timesteps,
                num_timesteps_beta=self.num_timesteps_beta,
                beta_schedule=self.beta_schedule,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                model_out_type="X_START",
                model_var_type="FIXED_LARGE",
            )
            return diffusion

        gd = forward()

        chex.assert_shape(gd.betas, (self.num_timesteps,))
        chex.assert_shape(gd.alphas_cumprod, (self.num_timesteps,))
        chex.assert_shape(gd.alphas_cumprod_prev, (self.num_timesteps,))
        chex.assert_shape(gd.alphas_cumprod_next, (self.num_timesteps,))
        chex.assert_shape(gd.sqrt_alphas_cumprod, (self.num_timesteps,))
        chex.assert_shape(
            gd.sqrt_one_minus_alphas_cumprod, (self.num_timesteps,)
        )
        chex.assert_shape(
            gd.log_one_minus_alphas_cumprod, (self.num_timesteps,)
        )
        chex.assert_shape(gd.sqrt_recip_alphas_cumprod, (self.num_timesteps,))
        chex.assert_shape(
            gd.sqrt_recip_alphas_cumprod_minus_one, (self.num_timesteps,)
        )
        chex.assert_shape(gd.posterior_mean_coeff_start, (self.num_timesteps,))
        chex.assert_shape(gd.posterior_mean_coeff_t, (self.num_timesteps,))
        chex.assert_shape(gd.posterior_variance, (self.num_timesteps,))
        chex.assert_shape(
            gd.posterior_log_variance_clipped, (self.num_timesteps,)
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        ("1d", (2,)),
        ("2d", (2, 3)),
        ("3d", (2, 3, 4)),
    )
    def test_q_mean_log_variance(
        self,
        in_shape: tuple[int, ...],
    ) -> None:
        """Test output shape.

        Args:
            in_shape: input shape.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            x_start: jnp.ndarray, t_index: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            diffusion = GaussianDiffusion(
                model=hk.Module(),
                num_timesteps=self.num_timesteps,
                num_timesteps_beta=self.num_timesteps_beta,
                beta_schedule=self.beta_schedule,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                model_out_type="X_START",
                model_var_type="FIXED_LARGE",
            )
            return diffusion.q_mean_log_variance(
                x_start=x_start, t_index=t_index
            )

        rng = jax.random.PRNGKey(0)
        rng_start, rng_t = jax.random.split(rng, num=2)
        dummy_x_start = jax.random.uniform(
            rng_start, shape=(self.batch_size, *in_shape)
        )
        dummy_t_index = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=0, maxval=self.num_timesteps
        )
        got_mean, got_log_var = forward(
            x_start=dummy_x_start, t_index=dummy_t_index
        )
        expanded_shape = (dummy_x_start.shape[0],) + (1,) * (
            dummy_x_start.ndim - 1
        )
        chex.assert_shape(got_mean, dummy_x_start.shape)
        chex.assert_shape(got_log_var, expanded_shape)

    @chex.all_variants()
    @parameterized.named_parameters(
        ("1d", (2,)),
        ("2d", (2, 3)),
        ("3d", (2, 3, 4)),
    )
    def test_q_sample(
        self,
        in_shape: tuple[int, ...],
    ) -> None:
        """Test output shape.

        Args:
            in_shape: input shape.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            x_start: jnp.ndarray, noise: jnp.ndarray, t_index: jnp.ndarray
        ) -> jnp.ndarray:
            diffusion = GaussianDiffusion(
                model=hk.Module(),
                num_timesteps=self.num_timesteps,
                num_timesteps_beta=self.num_timesteps_beta,
                beta_schedule=self.beta_schedule,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                model_out_type="X_START",
                model_var_type="FIXED_LARGE",
            )
            return diffusion.q_sample(
                x_start=x_start, noise=noise, t_index=t_index
            )

        rng = jax.random.PRNGKey(0)
        rng_start, rng_noise, rng_t = jax.random.split(rng, num=3)
        dummy_x_start = jax.random.uniform(
            rng_start, shape=(self.batch_size, *in_shape)
        )
        dummy_noise = jax.random.uniform(
            rng_noise, shape=(self.batch_size, *in_shape)
        )
        dummy_t_index = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=0, maxval=self.num_timesteps
        )
        got = forward(
            x_start=dummy_x_start, noise=dummy_noise, t_index=dummy_t_index
        )
        chex.assert_shape(got, dummy_x_start.shape)

    @chex.all_variants()
    @parameterized.named_parameters(
        ("1d", (2,)),
        ("2d", (2, 3)),
        ("3d", (2, 3, 4)),
    )
    def test_q_posterior_mean_variance(
        self,
        in_shape: tuple[int, ...],
    ) -> None:
        """Test output shape.

        Args:
            in_shape: input shape.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            x_start: jnp.ndarray, x_t: jnp.ndarray, t_index: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            diffusion = GaussianDiffusion(
                model=hk.Module(),
                num_timesteps=self.num_timesteps,
                num_timesteps_beta=self.num_timesteps_beta,
                beta_schedule=self.beta_schedule,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                model_out_type="X_START",
                model_var_type="FIXED_LARGE",
            )
            return diffusion.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t_index=t_index
            )

        rng_start = jax.random.PRNGKey(0)
        rng_start, rng_x_t, rng_t = jax.random.split(rng_start, num=3)
        dummy_x_start = jax.random.uniform(
            rng_start, shape=(self.batch_size, *in_shape)
        )
        dummy_x_t = jax.random.uniform(
            rng_x_t, shape=(self.batch_size, *in_shape)
        )
        dummy_t_index = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=0, maxval=self.num_timesteps
        )
        got_mean, got_log_var = forward(
            x_start=dummy_x_start, x_t=dummy_x_t, t_index=dummy_t_index
        )
        expanded_shape = (dummy_x_start.shape[0],) + (1,) * (
            dummy_x_start.ndim - 1
        )
        chex.assert_shape(got_mean, dummy_x_start.shape)
        chex.assert_shape(got_log_var, expanded_shape)

    @chex.all_variants()
    @parameterized.product(
        in_shape=[
            (2,),
            (2, 3),
            (2, 3, 4),
        ],
        t_per_class=[
            True,
            False,
        ],
        model_out_type=[
            "X_START",
            "NOISE",
        ],
        model_var_type=[
            "FIXED_SMALL",
            "FIXED_LARGE",
            "LEARNED",
            "LEARNED_RANGE",
        ],
    )
    def test_p_mean_variance(
        self,
        in_shape: tuple[int, ...],
        t_per_class: bool,
        model_out_type: str,
        model_var_type: str,
    ) -> None:
        """Test output shape.

        Args:
            in_shape: input shape.
            t_per_class: sample timesteps per class.
            model_out_type: define model output meaning.
            model_var_type: define p(x_{t-1} | x_t) variance.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            model_out: jnp.ndarray, x_t: jnp.ndarray, t_index: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            diffusion = GaussianDiffusion(
                model=hk.Module(),
                num_timesteps=self.num_timesteps,
                num_timesteps_beta=self.num_timesteps_beta,
                beta_schedule=self.beta_schedule,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                model_out_type=model_out_type,
                model_var_type=model_var_type,
            )
            return diffusion.p_mean_variance(
                model_out=model_out, x_t=x_t, t_index=t_index
            )

        rng_out = jax.random.PRNGKey(0)
        rng_out, rng_x_t, rng_t = jax.random.split(rng_out, num=3)
        num_out_channels = self.num_classes
        if model_var_type in [
            "LEARNED",
            "LEARNED_RANGE",
        ]:
            num_out_channels *= 2
        model_out_shape = (self.batch_size, *in_shape, num_out_channels)
        dummy_model_out = jax.random.uniform(
            rng_out,
            shape=model_out_shape,
        )
        dummy_x_t = jax.random.uniform(
            rng_x_t, shape=(self.batch_size, *in_shape, self.num_classes)
        )
        # for t = 0, x_prev is not well-defined
        if t_per_class:
            t_shape = (
                (self.batch_size,) + (1,) * len(in_shape) + (self.num_classes,)
            )
            dummy_t_index = jax.random.randint(
                rng_t, shape=t_shape, minval=1, maxval=self.num_timesteps
            )
        else:
            dummy_t_index = jax.random.randint(
                rng_t,
                shape=(self.batch_size,),
                minval=1,
                maxval=self.num_timesteps,
            )
        (
            got_x_start,
            got_model_mean,
            got_model_log_variance,
        ) = forward(
            model_out=dummy_model_out, x_t=dummy_x_t, t_index=dummy_t_index
        )
        if t_per_class:
            expanded_shape = (
                (dummy_x_t.shape[0],)
                + (1,) * len(in_shape)
                + (self.num_classes,)
            )
        else:
            expanded_shape = (dummy_x_t.shape[0],) + (1,) * (dummy_x_t.ndim - 1)
        assert (~jnp.isnan(got_x_start)).all()
        chex.assert_shape(got_x_start, dummy_x_t.shape)
        chex.assert_shape(got_model_mean, dummy_x_t.shape)
        if model_var_type in [
            "FIXED_SMALL",
            "FIXED_LARGE",
        ]:
            # variances are extended
            chex.assert_shape(got_model_log_variance, expanded_shape)
        else:
            chex.assert_shape(got_model_log_variance, dummy_x_t.shape)

    @chex.all_variants()
    @parameterized.named_parameters(
        ("1d", (2,), "FIXED_LARGE"),
        ("2d", (2, 3), "FIXED_LARGE"),
        ("3d", (2, 3, 4), "FIXED_LARGE"),
        ("3d", (2, 3, 8), "LEARNED"),
        ("3d", (2, 3, 8), "LEARNED_RANGE"),
    )
    def test_variational_lower_bound(
        self,
        model_out_shape: tuple[int, ...],
        model_var_type: str,
    ) -> None:
        """Test output shape.

        Args:
            model_out_shape: input shape.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            model_out: jnp.ndarray,
            x_start: jnp.ndarray,
            x_t: jnp.ndarray,
            t_index: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            diffusion = GaussianDiffusion(
                model=hk.Module(),
                num_timesteps=self.num_timesteps,
                num_timesteps_beta=self.num_timesteps_beta,
                beta_schedule=self.beta_schedule,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                model_out_type="X_START",
                model_var_type=model_var_type,
            )
            vlb_loss_batch, model_out = diffusion.variational_lower_bound(
                model_out=model_out, x_start=x_start, x_t=x_t, t_index=t_index
            )
            return vlb_loss_batch, model_out

        x_shape = model_out_shape
        if model_var_type in [
            "LEARNED",
            "LEARNED_RANGE",
        ]:
            # model_out is split into mean and variance
            x_shape = (*model_out_shape[:-1], model_out_shape[-1] // 2)
        model_out_shape = (self.batch_size, *model_out_shape)
        x_shape = (self.batch_size, *x_shape)

        rng_out = jax.random.PRNGKey(0)
        rng_out, rng_x_start, rng_x_t, rng_t = jax.random.split(rng_out, num=4)

        dummy_model_out = jax.random.uniform(
            rng_out,
            shape=model_out_shape,
        )
        dummy_x_start = jax.random.uniform(
            rng_x_start, shape=(self.batch_size, *x_shape)
        )
        dummy_x_t = jax.random.uniform(
            rng_x_t, shape=(self.batch_size, *x_shape)
        )
        dummy_t_index = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=0, maxval=self.num_timesteps
        )
        got, got_model_out = forward(
            model_out=dummy_model_out,
            x_start=dummy_x_start,
            x_t=dummy_x_t,
            t_index=dummy_t_index,
        )
        chex.assert_shape(got, (self.batch_size,))
        chex.assert_shape(got_model_out, x_shape)

    @chex.all_variants()
    @parameterized.product(
        in_shape=[
            (2,),
            (2, 3),
            (2, 3, 4),
        ],
        t_per_class=[
            True,
            False,
        ],
        sampler_class=[
            DDPMSegmentationSampler,
            DDIMSegmentationSampler,
        ],
    )
    def test_sample(
        self,
        in_shape: tuple[int, ...],
        t_per_class: bool,
        sampler_class: GaussianDiffusionSegmentation,
    ) -> None:
        """Test output shape.

        Args:
            in_shape: input shape.
            t_per_class: sample timesteps per class.
            sampler_class: sampler class.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            model_out: jnp.ndarray, x_t: jnp.ndarray, t_index: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            diffusion = sampler_class(
                model=hk.Module(),
                num_timesteps=self.num_timesteps,
                num_timesteps_beta=self.num_timesteps_beta,
                beta_schedule=self.beta_schedule,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                model_out_type="X_START",
                model_var_type="FIXED_LARGE",
                # sampler specific arguments
                # TODO does not test num_inference_timesteps != num_timesteps
                num_inference_timesteps=self.num_timesteps,
                classes_are_exclusive=True,
            )
            return diffusion.sample(
                model_out=model_out, x_t=x_t, t_index=t_index
            )

        rng_out = jax.random.PRNGKey(0)
        rng_out, rng_x_t, rng_t = jax.random.split(rng_out, num=3)
        model_out_shape = (self.batch_size, *in_shape)
        dummy_model_out = jax.random.uniform(
            rng_out,
            shape=model_out_shape,
        )
        dummy_x_t = jax.random.uniform(
            rng_x_t, shape=(self.batch_size, *in_shape)
        )
        if t_per_class:
            t_shape = (self.batch_size,) + (1,) * len(in_shape)
            dummy_t_index = jax.random.randint(
                rng_t, shape=t_shape, minval=0, maxval=self.num_timesteps
            )
        else:
            dummy_t_index = jax.random.randint(
                rng_t,
                shape=(self.batch_size,),
                minval=0,
                maxval=self.num_timesteps,
            )
        got_sample, got_x_start_pred = forward(
            model_out=dummy_model_out, x_t=dummy_x_t, t_index=dummy_t_index
        )
        chex.assert_shape(got_sample, dummy_x_t.shape)
        chex.assert_shape(got_x_start_pred, dummy_x_t.shape)
