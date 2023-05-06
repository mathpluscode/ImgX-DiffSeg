"""Test Gaussian diffusion related classes and functions."""
from typing import Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from chex._src import fake

from imgx.diffusion.gaussian_diffusion import (
    DiffusionBetaSchedule,
    DiffusionModelOutputType,
    DiffusionModelVarianceType,
    DiffusionSpace,
    GaussianDiffusion,
    extract_and_expand,
)
from imgx.model import Unet3dTime


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestExtractAndExpand(chex.TestCase):
    """Test extract_and_expand."""

    @chex.variants(without_jit=True, with_device=True, without_device=True)
    @parameterized.named_parameters(
        (
            "1d",
            1,
        ),
        (
            "2d",
            2,
        ),
        (
            "3d",
            3,
        ),
    )
    def test_shapes(
        self,
        ndim: int,
    ) -> None:
        """Test output shape.

        Args:
            ndim: number of dimensions.
        """
        batch_size = 2
        betas = jnp.array([0, 0.2, 0.5, 1.0])
        num_timesteps = len(betas)
        rng = jax.random.PRNGKey(0)
        t = jax.random.randint(
            rng, shape=(batch_size,), minval=0, maxval=num_timesteps
        )
        got = self.variant(extract_and_expand)(arr=betas, t=t, ndim=ndim)
        expected_shape = (batch_size,) + (1,) * (ndim - 1)
        chex.assert_shape(got, expected_shape)


class TestGaussianDiffusion(chex.TestCase):
    """Test the class GaussianDiffusion."""

    batch_size = 2

    # unet
    in_channels = 1
    num_classes = 2
    num_channels = (1, 2)

    num_timesteps = 5
    num_timesteps_beta = 1001
    beta_schedule = DiffusionBetaSchedule.QUADRADIC
    beta_start = 0.0001
    beta_end = 0.02
    x_limit = 1.0
    use_ddim = False

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
                model_out_type=DiffusionModelOutputType.X_START,
                model_var_type=DiffusionModelVarianceType.FIXED_LARGE,
                x_limit=self.x_limit,
                x_space=DiffusionSpace.SCALED_PROBS,
                use_ddim=self.use_ddim,
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

    @chex.all_variants
    @parameterized.named_parameters(
        ("1d", (2,)),
        ("2d", (2, 3)),
        ("3d", (2, 3, 4)),
    )
    def test_q_mean_log_variance(
        self,
        in_shape: Tuple[int, ...],
    ) -> None:
        """Test output shape.

        Args:
            in_shape: input shape.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            x_start: jnp.ndarray, t: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            diffusion = GaussianDiffusion(
                model=hk.Module(),
                num_timesteps=self.num_timesteps,
                num_timesteps_beta=self.num_timesteps_beta,
                beta_schedule=self.beta_schedule,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                model_out_type=DiffusionModelOutputType.X_START,
                model_var_type=DiffusionModelVarianceType.FIXED_LARGE,
                x_limit=self.x_limit,
                x_space=DiffusionSpace.SCALED_PROBS,
                use_ddim=self.use_ddim,
            )
            return diffusion.q_mean_log_variance(x_start=x_start, t=t)

        rng = jax.random.PRNGKey(0)
        rng_start, rng_t = jax.random.split(rng, num=2)
        dummy_x_start = jax.random.uniform(
            rng_start, shape=(self.batch_size, *in_shape)
        )
        dummy_t = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=0, maxval=self.num_timesteps
        )
        got_mean, got_log_var = forward(x_start=dummy_x_start, t=dummy_t)
        expanded_shape = (dummy_x_start.shape[0],) + (1,) * (
            dummy_x_start.ndim - 1
        )
        chex.assert_shape(got_mean, dummy_x_start.shape)
        chex.assert_shape(got_log_var, expanded_shape)

    @chex.all_variants
    @parameterized.named_parameters(
        ("1d", (2,)),
        ("2d", (2, 3)),
        ("3d", (2, 3, 4)),
    )
    def test_q_sample(
        self,
        in_shape: Tuple[int, ...],
    ) -> None:
        """Test output shape.

        Args:
            in_shape: input shape.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            x_start: jnp.ndarray, noise: jnp.ndarray, t: jnp.ndarray
        ) -> jnp.ndarray:
            diffusion = GaussianDiffusion(
                model=hk.Module(),
                num_timesteps=self.num_timesteps,
                num_timesteps_beta=self.num_timesteps_beta,
                beta_schedule=self.beta_schedule,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                model_out_type=DiffusionModelOutputType.X_START,
                model_var_type=DiffusionModelVarianceType.FIXED_LARGE,
                x_limit=self.x_limit,
                x_space=DiffusionSpace.SCALED_PROBS,
                use_ddim=self.use_ddim,
            )
            return diffusion.q_sample(x_start=x_start, noise=noise, t=t)

        rng = jax.random.PRNGKey(0)
        rng_start, rng_noise, rng_t = jax.random.split(rng, num=3)
        dummy_x_start = jax.random.uniform(
            rng_start, shape=(self.batch_size, *in_shape)
        )
        dummy_noise = jax.random.uniform(
            rng_noise, shape=(self.batch_size, *in_shape)
        )
        dummy_t = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=0, maxval=self.num_timesteps
        )
        got = forward(x_start=dummy_x_start, noise=dummy_noise, t=dummy_t)
        chex.assert_shape(got, dummy_x_start.shape)

    @chex.all_variants
    @parameterized.named_parameters(
        ("1d", (2,)),
        ("2d", (2, 3)),
        ("3d", (2, 3, 4)),
    )
    def test_q_posterior_mean_variance(
        self,
        in_shape: Tuple[int, ...],
    ) -> None:
        """Test output shape.

        Args:
            in_shape: input shape.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            x_start: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            diffusion = GaussianDiffusion(
                model=hk.Module(),
                num_timesteps=self.num_timesteps,
                num_timesteps_beta=self.num_timesteps_beta,
                beta_schedule=self.beta_schedule,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                model_out_type=DiffusionModelOutputType.X_START,
                model_var_type=DiffusionModelVarianceType.FIXED_LARGE,
                x_limit=self.x_limit,
                x_space=DiffusionSpace.SCALED_PROBS,
                use_ddim=self.use_ddim,
            )
            return diffusion.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )

        rng_start = jax.random.PRNGKey(0)
        rng_start, rng_x_t, rng_t = jax.random.split(rng_start, num=3)
        dummy_x_start = jax.random.uniform(
            rng_start, shape=(self.batch_size, *in_shape)
        )
        dummy_x_t = jax.random.uniform(
            rng_x_t, shape=(self.batch_size, *in_shape)
        )
        dummy_t = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=0, maxval=self.num_timesteps
        )
        got_mean, got_log_var = forward(
            x_start=dummy_x_start, x_t=dummy_x_t, t=dummy_t
        )
        expanded_shape = (dummy_x_start.shape[0],) + (1,) * (
            dummy_x_start.ndim - 1
        )
        chex.assert_shape(got_mean, dummy_x_start.shape)
        chex.assert_shape(got_log_var, expanded_shape)

    @chex.all_variants
    @parameterized.product(
        in_shape=[
            (2,),
            (2, 3),
            (2, 3, 4),
        ],
        model_out_type=[
            DiffusionModelOutputType.X_START,
            DiffusionModelOutputType.X_PREVIOUS,
            DiffusionModelOutputType.EPSILON,
        ],
        model_var_type=[
            DiffusionModelVarianceType.FIXED_SMALL,
            DiffusionModelVarianceType.FIXED_LARGE,
            DiffusionModelVarianceType.LEARNED,
            DiffusionModelVarianceType.LEARNED_RANGE,
        ],
    )
    def test_p_mean_variance(
        self,
        in_shape: Tuple[int, ...],
        model_out_type: DiffusionModelOutputType,
        model_var_type: DiffusionModelVarianceType,
    ) -> None:
        """Test output shape.

        Args:
            in_shape: input shape.
            model_out_type: define model output meaning.
            model_var_type: define p(x_{t-1} | x_t) variance.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            model_out: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            diffusion = GaussianDiffusion(
                model=hk.Module(),
                num_timesteps=self.num_timesteps,
                num_timesteps_beta=self.num_timesteps_beta,
                beta_schedule=self.beta_schedule,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                model_out_type=model_out_type,
                model_var_type=model_var_type,
                x_limit=self.x_limit,
                x_space=DiffusionSpace.SCALED_PROBS,
                use_ddim=self.use_ddim,
            )
            return diffusion.p_mean_variance(model_out=model_out, x_t=x_t, t=t)

        rng_out = jax.random.PRNGKey(0)
        rng_out, rng_x_t, rng_t = jax.random.split(rng_out, num=3)
        num_out_channels = self.num_classes
        if model_var_type in [
            DiffusionModelVarianceType.LEARNED,
            DiffusionModelVarianceType.LEARNED_RANGE,
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
        dummy_t = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=1, maxval=self.num_timesteps
        )
        (
            got_x_start,
            got_model_mean,
            got_model_log_variance,
        ) = forward(model_out=dummy_model_out, x_t=dummy_x_t, t=dummy_t)
        expanded_shape = (dummy_x_t.shape[0],) + (1,) * (dummy_x_t.ndim - 1)
        assert (~jnp.isnan(got_x_start)).all()
        chex.assert_shape(got_x_start, dummy_x_t.shape)
        chex.assert_shape(got_model_mean, dummy_x_t.shape)
        if model_var_type in [
            DiffusionModelVarianceType.FIXED_SMALL,
            DiffusionModelVarianceType.FIXED_LARGE,
        ]:
            # variances are extended
            chex.assert_shape(got_model_log_variance, expanded_shape)
        else:
            chex.assert_shape(got_model_log_variance, dummy_x_t.shape)

        # check value range
        chex.assert_scalar_in(
            jnp.min(got_x_start).item(), -self.x_limit, self.x_limit
        )
        chex.assert_scalar_in(
            jnp.max(got_x_start).item(), -self.x_limit, self.x_limit
        )

    @chex.all_variants
    @parameterized.named_parameters(
        ("1d", (2,)),
        ("2d", (2, 3)),
        ("3d", (2, 3, 4)),
    )
    def test_p_sample(
        self,
        in_shape: Tuple[int, ...],
    ) -> None:
        """Test output shape.

        Args:
            in_shape: input shape.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            model_out: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            diffusion = GaussianDiffusion(
                model=hk.Module(),
                num_timesteps=self.num_timesteps,
                num_timesteps_beta=self.num_timesteps_beta,
                beta_schedule=self.beta_schedule,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                model_out_type=DiffusionModelOutputType.X_START,
                model_var_type=DiffusionModelVarianceType.FIXED_LARGE,
                x_limit=self.x_limit,
                x_space=DiffusionSpace.SCALED_PROBS,
                use_ddim=self.use_ddim,
            )
            return diffusion.p_sample(model_out=model_out, x_t=x_t, t=t)

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
        dummy_t = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=0, maxval=self.num_timesteps
        )
        got_sample, got_x_start_pred = forward(
            model_out=dummy_model_out, x_t=dummy_x_t, t=dummy_t
        )
        chex.assert_shape(got_sample, dummy_x_t.shape)
        chex.assert_shape(got_x_start_pred, dummy_x_t.shape)

        # check value range
        chex.assert_scalar_in(
            jnp.min(got_sample).item(), -self.x_limit, self.x_limit
        )
        chex.assert_scalar_in(
            jnp.max(got_sample).item(), -self.x_limit, self.x_limit
        )

    @chex.all_variants
    def test_p_sample_mask(
        self,
    ) -> None:
        """Test output shape."""

        in_shape = (2, 3, 4)

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            image: jnp.ndarray,
            x_t: jnp.ndarray,
        ) -> jnp.ndarray:
            """Forward function for p_sample_mask.

            Args:
                image: (batch, w, h, d, in_channels).
                x_t: (batch, w, h, d, num_classes).

            Returns:
                p_sample_mask output.
            """
            model = Unet3dTime(
                in_shape=in_shape,
                in_channels=self.in_channels + self.num_classes,
                out_channels=self.num_classes,
                num_channels=self.num_channels,
                num_timesteps=self.num_timesteps,
            )
            diffusion = GaussianDiffusion(
                model=model,
                num_timesteps=self.num_timesteps,
                num_timesteps_beta=self.num_timesteps_beta,
                beta_schedule=self.beta_schedule,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                model_out_type=DiffusionModelOutputType.X_START,
                model_var_type=DiffusionModelVarianceType.FIXED_LARGE,
                x_limit=self.x_limit,
                x_space=DiffusionSpace.SCALED_PROBS,
                use_ddim=self.use_ddim,
            )
            return diffusion.sample_mask(
                image=image,
                x_t=x_t,
            )

        rng_image = jax.random.PRNGKey(0)
        rng_image, rng_x_t = jax.random.split(rng_image)
        image_shape = (self.batch_size, *in_shape, self.in_channels)
        dummy_image = jax.random.uniform(
            rng_image,
            shape=image_shape,
        )
        dummy_x_t = jax.random.uniform(
            rng_x_t, shape=(self.batch_size, *in_shape, self.num_classes)
        )

        got_sample = forward(
            image=dummy_image,
            x_t=dummy_x_t,
        )
        chex.assert_shape(got_sample, dummy_x_t.shape)

    @chex.all_variants
    @parameterized.named_parameters(
        ("1d", (2,)),
        ("2d", (2, 3)),
        ("3d", (2, 3, 4)),
    )
    def test_variational_lower_bound(
        self,
        in_shape: Tuple[int, ...],
    ) -> None:
        """Test output shape.

        Args:
            in_shape: input shape.
        """

        @hk.testing.transform_and_run(jax_transform=self.variant)
        def forward(
            model_out: jnp.ndarray,
            x_start: jnp.ndarray,
            x_t: jnp.ndarray,
            t: jnp.ndarray,
        ) -> jnp.ndarray:
            diffusion = GaussianDiffusion(
                model=hk.Module(),
                num_timesteps=self.num_timesteps,
                num_timesteps_beta=self.num_timesteps_beta,
                beta_schedule=self.beta_schedule,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                model_out_type=DiffusionModelOutputType.X_START,
                model_var_type=DiffusionModelVarianceType.FIXED_LARGE,
                x_limit=self.x_limit,
                x_space=DiffusionSpace.SCALED_PROBS,
                use_ddim=self.use_ddim,
            )
            return diffusion.variational_lower_bound(
                model_out=model_out, x_start=x_start, x_t=x_t, t=t
            )

        rng_out = jax.random.PRNGKey(0)
        rng_out, rng_x_start, rng_x_t, rng_t = jax.random.split(rng_out, num=4)
        model_out_shape = (self.batch_size, *in_shape)
        dummy_model_out = jax.random.uniform(
            rng_out,
            shape=model_out_shape,
        )
        dummy_x_start = jax.random.uniform(
            rng_x_start, shape=(self.batch_size, *in_shape)
        )
        dummy_x_t = jax.random.uniform(
            rng_x_t, shape=(self.batch_size, *in_shape)
        )
        dummy_t = jax.random.randint(
            rng_t, shape=(self.batch_size,), minval=0, maxval=self.num_timesteps
        )
        got = forward(
            model_out=dummy_model_out,
            x_start=dummy_x_start,
            x_t=dummy_x_t,
            t=dummy_t,
        )
        chex.assert_shape(got, (self.batch_size,))
