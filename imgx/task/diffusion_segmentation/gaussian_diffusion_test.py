"""Test Gaussian diffusion related classes and functions."""


import chex
from chex._src import fake

from imgx.task.diffusion_segmentation.gaussian_diffusion import GaussianDiffusionSegmentation


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule() -> None:  # pylint: disable=invalid-name
    """Fake multi-devices."""
    fake.set_n_cpu_devices(2)


class TestGaussianDiffusionSegmentation(chex.TestCase):
    """Test the class GaussianDiffusion."""

    batch_size = 2

    # unet
    in_channels = 1
    num_classes = 2
    num_channels = (1, 2)

    num_timesteps = 5
    num_timesteps_beta = 1001
    beta_schedule = "linear"
    beta_start = 0.0001
    beta_end = 0.02

    def test_attributes(
        self,
    ) -> None:
        """Test attribute shape."""
        gd = GaussianDiffusionSegmentation.create(
            classes_are_exclusive=True,
            num_timesteps=self.num_timesteps,
            num_timesteps_beta=self.num_timesteps_beta,
            beta_schedule=self.beta_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            model_out_type="x_start",
            model_var_type="fixed_large",
        )

        chex.assert_shape(gd.betas, (self.num_timesteps,))
        chex.assert_shape(gd.alphas_cumprod, (self.num_timesteps,))
        chex.assert_shape(gd.alphas_cumprod_prev, (self.num_timesteps,))
        chex.assert_shape(gd.alphas_cumprod_next, (self.num_timesteps,))
        chex.assert_shape(gd.sqrt_alphas_cumprod, (self.num_timesteps,))
        chex.assert_shape(gd.sqrt_one_minus_alphas_cumprod, (self.num_timesteps,))
        chex.assert_shape(gd.log_one_minus_alphas_cumprod, (self.num_timesteps,))
        chex.assert_shape(gd.sqrt_recip_alphas_cumprod, (self.num_timesteps,))
        chex.assert_shape(gd.sqrt_recip_alphas_cumprod_minus_one, (self.num_timesteps,))
        chex.assert_shape(gd.posterior_mean_coeff_start, (self.num_timesteps,))
        chex.assert_shape(gd.posterior_mean_coeff_t, (self.num_timesteps,))
        chex.assert_shape(gd.posterior_variance, (self.num_timesteps,))
        chex.assert_shape(gd.posterior_log_variance_clipped, (self.num_timesteps,))
