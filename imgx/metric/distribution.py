"""Metric functions for probability distributions."""
import jax.numpy as jnp


def normal_kl(
    p_mean: jnp.ndarray,
    p_log_variance: jnp.ndarray,
    q_mean: jnp.ndarray,
    q_log_variance: jnp.ndarray,
) -> jnp.ndarray:
    r"""Compute the KL divergence between two 1D normal distributions.

    KL[p||q] = \int p \log (p / q) dx

    Although the inputs are arrays, each value is considered independently.
    This function is not symmetric.

    Input array shapes should be broadcast-compatible.

    Args:
        p_mean: mean of distribution p.
        p_log_variance: log variance of distribution p.
        q_mean: mean of distribution q.
        q_log_variance: log variance of distribution q.

    Returns:
        KL divergence.
    """
    return 0.5 * (
        -1.0
        + q_log_variance
        - p_log_variance
        + jnp.exp(p_log_variance - q_log_variance)
        + ((p_mean - q_mean) ** 2) * jnp.exp(-q_log_variance)
    )


def approx_standard_normal_cdf(x: jnp.ndarray) -> jnp.ndarray:
    """Approximate cumulative distribution function of standard normal.

    if x ~ Normal(mean, var), then cdf(z) = p(x <= z)

    https://www.aimspress.com/article/doi/10.3934/math.2022648#b13
    https://www.jstor.org/stable/2346872

    Args:
        x: array of any shape with any float values.

    Returns:
        CDF estimation.
    """
    return 0.5 * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))


def discretized_gaussian_log_likelihood(
    x: jnp.ndarray,
    mean: jnp.ndarray,
    log_variance: jnp.ndarray,
    x_delta: float = 1.0 / 255.0,
    x_bound: float = 0.999,
) -> jnp.ndarray:
    """Log-likelihood of a normal distribution discretizing to an image.

    p(y=x) is approximated by p(y <= x+delta) - p(y <= x-delta).

    Args:
        x: target image, with value in [-1, 1].
        mean: normal distribution mean.
        log_variance: log of distribution variance.
        x_delta: discretization step, used to estimate probability.
        x_bound: values with abs > x_bound are calculated differently.

    Returns:
        Discretized log likelihood over 2*delta.
    """
    log_scales = 0.5 * log_variance
    centered_x = x - mean
    inv_stdv = jnp.exp(-log_scales)

    # let y be a variable
    # cdf(z+delta) = p(y <= z+delta)
    plus_in = inv_stdv * (centered_x + x_delta)  # z
    cdf_plus = approx_standard_normal_cdf(plus_in)
    # log( p(y <= z+delta) )
    log_cdf_plus = jnp.log(cdf_plus.clip(min=1e-12))

    # cdf(z-delta) = p(y <= z-delta)
    minus_in = inv_stdv * (centered_x - x_delta)
    cdf_minus = approx_standard_normal_cdf(minus_in)
    # log( 1-p(y <= z-delta) ) = log( p(y > z-delta) )
    log_one_minus_cdf_minus = jnp.log((1.0 - cdf_minus).clip(min=1e-12))

    # p(z-delta < y <= z+delta)
    cdf_delta = cdf_plus - cdf_minus
    log_cdf_delta = jnp.log(cdf_delta.clip(min=1e-12))

    # if x < -0.999, log( p(y <= z+delta) )
    # if x > 0.999, log( p(y > z-delta) )
    # if -0.999 <= x <= 0.999, log( p(z-delta < y <= z+delta) )
    return jnp.where(
        x < -x_bound,
        log_cdf_plus,
        jnp.where(x > x_bound, log_one_minus_cdf_minus, log_cdf_delta),
    )
