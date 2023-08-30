"""Class to hold arguments to calculate diffusion loss."""
import dataclasses

import jax.numpy as jnp


@dataclasses.dataclass
class DiffusionLossArgs:
    """Arguments to calculate diffusion loss."""

    t_index: jnp.ndarray
    x_t: jnp.ndarray
    noise: jnp.ndarray
    model_out: jnp.ndarray
