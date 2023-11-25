"""Module to handle data."""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

AugmentationFn = Callable[[jax.Array, dict[str, jnp.ndarray]], dict[str, jnp.ndarray]]
