"""Shared utility functions."""
from __future__ import annotations

from jax import numpy as jnp


def decode_uids(uids: jnp.ndarray) -> list[str]:
    """Decode uids.

    Args:
        uids: uids in bytes or int.

    Returns:
        decoded uids.
    """
    decoded = []
    for x in uids.tolist():
        if isinstance(x, bytes):
            decoded.append(x.decode("utf-8"))
        elif x == 0:
            # the batch was not complete, padded with zero
            decoded.append("")
        else:
            raise ValueError(f"uid {x} is not supported.")
    return decoded
