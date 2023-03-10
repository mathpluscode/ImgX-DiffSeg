"""Mixed precision related functions."""
from functools import partial

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jmp

from imgx import model
from imgx.model import CONFIG_NAME_TO_MODEL_CLS_NAME


def get_mixed_precision_policy(use_mp: bool) -> jmp.Policy:
    """Return general mixed precision policy.

    Args:
        use_mp: use mixed precision if True.

    Returns:
        Policy instance.
    """
    full = jnp.float32
    if not use_mp:
        return jmp.Policy(
            compute_dtype=full, param_dtype=full, output_dtype=full
        )
    half = jmp.half_dtype()
    return jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=full)


def get_mixed_precision_policy_for_normalization(use_mp: bool) -> jmp.Policy:
    """Return mixed precision policy for norms.

    Args:
        use_mp: use mixed precision if True.

    Returns:
        Policy instance.
    """
    full = jnp.float32
    if not use_mp:
        return jmp.Policy(
            compute_dtype=full, param_dtype=full, output_dtype=full
        )
    half = jmp.half_dtype()
    return jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=half)


def select_tree(
    pred: jnp.ndarray, a: chex.ArrayTree, b: chex.ArrayTree
) -> chex.ArrayTree:
    """Selects a pytree based on the given predicate.

    Replace jmp.select_tree as it used jax.tree_multimap
    which has been deprecated.

    Args:
      pred: bool array.
      a: values for true.
      b: values for false.

    Returns:
        Selected tree.

    Raises:
        ValueError: if pred dtype or shape is wrong.
    """
    if not (pred.ndim == 0 and pred.dtype == jnp.bool_):
        raise ValueError("expected boolean scalar")
    return jax.tree_map(partial(jax.lax.select, pred), a, b)


def set_mixed_precision_policy(use_mp: bool, model_name: str) -> None:
    """Set mixed precision policy for networks.

    Args:
        use_mp: use mixed precision if True.
        model_name: name of the model.
    """
    # assign mixed precision policies to modules
    # for norms, use the full precision for stability
    mp_policy = get_mixed_precision_policy(use_mp)
    mp_norm_policy = get_mixed_precision_policy_for_normalization(use_mp)

    # the order we call `set_policy` doesn't matter, when a method on a
    # class is called the policy for that class will be applied, or it will
    # inherit the policy from its parent module.
    hk.mixed_precision.set_policy(hk.BatchNorm, mp_norm_policy)
    hk.mixed_precision.set_policy(hk.GroupNorm, mp_norm_policy)
    hk.mixed_precision.set_policy(hk.LayerNorm, mp_norm_policy)
    hk.mixed_precision.set_policy(hk.InstanceNorm, mp_norm_policy)

    if model_name not in CONFIG_NAME_TO_MODEL_CLS_NAME:
        raise ValueError(f"Unknown model name {model_name}.")
    model_cls_name = CONFIG_NAME_TO_MODEL_CLS_NAME[model_name]
    hk.mixed_precision.set_policy(getattr(model, model_cls_name), mp_policy)
