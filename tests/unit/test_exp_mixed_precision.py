"""Test mixed precision related functions in factory."""
import haiku as hk
import pytest

from imgx import model
from imgx.exp.mixed_precision import set_mixed_precision_policy
from imgx.model import MODEL_CLS_NAME_TO_CONFIG_NAME
from imgx.model import __all__ as all_model_classes


@pytest.mark.parametrize(
    "model_class",
    all_model_classes,
    ids=all_model_classes,
)
def test_set_mixed_precision_policy(model_class: str) -> None:
    """Test all supported models.

    Args:
        model_class: name of model class.
    """
    set_mixed_precision_policy(True, MODEL_CLS_NAME_TO_CONFIG_NAME[model_class])
    # clear policy, otherwise impact other tests
    hk.mixed_precision.clear_policy(hk.BatchNorm)
    hk.mixed_precision.clear_policy(hk.GroupNorm)
    hk.mixed_precision.clear_policy(hk.LayerNorm)
    hk.mixed_precision.clear_policy(hk.InstanceNorm)
    hk.mixed_precision.clear_policy(getattr(model, model_class))
