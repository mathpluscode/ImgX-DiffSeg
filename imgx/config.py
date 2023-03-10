"""Module for configuration related functions."""

from typing import Dict


def flatten_dict(d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
    """Flat a nested dict.

    Args:
        d: dict to flat.
        parent_key: key of the parent.
        sep: separation string.

    Returns:
        flatten dict.
    """
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Dict):
            items.update(flatten_dict(d=v, parent_key=new_key, sep=sep))
        else:
            items[new_key] = v
    return dict(items)
