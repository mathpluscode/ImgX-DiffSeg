"""Module for configuration related functions."""


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
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
        if isinstance(v, dict):
            items.update(flatten_dict(d=v, parent_key=new_key, sep=sep))
        else:
            items[new_key] = v
    return dict(items)
