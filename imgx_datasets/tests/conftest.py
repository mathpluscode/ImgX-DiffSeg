"""Fixtures for tests."""
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    """Directory path containing fixture data.

    Returns:
        Folder path containing the data.
    """
    return Path(__file__).resolve().parent / "fixtures"
