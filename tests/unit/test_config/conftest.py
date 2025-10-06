"""Configuration for config tests."""

import os

import pytest


@pytest.fixture
def clean_env():
    """Provide a clean environment for testing."""
    # Store original environment
    original_env = os.environ.copy()
    # Clear relevant environment variables
    for key in list(os.environ.keys()):
        if key.startswith("CLSTOCK_"):
            del os.environ[key]
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
