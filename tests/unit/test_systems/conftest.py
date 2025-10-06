"""Configuration for systems tests."""

import pytest


@pytest.fixture
def mock_process_manager():
    """Mock process manager fixture."""
    from ClStock.systems.process_manager import ProcessManager

    return ProcessManager()
