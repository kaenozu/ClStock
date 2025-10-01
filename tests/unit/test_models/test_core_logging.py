"""Regression tests covering logging side effects in models.core."""

import importlib

import pytest


@pytest.mark.usefixtures("monkeypatch")
def test_core_reload_does_not_trigger_setup_logging(monkeypatch):
    """Reloading models.core should not mutate logging via setup_logging."""
    import models.core as core

    def _fail(*args, **kwargs):
        raise RuntimeError("setup_logging should not be called during models.core import")

    monkeypatch.setattr("utils.logger.setup_logging", _fail)

    # If models.core calls setup_logging during import, this will raise and fail the test.
    importlib.reload(core)
