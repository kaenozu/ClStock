import types
import sys

import pytest

from utils import context_managers


def test_final_cleanup_does_not_log_after_logging_shutdown(monkeypatch):
    events = []
    shutdown_marker = {"called": False}

    monkeypatch.setattr(context_managers.signal, "signal", lambda *args, **kwargs: None)
    monkeypatch.setattr(context_managers.atexit, "register", lambda func: None)

    class FakeLogger:
        def info(self, message):
            if shutdown_marker["called"]:
                raise AssertionError("Logging occurred after shutdown")
            events.append(("info", message))

        def warning(self, message):
            events.append(("warning", message))

        def error(self, message):
            events.append(("error", message))

    fake_logger = FakeLogger()
    monkeypatch.setattr(context_managers, "logger", fake_logger)

    def fake_logging_shutdown():
        shutdown_marker["called"] = True
        events.append(("logging.shutdown", None))

    monkeypatch.setattr(context_managers.logging, "shutdown", fake_logging_shutdown)

    fake_cache_module = types.SimpleNamespace(
        shutdown_cache=lambda: events.append(("cache", "shutdown"))
    )
    monkeypatch.setitem(sys.modules, "utils.cache", fake_cache_module)

    manager = context_managers.GracefulShutdownManager()
    manager._final_cleanup()

    assert shutdown_marker["called"] is True
    assert ("logging.shutdown", None) in events
