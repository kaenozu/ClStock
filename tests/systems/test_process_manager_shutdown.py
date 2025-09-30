import os
import threading

import pytest

from systems.process_manager import ProcessManager


class DummyExecutor:
    def __init__(self):
        self.shutdown_calls = []

    def shutdown(self, wait=True):
        self.shutdown_calls.append(wait)


class DummyProcessManager(ProcessManager):
    def __init__(self, executor):
        # 必要な属性のみ初期化し、副作用を避ける
        self.processes = {}
        self.monitoring_active = False
        self.monitor_thread = None
        self._shutdown_event = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._shutdown_thread = None
        self._executor = executor
        self._resource_limit_lock = threading.Lock()
        self._current_cpu_usage = 0.0
        self._current_memory_usage = 0.0
        self._max_system_cpu_percent = 80
        self._max_system_memory_percent = 80
        self._executor_futures = set()
        self._executor_futures_lock = threading.Lock()

    def stop_all_services(self, force=False):
        self.stop_called = True
        return True


@pytest.mark.parametrize("force", [True, False])
def test_graceful_shutdown_completes_without_type_error(monkeypatch, force):
    executor = DummyExecutor()
    manager = DummyProcessManager(executor)

    exit_calls = []

    def fake_exit(code):
        exit_calls.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", fake_exit)

    manager.stop_called = False

    with pytest.raises(SystemExit) as excinfo:
        manager._graceful_shutdown()

    assert excinfo.value.code == 0
    assert exit_calls == [0]
    assert executor.shutdown_calls == [True]
    assert manager.stop_called is True
