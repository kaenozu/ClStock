import types

import pytest

from systems import monitoring
from systems.service_registry import ProcessInfo, ProcessStatus, ServiceRegistry


class DummyProcess:
    def poll(self):
        return None


def test_check_process_health_updates_usage(monkeypatch):
    registry = ServiceRegistry()
    monitoring_loop = monitoring.MonitoringLoop(registry)

    process_info = ProcessInfo(name="dummy", command="echo")
    process_info.status = ProcessStatus.RUNNING
    process_info.pid = 123
    process_info.process = DummyProcess()

    class FakePsProcess:
        def __init__(self, pid):
            self.pid = pid

        def memory_info(self):
            return types.SimpleNamespace(rss=200 * 1024 * 1024)

        def cpu_percent(self):
            return 12.5

    def fake_process_factory(pid):
        return FakePsProcess(pid)

    monkeypatch.setattr(monitoring.psutil, "Process", fake_process_factory)

    monitoring_loop.check_process_health(process_info)

    assert process_info.memory_usage == pytest.approx(200.0)
    assert process_info.cpu_usage == pytest.approx(12.5)
