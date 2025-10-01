from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def restore_process_manager_singleton():
    """Ensure each test runs with a fresh process manager singleton."""
    # Import lazily to avoid circular imports during test collection
    from importlib import reload

    import systems.process_manager as process_manager_module

    reload(process_manager_module)
    yield
    reload(process_manager_module)


def test_resource_monitor_caches_system_metrics(monkeypatch):
    from systems.resource_monitor import ResourceMonitor

    cpu_calls = []
    mem_calls = []

    def fake_cpu_percent(*, interval=None):
        cpu_calls.append(interval)
        return 12.5

    def fake_virtual_memory():
        mem_calls.append(True)
        return SimpleNamespace(percent=34.2, total=8 * 1024 * 1024 * 1024)

    monkeypatch.setattr(
        "systems.resource_monitor.psutil.cpu_percent", fake_cpu_percent
    )
    monkeypatch.setattr(
        "systems.resource_monitor.psutil.virtual_memory", fake_virtual_memory
    )

    monitor = ResourceMonitor(cache_ttl=0.5)

    first_usage = monitor.get_system_usage()
    second_usage = monitor.get_system_usage()

    assert first_usage.cpu_percent == pytest.approx(12.5)
    assert second_usage.cpu_percent == pytest.approx(12.5)
    assert first_usage.memory_percent == pytest.approx(34.2)
    assert second_usage.memory_percent == pytest.approx(34.2)
    assert cpu_calls == [None]
    assert len(mem_calls) == 1


def test_process_manager_blocks_start_when_limits_exceeded(monkeypatch):
    from systems.process_manager import ProcessInfo, ProcessManager

    manager = ProcessManager()
    test_process = ProcessInfo(
        name="heavy_job",
        command="echo heavy",
        max_cpu_percent=40,
        max_memory_mb=500,
    )
    manager.register_service(test_process)

    fake_monitor = MagicMock()
    # Current system usage is already too high for additional load
    fake_monitor.get_system_usage.return_value = SimpleNamespace(
        cpu_percent=70.0,
        memory_percent=75.0,
        memory_total=16 * 1024 * 1024 * 1024,
    )

    monkeypatch.setattr(manager, "resource_monitor", fake_monitor)

    result = manager.start_service("heavy_job")

    assert result is False
    fake_monitor.get_system_usage.assert_called_once()
