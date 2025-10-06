from datetime import datetime, timedelta
from types import SimpleNamespace

import psutil
import pytest

from monitoring.system_monitor import (
    PerformanceAlert,
    ProcessMetrics,
    SystemMetrics,
    SystemMonitor,
)


class DummyThread:
    def __init__(self, target=None, args=None, daemon=None):
        self.target = target
        self.args = args or ()
        self.daemon = daemon
        self.started = False
        self.joined = False
        self.join_timeout = None

    def start(self):
        self.started = True

    def is_alive(self):
        return True

    def join(self, timeout=None):
        self.joined = True
        self.join_timeout = timeout


class DummyEvent:
    def __init__(self):
        self._is_set = False
        self.wait_calls = []

    def is_set(self):
        return self._is_set

    def set(self):
        self._is_set = True

    def clear(self):
        self._is_set = False

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        self._is_set = True


@pytest.fixture
def monitor(monkeypatch):
    monitor = SystemMonitor()
    dummy_thread_factory_calls = []

    def dummy_thread_factory(*args, **kwargs):
        thread = DummyThread(*args, **kwargs)
        dummy_thread_factory_calls.append(thread)
        return thread

    monkeypatch.setattr(
        "monitoring.system_monitor.threading.Thread", dummy_thread_factory,
    )
    monitor._shutdown_event = DummyEvent()
    monitor._thread_factory_calls = dummy_thread_factory_calls
    return monitor


def test_start_and_stop_monitoring_toggles_state(monkeypatch, monitor):
    monitor.start_monitoring(interval_seconds=1)

    assert monitor.monitoring_active is True
    assert not monitor._shutdown_event.is_set()
    assert len(monitor._thread_factory_calls) == 1
    created_thread = monitor._thread_factory_calls[0]
    assert created_thread.started is True

    # Second start should be ignored
    monitor.start_monitoring(interval_seconds=1)
    assert len(monitor._thread_factory_calls) == 1

    monitor.stop_monitoring()
    assert monitor.monitoring_active is False
    assert monitor._shutdown_event.is_set()
    assert created_thread.joined is True
    assert created_thread.join_timeout == 10


def test_monitoring_loop_collects_metrics_and_processes(monkeypatch, monitor):
    def fake_cpu_percent(interval):
        return 42.0

    def fake_virtual_memory():
        return SimpleNamespace(percent=55.0, available=4 * 1024 * 1024 * 1024)

    def fake_disk_usage(path):
        return SimpleNamespace(percent=70.0, free=200 * 1024 * 1024 * 1024)

    def fake_net_io_counters():
        return SimpleNamespace(
            bytes_sent=100 * 1024 * 1024, bytes_recv=200 * 1024 * 1024,
        )

    def fake_pids():
        return [1, 2, 3]

    class DummyProcess:
        def __init__(self, pid, name, cpu, memory):
            self.info = {
                "pid": pid,
                "name": name,
                "cpu_percent": cpu,
                "memory_percent": memory,
                "memory_info": SimpleNamespace(
                    rss=50 * 1024 * 1024, vms=100 * 1024 * 1024,
                ),
                "status": "running",
                "create_time": datetime.now().timestamp() - 3600,
                "num_threads": 4,
            }

        def num_fds(self):
            return 10

    def fake_process_iter(attrs):
        yield DummyProcess(1, "proc1", 12.0, 1.2)
        yield DummyProcess(2, "proc2", 25.0, 0.5)

    monkeypatch.setattr(
        "monitoring.system_monitor.psutil.cpu_percent", fake_cpu_percent,
    )
    monkeypatch.setattr(
        "monitoring.system_monitor.psutil.virtual_memory", fake_virtual_memory,
    )
    monkeypatch.setattr("monitoring.system_monitor.psutil.disk_usage", fake_disk_usage)
    monkeypatch.setattr(
        "monitoring.system_monitor.psutil.net_io_counters", fake_net_io_counters,
    )
    monkeypatch.setattr("monitoring.system_monitor.psutil.pids", fake_pids)
    monkeypatch.setattr(
        "monitoring.system_monitor.psutil.process_iter", fake_process_iter,
    )

    monitor.monitoring_active = True
    monitor._shutdown_event.clear()

    monitor._monitoring_loop(interval_seconds=1)

    assert len(monitor.system_metrics_history) == 1
    metrics = monitor.system_metrics_history[0]
    assert isinstance(metrics, SystemMetrics)
    assert metrics.cpu_percent == 42.0
    assert metrics.memory_percent == 55.0
    assert metrics.disk_usage_percent == 70.0
    assert metrics.network_sent_mb == pytest.approx(100)
    assert metrics.network_recv_mb == pytest.approx(200)
    assert metrics.active_processes == 3

    # Process metrics stored
    summary = monitor.get_process_summary()
    assert len(summary) == 2
    assert summary[0]["cpu_percent"] >= summary[1]["cpu_percent"]


def test_collect_system_metrics_handles_exceptions(monkeypatch, monitor):
    def raising(*args, **kwargs):  # pragma: no cover - helper
        raise RuntimeError("boom")

    monkeypatch.setattr("monitoring.system_monitor.psutil.cpu_percent", raising)
    monkeypatch.setattr("monitoring.system_monitor.psutil.virtual_memory", raising)
    monkeypatch.setattr("monitoring.system_monitor.psutil.disk_usage", raising)
    monkeypatch.setattr("monitoring.system_monitor.psutil.net_io_counters", raising)
    monkeypatch.setattr("monitoring.system_monitor.psutil.pids", raising)

    metrics = monitor._collect_system_metrics()

    assert isinstance(metrics, SystemMetrics)
    assert metrics.cpu_percent == 0
    assert metrics.memory_percent == 0
    assert metrics.disk_usage_percent == 0
    assert metrics.network_sent_mb == 0
    assert metrics.network_recv_mb == 0
    assert metrics.active_processes == 0


def test_collect_process_metrics_handles_failures(monkeypatch, monitor):
    class DummyPsutilProcess:
        def __init__(self):
            self.info = {
                "pid": 10,
                "name": "good",
                "cpu_percent": 5.0,
                "memory_percent": 0.5,
                "memory_info": SimpleNamespace(
                    rss=10 * 1024 * 1024, vms=20 * 1024 * 1024,
                ),
                "status": "sleeping",
                "create_time": datetime.now().timestamp(),
                "num_threads": 2,
            }

        def num_fds(self):
            return 5

    class AccessDeniedProcess(DummyPsutilProcess):
        def __init__(self):
            super().__init__()
            self.info["pid"] = 11
            self.info["name"] = "denied"

        def num_fds(self):
            raise psutil.AccessDenied()

    def fake_process_iter(attrs):
        yield DummyPsutilProcess()
        yield AccessDeniedProcess()
        raise RuntimeError("process failure")

    monkeypatch.setattr(
        "monitoring.system_monitor.psutil.process_iter", fake_process_iter,
    )

    monitor._collect_process_metrics()

    assert 10 in monitor.process_metrics_history
    assert monitor.process_metrics_history[10][-1].pid == 10


def test_check_system_health_generates_alerts():
    monitor = SystemMonitor()
    now = datetime.now()
    metrics = SystemMetrics(
        timestamp=now,
        cpu_percent=monitor.cpu_warning_threshold + 1,
        memory_percent=96.0,
        memory_available_mb=100.0,
        disk_usage_percent=monitor.disk_warning_threshold + 10,
        disk_free_gb=5.0,
        network_sent_mb=0,
        network_recv_mb=0,
        active_processes=1,
    )

    monitor._check_system_health(metrics)

    assert len(monitor.alerts) == 3
    categories = {alert.category for alert in monitor.alerts}
    assert categories == {"CPU", "MEMORY", "DISK"}
    assert monitor.total_alerts == 3

    monitor.alerts.clear()
    monitor.total_alerts = 0
    safe_metrics = SystemMetrics(
        timestamp=now,
        cpu_percent=monitor.cpu_warning_threshold - 10,
        memory_percent=50.0,
        memory_available_mb=500.0,
        disk_usage_percent=monitor.disk_warning_threshold - 10,
        disk_free_gb=50.0,
        network_sent_mb=0,
        network_recv_mb=0,
        active_processes=1,
    )

    monitor._check_system_health(safe_metrics)

    assert len(monitor.alerts) == 0
    assert monitor.total_alerts == 0


def test_status_and_reports(monkeypatch):
    monitor = SystemMonitor()
    now = datetime.now()

    metrics1 = SystemMetrics(
        timestamp=now - timedelta(minutes=5),
        cpu_percent=20,
        memory_percent=40,
        memory_available_mb=2000,
        disk_usage_percent=50,
        disk_free_gb=300,
        network_sent_mb=10,
        network_recv_mb=20,
        active_processes=50,
    )
    metrics2 = SystemMetrics(
        timestamp=now,
        cpu_percent=60,
        memory_percent=70,
        memory_available_mb=1500,
        disk_usage_percent=80,
        disk_free_gb=200,
        network_sent_mb=30,
        network_recv_mb=40,
        active_processes=60,
    )
    monitor.system_metrics_history.extend([metrics1, metrics2])

    proc_metric = ProcessMetrics(
        timestamp=now,
        pid=1,
        name="proc",
        cpu_percent=30,
        memory_percent=1.0,
        memory_rss_mb=200,
        memory_vms_mb=400,
        status="running",
        create_time=now - timedelta(hours=2),
        num_threads=3,
    )
    monitor.process_metrics_history[1].append(proc_metric)

    alert = PerformanceAlert(
        timestamp=now - timedelta(minutes=1),
        level="WARNING",
        category="CPU",
        message="High CPU",
    )
    monitor.alerts.append(alert)
    monitor.total_alerts = 1
    monitor.start_time = now - timedelta(hours=3)

    status = monitor.get_current_system_status()
    assert status["status"] == "warning"
    assert status["system"]["cpu_percent"] == 60
    assert status["alerts"]["total"] == 1
    assert status["alerts"]["recent"] == 1

    process_summary = monitor.get_process_summary()
    assert process_summary == [
        {
            "pid": 1,
            "name": "proc",
            "cpu_percent": 30,
            "memory_percent": 1.0,
            "memory_rss_mb": 200,
            "status": "running",
            "num_threads": 3,
            "uptime_hours": pytest.approx(2, rel=1e-2),
        },
    ]

    report = monitor.generate_performance_report()
    assert report["data_points"] == 2
    assert report["cpu_stats"]["max"] == 60
    assert report["memory_stats"]["avg"] == pytest.approx(55)
    assert report["total_alerts"] == 1
    assert report["top_processes"][0]["pid"] == 1

    # No data scenarios
    empty_monitor = SystemMonitor()
    assert empty_monitor.get_current_system_status() == {
        "status": "no_data",
        "message": "監視データなし",
    }
    assert empty_monitor.generate_performance_report() == {"error": "監視データなし"}

    # Performance trends time window behaviour
    old_metric = SystemMetrics(
        timestamp=now - timedelta(hours=5),
        cpu_percent=10,
        memory_percent=20,
        memory_available_mb=3000,
        disk_usage_percent=30,
        disk_free_gb=400,
        network_sent_mb=0,
        network_recv_mb=0,
        active_processes=10,
    )
    empty_monitor.system_metrics_history.append(old_metric)
    assert empty_monitor.get_performance_trends(hours=1) == {
        "cpu": [],
        "memory": [],
        "disk": [],
        "timestamps": [],
    }
