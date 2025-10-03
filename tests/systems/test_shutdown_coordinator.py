import signal
import threading

import pytest

from systems.shutdown_coordinator import ShutdownCoordinator


class DummyManager:
    def __init__(self):
        self.stop_calls = []

    def stop_all_services(self, force: bool = False):
        self.stop_calls.append(force)


class DummyServiceRegistry:
    def __init__(self):
        self.cleanup_called = 0

    def cleanup_executor(self):
        self.cleanup_called += 1


class DummyMonitoringLoop:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


class FakeThread:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = False
        self.joined = False

    def start(self):
        self.started = True

    def join(self, timeout=None):  # pragma: no cover - behaviour verified via state
        self.joined = True

    def is_alive(self):
        return self.started and not self.joined


@pytest.fixture
def coordinator():
    return ShutdownCoordinator(
        manager=DummyManager(),
        service_registry=DummyServiceRegistry(),
        monitoring_loop=DummyMonitoringLoop(),
    )


def test_handle_signal_sets_event_and_starts_thread_once(monkeypatch, coordinator):
    created_threads = []

    def fake_thread_factory(*args, **kwargs):
        thread = FakeThread(*args, **kwargs)
        created_threads.append(thread)
        return thread

    monkeypatch.setattr("systems.shutdown_coordinator.threading.Thread", fake_thread_factory)

    coordinator.handle_signal(signum=signal.SIGTERM)
    assert coordinator.monitoring_loop.shutdown_event.is_set()
    assert len(created_threads) == 1
    assert created_threads[0].started

    coordinator.handle_signal(signum=signal.SIGTERM)
    assert len(created_threads) == 1


def test_wait_for_shutdown_joins_thread_and_second_signal_does_not_spawn_new_thread(monkeypatch, coordinator):
    created_threads = []

    def fake_thread_factory(*args, **kwargs):
        thread = FakeThread(*args, **kwargs)
        created_threads.append(thread)
        return thread

    monkeypatch.setattr("systems.shutdown_coordinator.threading.Thread", fake_thread_factory)

    coordinator.handle_signal(signum=signal.SIGINT)
    assert len(created_threads) == 1

    coordinator.wait_for_shutdown()
    assert created_threads[0].joined

    coordinator.monitoring_loop.shutdown_event.clear()
    coordinator.handle_signal(signum=signal.SIGINT)
    assert len(created_threads) == 1


def test_graceful_shutdown_calls_dependencies_and_exits(monkeypatch, coordinator):
    exit_codes = []
    monkeypatch.setattr("systems.shutdown_coordinator.os._exit", exit_codes.append)

    coordinator._graceful_shutdown()

    assert coordinator.manager.stop_calls == [True]
    assert coordinator.service_registry.cleanup_called == 1
    assert not coordinator.monitoring_loop.shutdown_event.is_set()
    assert exit_codes == [0]


def test_shutdown_calls_dependencies_in_order(monkeypatch):
    calls = []

    class RecordingMonitoringLoop:
        def __init__(self):
            self.shutdown_event = threading.Event()

        def stop(self):
            calls.append("monitoring_stop")

    class RecordingManager:
        def stop_all_services(self, force):
            calls.append(("manager_stop", force))

    class RecordingServiceRegistry:
        def cleanup_executor(self):
            calls.append("service_cleanup")

    loop = RecordingMonitoringLoop()
    coordinator = ShutdownCoordinator(
        manager=RecordingManager(),
        service_registry=RecordingServiceRegistry(),
        monitoring_loop=loop,
    )

    coordinator.shutdown(force=True)
    assert calls == ["monitoring_stop", ("manager_stop", True), "service_cleanup"]

    loop2 = RecordingMonitoringLoop()
    coordinator2 = ShutdownCoordinator(
        manager=RecordingManager(),
        service_registry=RecordingServiceRegistry(),
        monitoring_loop=loop2,
    )
    coordinator2.shutdown(force=False)
    assert not loop2.shutdown_event.is_set()


def test_install_signal_handlers_registers_expected_signals(monkeypatch, coordinator):
    registered = {}

    def fake_signal(signum, handler):
        registered[signum] = handler

    monkeypatch.setattr("systems.shutdown_coordinator.signal.signal", fake_signal)

    coordinator.install_signal_handlers()

    sigint_handler = registered[signal.SIGINT]
    sigterm_handler = registered[signal.SIGTERM]

    assert sigint_handler.__self__ is coordinator
    assert sigterm_handler.__self__ is coordinator
    assert sigint_handler.__func__ is coordinator.handle_signal.__func__
    assert sigterm_handler.__func__ is coordinator.handle_signal.__func__
