from unittest.mock import MagicMock

import pytest

from systems.process_manager import ProcessManager


@pytest.fixture
def components():
    registry = MagicMock()
    monitoring = MagicMock()
    monitoring.shutdown_event = MagicMock()
    shutdown = MagicMock()
    return registry, monitoring, shutdown


def test_process_manager_delegates_to_composed_components(components):
    registry, monitoring, shutdown = components

    manager = ProcessManager(
        service_registry=registry,
        monitoring_loop=monitoring,
        shutdown_coordinator=shutdown,
        install_signal_handlers=False,
    )

    manager.register_service("svc")
    registry.register_service.assert_called_once_with("svc")

    manager.start_service("svc")
    registry.start_service.assert_called_once_with("svc")

    manager.stop_service("svc", force=True)
    registry.stop_service.assert_called_once_with("svc", force=True)

    manager.start_monitoring()
    monitoring.start.assert_called_once_with(registry)

    manager.stop_monitoring()
    monitoring.stop.assert_called_once_with()

    manager.wait_for_shutdown()
    monitoring.wait_for_shutdown.assert_called_once_with(60)
    shutdown.wait_for_shutdown.assert_called_once_with(60)

    manager.shutdown(force=True)
    shutdown.shutdown.assert_called_once_with(registry, monitoring, force=True)
