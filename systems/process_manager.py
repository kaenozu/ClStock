"""Thin facade composing the process management subsystems."""

from __future__ import annotations

import os  # re-exported for backward compatibility in tests
import psutil  # noqa: F401  # re-exported for test patches
import subprocess  # noqa: F401  # re-exported for test patches
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from .monitoring import MonitoringLoop
from .service_registry import (
    ProcessInfo,
    ProcessStatus,
    ServiceRegistry,
    logger,
    settings,
)
from .shutdown_coordinator import ShutdownCoordinator


class ProcessManager:
    """Facade coordinating service registry, monitoring, and shutdown."""

    def __init__(
        self,
        service_registry: Optional[ServiceRegistry] = None,
        monitoring_loop: Optional[MonitoringLoop] = None,
        shutdown_coordinator: Optional[ShutdownCoordinator] = None,
        *,
        install_signal_handlers: bool = True,
    ) -> None:
        self.service_registry = service_registry or ServiceRegistry()
        self.monitoring_loop = monitoring_loop or MonitoringLoop(self.service_registry)
        self.shutdown_coordinator = shutdown_coordinator or ShutdownCoordinator(
            self, self.service_registry, self.monitoring_loop
        )

        if install_signal_handlers and service_registry is None and monitoring_loop is None:
            self.shutdown_coordinator.install_signal_handlers()

    # ------------------------------------------------------------------
    # Properties exposing underlying state
    # ------------------------------------------------------------------
    @property
    def processes(self) -> Dict[str, ProcessInfo]:
        if hasattr(self, "service_registry"):
            return self.service_registry.processes
        return getattr(self, "_processes", {})

    @processes.setter
    def processes(self, value: Dict[str, ProcessInfo]) -> None:
        if hasattr(self, "service_registry"):
            self.service_registry.processes = value
        else:
            self._processes = value

    @property
    def monitoring_active(self) -> bool:
        if hasattr(self, "monitoring_loop"):
            return getattr(self.monitoring_loop, "monitoring_active", False)
        return getattr(self, "_monitoring_active", False)

    @monitoring_active.setter
    def monitoring_active(self, value: bool) -> None:
        if hasattr(self, "monitoring_loop"):
            self.monitoring_loop.monitoring_active = value
        else:
            self._monitoring_active = value

    @property
    def monitor_thread(self):
        if hasattr(self, "monitoring_loop"):
            return getattr(self.monitoring_loop, "monitor_thread", None)
        return getattr(self, "_monitor_thread", None)

    @monitor_thread.setter
    def monitor_thread(self, value) -> None:
        if hasattr(self, "monitoring_loop"):
            self.monitoring_loop.monitor_thread = value
        else:
            self._monitor_thread = value

    @property
    def _shutdown_event(self):  # type: ignore[override]
        if hasattr(self, "monitoring_loop"):
            return getattr(self.monitoring_loop, "shutdown_event", None)
        return getattr(self, "__shutdown_event", None)

    @_shutdown_event.setter
    def _shutdown_event(self, value) -> None:  # type: ignore[override]
        if hasattr(self, "monitoring_loop") and hasattr(
            self.monitoring_loop, "shutdown_event"
        ):
            self.monitoring_loop._shutdown_event = value  # type: ignore[attr-defined]
        else:
            self.__shutdown_event = value
            self._legacy_shutdown_event = value

    @property
    def _shutdown_lock(self):  # type: ignore[override]
        if hasattr(self, "shutdown_coordinator"):
            return getattr(self.shutdown_coordinator, "_shutdown_lock")
        return getattr(self, "__shutdown_lock")

    @_shutdown_lock.setter
    def _shutdown_lock(self, value) -> None:  # type: ignore[override]
        if hasattr(self, "shutdown_coordinator"):
            self.shutdown_coordinator._shutdown_lock = value  # type: ignore[attr-defined]
        else:
            self.__shutdown_lock = value
            self._legacy_shutdown_lock = value

    @property
    def _shutdown_thread(self):  # type: ignore[override]
        if hasattr(self, "shutdown_coordinator"):
            return getattr(self.shutdown_coordinator, "_shutdown_thread", None)
        return getattr(self, "__shutdown_thread", None)

    @_shutdown_thread.setter
    def _shutdown_thread(self, value) -> None:  # type: ignore[override]
        if hasattr(self, "shutdown_coordinator"):
            self.shutdown_coordinator._shutdown_thread = value  # type: ignore[attr-defined]
        else:
            self.__shutdown_thread = value
            self._legacy_shutdown_thread = value

    @property
    def _executor(self):  # type: ignore[override]
        if hasattr(self, "service_registry"):
            return getattr(self.service_registry, "_executor")
        return getattr(self, "__executor")

    @_executor.setter
    def _executor(self, value) -> None:  # type: ignore[override]
        if hasattr(self, "service_registry"):
            setattr(self.service_registry, "_executor", value)
        else:
            self.__executor = value
            self._legacy_executor = value

    # ------------------------------------------------------------------
    # Registry delegates
    # ------------------------------------------------------------------
    def register_service(self, process_info: ProcessInfo) -> bool:
        if process_info is None:
            raise ValueError("process_info must not be None")
        return self.service_registry.register_service(process_info)

    def start_service(self, name: str) -> bool:
        return self.service_registry.start_service(name)

    def stop_service(self, name: str, force: bool = False) -> bool:
        return self.service_registry.stop_service(name, force=force)

    def restart_service(self, name: str) -> bool:
        return self.service_registry.restart_service(name)

    def get_service_status(self, name: str) -> Optional[ProcessInfo]:
        return self.service_registry.get_service_status(name)

    def _check_resource_limits(self, process_info: ProcessInfo) -> bool:
        return self.service_registry._check_resource_limits(process_info)

    def list_services(self) -> List[ProcessInfo]:
        return self.service_registry.list_services()

    def start_multiple_services(
        self, names: Iterable[str], max_parallel: int = 3
    ) -> Dict[str, bool]:
        return self.service_registry.start_multiple_services(names, max_parallel)

    def stop_all_services(self, force: bool = False):
        return self.service_registry.stop_all_services(force=force)

    # ------------------------------------------------------------------
    # Monitoring delegates
    # ------------------------------------------------------------------
    def start_monitoring(self) -> None:
        self.monitoring_loop.start(self.service_registry)

    def stop_monitoring(self) -> None:
        self.monitoring_loop.stop()

    def wait_for_shutdown(self, timeout: int = 60) -> None:
        self.monitoring_loop.wait_for_shutdown(timeout)
        self.shutdown_coordinator.wait_for_shutdown(timeout)

    def _check_process_health(self, process_info: ProcessInfo) -> None:
        self.monitoring_loop.check_process_health(process_info)

    # ------------------------------------------------------------------
    # Shutdown delegates
    # ------------------------------------------------------------------
    def _signal_handler(self, signum: int) -> None:
        self.shutdown_coordinator.handle_signal(signum)

    def _graceful_shutdown(self) -> None:
        if hasattr(self, "shutdown_coordinator"):
            self.shutdown_coordinator._graceful_shutdown()  # pragma: no cover - invoked in tests
            return

        # Fallback path for test doubles that bypass initialization
        self.stop_all_services(force=True)
        executor = getattr(self, "__executor", getattr(self, "_legacy_executor", None))
        if executor is not None:
            executor.shutdown(wait=True)
        event = getattr(
            self,
            "__shutdown_event",
            getattr(self, "_legacy_shutdown_event", None),
        )
        if event is not None:
            try:
                event.clear()
            except AttributeError:
                pass
        os._exit(0)

    def shutdown(self, force: bool = False) -> None:
        self.shutdown_coordinator.shutdown(
            self.service_registry, self.monitoring_loop, force=force
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_system_status(self) -> Dict:
        running_count = sum(
            1 for p in self.processes.values() if p.status == ProcessStatus.RUNNING
        )
        failed_count = sum(
            1 for p in self.processes.values() if p.status == ProcessStatus.FAILED
        )
        return {
            "total_services": len(self.processes),
            "running": running_count,
            "failed": failed_count,
            "monitoring_active": self.monitoring_active,
            "timestamp": datetime.now(),
        }


process_manager = ProcessManager()


def get_process_manager() -> ProcessManager:
    return process_manager
