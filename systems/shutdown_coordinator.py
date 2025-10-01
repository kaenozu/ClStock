"""Shutdown coordination for the process manager."""

from __future__ import annotations

import os
import signal
import threading
from typing import Optional, TYPE_CHECKING

from ClStock.utils.logger_config import get_logger

from .monitoring import MonitoringLoop
from .service_registry import ServiceRegistry

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .process_manager import ProcessManager


logger = get_logger(__name__)


class ShutdownCoordinator:
    """Coordinate graceful shutdown across subsystems."""

    def __init__(
        self,
        manager: "ProcessManager",
        service_registry: ServiceRegistry,
        monitoring_loop: MonitoringLoop,
    ) -> None:
        self.manager = manager
        self.service_registry = service_registry
        self.monitoring_loop = monitoring_loop
        self._shutdown_lock = threading.Lock()
        self._shutdown_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def handle_signal(self, signum: int) -> None:
        logger.info("シグナル受信: %s", signum)
        if not self._shutdown_lock.acquire(blocking=False):
            logger.info("シャットダウンは既に進行中")
            return

        try:
            if self.monitoring_loop.shutdown_event.is_set():
                logger.info("シャットダウンは既に進行中")
                return

            self.monitoring_loop.shutdown_event.set()
            self._shutdown_thread = threading.Thread(
                target=self._graceful_shutdown,
                daemon=True,
                name="ShutdownThread",
            )
            self._shutdown_thread.start()
        finally:
            self._shutdown_lock.release()

    def shutdown(
        self,
        service_registry: Optional[ServiceRegistry] = None,
        monitoring_loop: Optional[MonitoringLoop] = None,
        force: bool = False,
    ) -> None:
        if service_registry is not None:
            self.service_registry = service_registry
        if monitoring_loop is not None:
            self.monitoring_loop = monitoring_loop

        if force:
            self.monitoring_loop.shutdown_event.set()

        self.monitoring_loop.stop()
        self.manager.stop_all_services(force=force)
        self.service_registry.cleanup_executor()

    def wait_for_shutdown(self, timeout: int = 60) -> None:
        thread = self._shutdown_thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("シャットダウンスレッドの終了待機タイムアウト")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _graceful_shutdown(self) -> None:
        try:
            logger.info("グレースフルシャットダウン開始")
            self.manager.stop_all_services(force=True)
            self.service_registry.cleanup_executor()
            self.monitoring_loop.shutdown_event.clear()
            os._exit(0)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("シャットダウン中にエラー発生: %s", exc)
            os._exit(1)

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)