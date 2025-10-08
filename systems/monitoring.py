"""Monitoring loop for managed services."""

from __future__ import annotations

import threading
import time
from typing import Optional

import psutil

from ClStock.config.settings import get_settings
from ClStock.systems.resource_monitor import ResourceMonitor
from ClStock.utils.logger_config import get_logger

from .service_registry import ProcessInfo, ProcessStatus, ServiceRegistry

logger = get_logger(__name__)
settings = get_settings()


class MonitoringLoop:
    """Manage background monitoring of registered services."""

    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.monitoring_active: bool = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self.resource_monitor = ResourceMonitor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self, registry: Optional[ServiceRegistry] = None) -> None:
        if registry is not None:
            self.service_registry = registry

        if self.monitoring_active:
            return

        self._shutdown_event.clear()
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_processes,
            daemon=True,
            name="process-monitor",
        )
        self.monitor_thread.start()
        logger.info("プロセス監視開始")

    def stop(self) -> None:
        self.monitoring_active = False
        self._shutdown_event.set()

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=30)
            if self.monitor_thread.is_alive():
                logger.warning("監視スレッドの終了待機タイムアウト")

        self.monitor_thread = None
        logger.info("プロセス監視停止")

    def wait_for_shutdown(self, timeout: int = 60) -> None:
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=timeout)
            if self.monitor_thread.is_alive():
                logger.warning("監視スレッドの終了待機タイムアウト")

    # ------------------------------------------------------------------
    # Monitoring internals
    # ------------------------------------------------------------------
    def _adjust_process_priorities(self) -> None:
        try:
            # システム全体のリソース使用状況を取得
            system_usage = self.resource_monitor.get_system_usage()
            system_cpu_percent = system_usage.cpu_percent
            system_memory_percent = system_usage.memory_percent

            # 高負荷の場合、低優先度プロセスの制限を検討
            if system_cpu_percent > 80 or system_memory_percent > 80:
                logger.info(
                    f"高負荷検出: CPU {system_cpu_percent:.1f}%, メモリ {system_memory_percent:.1f}%",
                )

                # 優先度の低いプロセスを一時停止またはリソース制限を強化
                low_priority_processes = [
                    p
                    for p in self.service_registry.processes.values()
                    if p.status == ProcessStatus.RUNNING and p.priority < 5
                ]

                for proc_info in low_priority_processes:
                    logger.info(
                        f"低優先度プロセス {proc_info.name} にリソース制限を強化: CPU {proc_info.max_cpu_percent * 0.7:.1f}%, メモリ {proc_info.max_memory_mb * 0.7:.0f}MB",
                    )
                    # 実際にはプロセスの制限を変更するにはより高度な制御が必要ですが、ここではログのみ
                    proc_info.max_cpu_percent *= 0.7  # CPU制限を70%に縮小
                    proc_info.max_memory_mb *= 0.7  # メモリ制限を70%に縮小

            elif system_cpu_percent < 30 and system_memory_percent < 50:
                # 負荷が低い場合は制限を元に戻す
                normal_priority_processes = [
                    p
                    for p in self.service_registry.processes.values()
                    if p.status == ProcessStatus.RUNNING and p.priority < 5
                ]

                for proc_info in normal_priority_processes:
                    # 制限を元の設定に戻す
                    original_settings = settings.process  # 設定から元の値を取得
                    proc_info.max_cpu_percent = (
                        original_settings.max_cpu_percent_per_process
                        if hasattr(original_settings, "max_cpu_percent_per_process")
                        else 50
                    )
                    proc_info.max_memory_mb = (
                        original_settings.max_memory_per_process_mb
                        if hasattr(original_settings, "max_memory_per_process_mb")
                        else 1000
                    )

        except Exception as e:
            logger.error(f"プロセス優先度調整エラー: {e}")

    def _monitor_processes(self) -> None:
        while self.monitoring_active and not self._shutdown_event.is_set():
            try:
                for process_info in list(self.service_registry.processes.values()):
                    if process_info.status == ProcessStatus.RUNNING:
                        self.check_process_health(process_info)
                self._adjust_process_priorities()
                time.sleep(5)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("プロセス監視エラー: %s", exc)
                time.sleep(10)

    def check_process_health(self, process_info: ProcessInfo) -> None:
        try:
            if not process_info.process or process_info.process.poll() is not None:
                logger.warning("プロセス異常終了検出: %s", process_info.name)
                process_info.status = ProcessStatus.FAILED

                if (
                    process_info.auto_restart
                    and process_info.restart_count < process_info.max_restart_attempts
                ):
                    logger.info(
                        "自動再起動実行: %s (試行 %s)",
                        process_info.name,
                        process_info.restart_count + 1,
                    )
                    process_info.restart_count += 1
                    time.sleep(process_info.restart_delay)
                    self.service_registry.start_service(process_info.name)
                else:
                    logger.error("再起動制限超過: %s", process_info.name)

            if process_info.pid:
                try:
                    process = psutil.Process(process_info.pid)
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()

                    process_info.memory_usage = memory_mb
                    process_info.cpu_usage = cpu_percent

                    if memory_mb > process_info.max_memory_mb:
                        logger.warning(
                            "メモリ使用量超過: %s (%.1fMB > %sMB)",
                            process_info.name,
                            memory_mb,
                            process_info.max_memory_mb,
                        )
                        if memory_mb > process_info.max_memory_mb * 1.2:
                            logger.error(
                                "危険なメモリ使用量: %s (%.1fMB)",
                                process_info.name,
                                memory_mb,
                            )
                            self.service_registry.stop_service(
                                process_info.name,
                                force=True,
                            )

                    if cpu_percent > process_info.max_cpu_percent:
                        logger.warning(
                            "CPU使用率超過: %s (%.1f%% > %s%%)",
                            process_info.name,
                            cpu_percent,
                            process_info.max_cpu_percent,
                        )

                except psutil.NoSuchProcess:
                    logger.warning("プロセス消失: %s", process_info.name)
                    process_info.status = ProcessStatus.FAILED
                except psutil.AccessDenied:
                    logger.warning("プロセス情報アクセス不可: %s", process_info.name)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("ヘルスチェックエラー %s: %s", process_info.name, exc)

    # ------------------------------------------------------------------
    # Shutdown helpers
    # ------------------------------------------------------------------
    @property
    def shutdown_event(self) -> threading.Event:
        return self._shutdown_event
