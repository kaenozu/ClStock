"""Thin facade composing the process management subsystems."""

from __future__ import annotations

import os  # re-exported for backward compatibility in tests
import psutil  # noqa: F401  # re-exported for test patches
import subprocess  # noqa: F401  # re-exported for test patches
from datetime import datetime
from typing import Dict, Iterable, List, Optional
import shlex # Added for start_service
import threading # Added for start_service
import time # Added for start_service
from enum import Enum # Added for start_service
from dataclasses import dataclass, field # Added for start_service
from pathlib import Path # Added for start_service
from typing import Union, Sequence # Added for start_service

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent.parent
import sys # Added for start_service
sys.path.append(str(PROJECT_ROOT))

from utils.logger_config import get_logger
from config.settings import get_settings

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
        """サービスの開始"""
        if name not in self.processes:
            logger.error(f"未登録のサービス: {name}")
            return False

        process_info = self.processes[name]

        if process_info.status == ProcessStatus.RUNNING:
            logger.warning(f"サービスは既に実行中: {name}")
            return True

        # リソース制限チェック
        if not self._check_resource_limits(process_info):
            logger.warning(f"リソース不足のためサービス {name} の起動を延期")
            return False

        try:
            process_info.status = ProcessStatus.STARTING
            logger.info(f"サービス開始: {name}")

            # 環境変数設定
            env = os.environ.copy()
            env.update(process_info.env_vars)

            process_settings = get_settings().process
            capture_output = bool(
                getattr(process_settings, "log_process_output", False)
            )

            popen_kwargs = {
                "cwd": process_info.working_dir,
                "env": env,
            }

            if capture_output:
                popen_kwargs.update(
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

            # プロセス開始
            command = process_info.command
            if isinstance(command, str):
                argv = shlex.split(command)
            elif isinstance(command, Sequence):
                argv = list(command)
            else:
                raise TypeError(
                    f"Unsupported command type for service {name}: {type(command)!r}"
                )

            process_info.process = subprocess.Popen(
                argv,
                **popen_kwargs,
            )

            process_info.pid = process_info.process.pid
            process_info.start_time = datetime.now()
            process_info.status = ProcessStatus.RUNNING
            process_info.last_error = None

            if capture_output:
                self._start_output_logging(process_info)
            else:
                process_info.stdout_thread = None
                process_info.stderr_thread = None

            logger.info(f"サービス開始完了: {name} (PID: {process_info.pid})")
            return True

        except FileNotFoundError as e:
            process_info.status = ProcessStatus.FAILED
            process_info.last_error = f"実行ファイルが見つかりません: {e}"
            logger.error(f"サービス開始失敗 {name}: 実行ファイルが見つかりません - {e}")
            return False
        except PermissionError as e:
            process_info.status = ProcessStatus.FAILED
            process_info.last_error = f"実行権限がありません: {e}"
            logger.error(f"サービス開始失敗 {name}: 実行権限エラー - {e}")
            return False
        except subprocess.SubprocessError as e:
            process_info.status = ProcessStatus.FAILED
            process_info.last_error = str(e)
            logger.error(f"サービス開始失敗 {name}: サブプロセスエラー - {e}")
            return False
        except Exception as e:
            process_info.status = ProcessStatus.FAILED
            process_info.last_error = str(e)
            logger.error(f"サービス開始失敗 {name}: 予期しないエラー - {e}")
            return False

    def _start_output_logging(self, process_info: ProcessInfo) -> None:
        """プロセス標準出力/標準エラーの非同期読み取りを開始"""

        process = process_info.process
        if not process:
            return

        def _consume_stream(stream, log_func, stream_name: str):
            if stream is None:
                return None

            def _reader():
                try:
                    for line in iter(stream.readline, ""):
                        if not line:
                            break
                        log_func(f"[{process_info.name}][{stream_name}] {line.rstrip()}")
                except Exception as e:
                    logger.debug(
                        f"{process_info.name} の {stream_name} ログ読み取り中にエラー: {e}"
                    )
                finally:
                    try:
                        stream.close()
                    except Exception:
                        pass

            thread = threading.Thread(
                target=_reader,
                daemon=True,
                name=f"{process_info.name}-{stream_name}-logger",
            )
            thread.start()
            return thread

        process_info.stdout_thread = _consume_stream(
            process.stdout, logger.info, "stdout"
        )
        process_info.stderr_thread = _consume_stream(
            process.stderr, logger.warning, "stderr"
        )

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

    def start_monitoring(self):
        """監視の開始"""
        if self.monitoring_active:
            return

        # 以前の監視停止時にセットされたシャットダウンフラグをリセット
        self._shutdown_event.clear()
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_processes, daemon=True
        )
        self.monitor_thread.start()
        logger.info("プロセス監視開始")

    def stop_monitoring(self):
        """監視の停止"""
        self.monitoring_active = False
        self._shutdown_event.set()

        if self.monitor_thread and self.monitor_thread.is_alive():
            # タイムアウト付きでスレッド終了を待機（30秒に延長）
            self.monitor_thread.join(timeout=30)
            if self.monitor_thread.is_alive():
                logger.warning("監視スレッドの終了待機タイムアウト")
                # デーモンスレッドなので強制終了は不要

        # 再起動時の状態を明示的に初期化
        self.monitor_thread = None

        logger.info("プロセス監視停止")

    def wait_for_shutdown(self, timeout: int = 60):
        """シャットダウン完了を待機"""
        if self._shutdown_thread and self._shutdown_thread.is_alive():
            self._shutdown_thread.join(timeout=timeout)
            if self._shutdown_thread.is_alive():
                logger.warning("シャットダウンスレッドの終了待機タイムアウト")

    def _adjust_process_priorities(self):
        """プロセス優先度の動的調整"""
        try:
            # システム全体のリソース使用状況を取得
            system_cpu_percent = psutil.cpu_percent(interval=1)
            system_memory_percent = psutil.virtual_memory().percent

            # 高負荷の場合、低優先度プロセスの制限を検討
            if system_cpu_percent > 80 or system_memory_percent > 80:
                logger.info(f"高負荷検出: CPU {system_cpu_percent:.1f}%, メモリ {system_memory_percent:.1f}%")
                
                # 優先度の低いプロセスを一時停止またはリソース制限を強化
                low_priority_processes = [
                    p for p in self.processes.values() 
                    if p.status == ProcessStatus.RUNNING and p.priority < 5
                ]
                
                for proc_info in low_priority_processes:
                    logger.info(f"低優先度プロセス {proc_info.name} にリソース制限を強化: CPU {proc_info.max_cpu_percent*0.7:.1f}%, メモリ {proc_info.max_memory_mb*0.7:.0f}MB")
                    # 実際にはプロセスの制限を変更するにはより高度な制御が必要ですが、ここではログのみ
                    proc_info.max_cpu_percent *= 0.7  # CPU制限を70%に縮小
                    proc_info.max_memory_mb *= 0.7    # メモリ制限を70%に縮小

            elif system_cpu_percent < 30 and system_memory_percent < 50:
                # 負荷が低い場合は制限を元に戻す
                normal_priority_processes = [
                    p for p in self.processes.values() 
                    if p.status == ProcessStatus.RUNNING and p.priority < 5
                ]
                
                for proc_info in normal_priority_processes:
                    # 制限を元の設定に戻す
                    original_settings = get_settings().process  # 設定から元の値を取得
                    proc_info.max_cpu_percent = original_settings.max_cpu_percent_per_process if hasattr(original_settings, 'max_cpu_percent_per_process') else 50
                    proc_info.max_memory_mb = original_settings.max_memory_per_process_mb if hasattr(original_settings, 'max_memory_per_process_mb') else 1000
                    
        except Exception as e:
            logger.error(f"プロセス優先度調整エラー: {e}")

    def _monitor_processes(self):
        """プロセス監視メインループ"""
        while self.monitoring_active and not self._shutdown_event.is_set():
            try:
                for name, process_info in self.processes.items():
                    if process_info.status == ProcessStatus.RUNNING:
                        self._check_process_health(process_info)

                # プロセス優先度の調整
                self._adjust_process_priorities()

                time.sleep(5)  # 5秒間隔で監視

            except Exception as e:
                logger.error(f"プロセス監視エラー: {e}")
                time.sleep(10)

    def _check_process_health(self, process_info: ProcessInfo):
        """プロセスの健全性チェック"""
        try:
            if not process_info.process or process_info.process.poll() is not None:
                # プロセスが停止している
                logger.warning(f"プロセス異常終了検出: {process_info.name}")
                process_info.status = ProcessStatus.FAILED

                # 自動再起動
                if (
                    process_info.auto_restart
                    and process_info.restart_count < process_info.max_restart_attempts
                ):
                    logger.info(
                        f"自動再起動実行: {process_info.name} (試行 {process_info.restart_count + 1})"
                    )
                    process_info.restart_count += 1

                    time.sleep(process_info.restart_delay)
                    self.start_service(process_info.name)
                else:
                    logger.error(f"再起動制限超過: {process_info.name}")

            # プロセスのリソース使用量チェック
            if process_info.pid:
                try:
                    process = psutil.Process(process_info.pid)
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()

                    # リソース使用量を更新
                    process_info.memory_usage = memory_mb
                    process_info.cpu_usage = cpu_percent

                    # メモリ使用量チェック
                    if memory_mb > process_info.max_memory_mb:
                        logger.warning(
                            f"メモリ使用量超過: {process_info.name} ({memory_mb:.1f}MB > {process_info.max_memory_mb}MB)"
                        )
                        # 設定されたメモリ制限の120%を超えた場合は警告
                        if memory_mb > process_info.max_memory_mb * 1.2:
                            logger.error(
                                f"危険なメモリ使用量: {process_info.name} ({memory_mb:.1f}MB), サービスを停止します"
                            )
                            self.stop_service(process_info.name, force=True)

                    # CPU使用率チェック
                    if cpu_percent > process_info.max_cpu_percent:
                        logger.warning(
                            f"CPU使用率超過: {process_info.name} ({cpu_percent:.1f}% > {process_info.max_cpu_percent}%)"
                        )

                except psutil.NoSuchProcess:
                    logger.warning(f"プロセス消失: {process_info.name}")
                    process_info.status = ProcessStatus.FAILED
                except psutil.AccessDenied:
                    logger.warning(f"プロセス情報アクセス不可: {process_info.name}")
                    # アクセスが拒否された場合も状態をUNKNOWNに設定

        except Exception as e:
            logger.error(f"ヘルスチェックエラー {process_info.name}: {e}")

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
