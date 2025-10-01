"""Service registry and process definitions for ClStock systems."""

from __future__ import annotations

import concurrent.futures
import os
import shlex
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union

import psutil

from config.settings import get_settings
from utils.logger_config import get_logger


PROJECT_ROOT = Path(__file__).parent.parent

logger = get_logger(__name__)
settings = get_settings()


class ProcessStatus(Enum):
    """State of a managed process."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ProcessInfo:
    """Definition and runtime metadata for a managed process."""

    name: str
    command: Union[str, Sequence[str]]
    working_dir: str = str(PROJECT_ROOT)
    env_vars: Dict[str, str] = field(default_factory=dict)
    auto_restart: bool = True
    max_restart_attempts: int = 3
    restart_delay: int = 5
    timeout: int = 300
    priority: int = 5
    max_memory_mb: int = 1000
    max_cpu_percent: float = 80

    pid: Optional[int] = None
    process: Optional[subprocess.Popen] = None
    status: ProcessStatus = ProcessStatus.STOPPED
    start_time: Optional[datetime] = None
    restart_count: int = 0
    last_error: Optional[str] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    stdout_thread: Optional[threading.Thread] = None
    stderr_thread: Optional[threading.Thread] = None


class ServiceRegistry:
    """Service lifecycle management and registry."""

    def __init__(self):
        self.processes: Dict[str, ProcessInfo] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self._executor_futures_lock = threading.Lock()
        self._executor_futures: Set[concurrent.futures.Future] = set()
        self._resource_limit_lock = threading.Lock()
        self._max_system_cpu_percent = 200
        self._max_system_memory_percent = 200
        self._define_default_services()

    # ------------------------------------------------------------------
    # Registry operations
    # ------------------------------------------------------------------
    def _define_default_services(self) -> None:
        services = {
            "dashboard": ProcessInfo(
                name="dashboard",
                command="python app/personal_dashboard.py",
                working_dir=str(PROJECT_ROOT),
            ),
            "demo_trading": ProcessInfo(
                name="demo_trading",
                command="python demo_start.py",
                working_dir=str(PROJECT_ROOT),
                auto_restart=False,
            ),
            "investment_system": ProcessInfo(
                name="investment_system",
                command="python full_auto_system.py",
                working_dir=str(PROJECT_ROOT),
            ),
            "deep_learning": ProcessInfo(
                name="deep_learning",
                command="python research/big_data_deep_learning.py",
                working_dir=str(PROJECT_ROOT),
            ),
            "integration_test": ProcessInfo(
                name="integration_test",
                command="python integration_test_enhanced.py",
                working_dir=str(PROJECT_ROOT),
                auto_restart=False,
            ),
            "optimized_system": ProcessInfo(
                name="optimized_system",
                command="python ultra_optimized_system.py",
                working_dir=str(PROJECT_ROOT),
            ),
            "selective_system": ProcessInfo(
                name="selective_system",
                command="python performance_test_enhanced.py",
                working_dir=str(PROJECT_ROOT),
            ),
        }

        for info in services.values():
            self.register_service(info)

    def register_service(self, process_info: ProcessInfo) -> bool:
        try:
            self.processes[process_info.name] = process_info
            logger.info("サービス登録: %s", process_info.name)
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("サービス登録失敗 %s: %s", process_info.name, exc)
            return False

    # ------------------------------------------------------------------
    # Service lifecycle operations
    # ------------------------------------------------------------------
    def _check_resource_limits(self, process_info: ProcessInfo) -> bool:
        with self._resource_limit_lock:
            system_cpu_percent = psutil.cpu_percent(interval=1)
            system_memory_percent = psutil.virtual_memory().percent

            if (
                system_cpu_percent + process_info.max_cpu_percent
                > self._max_system_cpu_percent
            ):
                logger.warning(
                    "CPUリソース不足のため %s の起動を延期", process_info.name
                )
                return False

            memory_percent = (
                process_info.max_memory_mb / psutil.virtual_memory().total * 100
            )
            if (
                system_memory_percent + memory_percent
                > self._max_system_memory_percent
            ):
                logger.warning(
                    "メモリリソース不足のため %s の起動を延期", process_info.name
                )
                return False

        return True

    def _start_output_logging(self, process_info: ProcessInfo) -> None:
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
                        log_func(
                            "[%s][%s] %s",
                            process_info.name,
                            stream_name,
                            line.rstrip(),
                        )
                except Exception as exc:  # pragma: no cover - debug logging
                    logger.debug(
                        "%s の %s ログ読み取り中にエラー: %s",
                        process_info.name,
                        stream_name,
                        exc,
                    )
                finally:
                    try:
                        stream.close()
                    except Exception:  # pragma: no cover - defensive
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

    def start_service(self, name: str) -> bool:
        if name not in self.processes:
            logger.error("未登録のサービス: %s", name)
            return False

        process_info = self.processes[name]

        if process_info.status == ProcessStatus.RUNNING:
            logger.warning("サービスは既に実行中: %s", name)
            return True

        if not self._check_resource_limits(process_info):
            return False

        try:
            process_info.status = ProcessStatus.STARTING
            logger.info("サービス開始: %s", name)

            env = os.environ.copy()
            env.update(process_info.env_vars)

            capture_output = bool(
                getattr(settings.process, "log_process_output", False)
            )

            popen_kwargs = {"cwd": process_info.working_dir, "env": env}
            if capture_output:
                popen_kwargs.update(
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

            command = process_info.command
            if isinstance(command, str):
                argv = shlex.split(command)
            elif isinstance(command, Sequence):
                argv = list(command)
            else:
                raise TypeError(
                    f"Unsupported command type for service {name}: {type(command)!r}"
                )

            process_info.process = subprocess.Popen(argv, **popen_kwargs)
            process_info.pid = process_info.process.pid
            process_info.start_time = datetime.now()
            process_info.status = ProcessStatus.RUNNING
            process_info.last_error = None

            if capture_output:
                self._start_output_logging(process_info)
            else:
                process_info.stdout_thread = None
                process_info.stderr_thread = None

            logger.info(
                "サービス開始完了: %s (PID: %s)",
                name,
                process_info.pid,
            )
            return True
        except FileNotFoundError as exc:
            process_info.status = ProcessStatus.FAILED
            process_info.last_error = f"実行ファイルが見つかりません: {exc}"
            logger.error(
                "サービス開始失敗 %s: 実行ファイルが見つかりません - %s",
                name,
                exc,
            )
            return False
        except PermissionError as exc:
            process_info.status = ProcessStatus.FAILED
            process_info.last_error = f"実行権限がありません: {exc}"
            logger.error("サービス開始失敗 %s: 実行権限エラー - %s", name, exc)
            return False
        except subprocess.SubprocessError as exc:
            process_info.status = ProcessStatus.FAILED
            process_info.last_error = str(exc)
            logger.error("サービス開始失敗 %s: サブプロセスエラー - %s", name, exc)
            return False
        except Exception as exc:  # pragma: no cover - defensive logging
            process_info.status = ProcessStatus.FAILED
            process_info.last_error = str(exc)
            logger.error("サービス開始失敗 %s: 予期しないエラー - %s", name, exc)
            return False

    def stop_service(self, name: str, force: bool = False) -> bool:
        if name not in self.processes:
            logger.error("未登録のサービス: %s", name)
            return False

        process_info = self.processes[name]

        if process_info.status != ProcessStatus.RUNNING:
            logger.warning("サービスは実行中ではありません: %s", name)
            return True

        try:
            process_info.status = ProcessStatus.STOPPING
            logger.info("サービス停止: %s", name)

            if process_info.process:
                if force:
                    process_info.process.kill()
                else:
                    process_info.process.terminate()

                try:
                    process_info.process.wait(timeout=process_info.timeout)
                except subprocess.TimeoutExpired:
                    logger.warning("タイムアウト、強制終了: %s", name)
                    process_info.process.kill()
                    process_info.process.wait(timeout=10)

            process_info.status = ProcessStatus.STOPPED
            process_info.pid = None
            process_info.process = None
            if process_info.stdout_thread and process_info.stdout_thread.is_alive():
                process_info.stdout_thread.join(timeout=1)
            if process_info.stderr_thread and process_info.stderr_thread.is_alive():
                process_info.stderr_thread.join(timeout=1)
            process_info.stdout_thread = None
            process_info.stderr_thread = None

            logger.info("サービス停止完了: %s", name)
            return True
        except ProcessLookupError as exc:
            logger.warning("プロセスが既に終了しています: %s - %s", name, exc)
            process_info.status = ProcessStatus.STOPPED
            process_info.pid = None
            process_info.process = None
            return True
        except subprocess.SubprocessError as exc:
            logger.error("サービス停止失敗 %s: サブプロセスエラー - %s", name, exc)
            return False
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("サービス停止失敗 %s: 予期しないエラー - %s", name, exc)
            return False

    # ------------------------------------------------------------------
    # High level helpers
    # ------------------------------------------------------------------
    def restart_service(self, name: str) -> bool:
        logger.info("サービス再起動: %s", name)
        if not self.stop_service(name):
            return False
        time.sleep(2)
        return self.start_service(name)

    def get_service_status(self, name: str) -> Optional[ProcessInfo]:
        return self.processes.get(name)

    def list_services(self) -> List[ProcessInfo]:
        return list(self.processes.values())

    def start_multiple_services(
        self, names: Iterable[str], max_parallel: int = 3
    ) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        sorted_services = sorted(
            [name for name in names if name in self.processes],
            key=lambda svc: self.processes[svc].priority,
            reverse=True,
        )

        service_groups = [
            sorted_services[i : i + max_parallel]
            for i in range(0, len(sorted_services), max_parallel)
        ]

        for group in service_groups:
            futures: Dict[concurrent.futures.Future, str] = {}
            for name in group:
                if not self._check_resource_limits(self.processes[name]):
                    results[name] = False
                    continue
                future = self._executor.submit(self.start_service, name)
                self._register_executor_future(future)
                futures[future] = name

            for future in concurrent.futures.as_completed(futures):
                svc = futures[future]
                try:
                    results[svc] = future.result(timeout=10)
                except concurrent.futures.TimeoutError:
                    logger.error("サービス %s の起動がタイムアウト", svc)
                    results[svc] = False
                except Exception as exc:  # pragma: no cover - logging
                    logger.error("サービス %s の起動中にエラー: %s", svc, exc)
                    results[svc] = False

            time.sleep(1)

        return results

    def stop_all_services(self, force: bool = False) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        for name in list(self.processes.keys()):
            results[name] = self.stop_service(name, force=force)
        return results

    # ------------------------------------------------------------------
    # Executor helpers
    # ------------------------------------------------------------------
    def _register_executor_future(self, future: concurrent.futures.Future) -> None:
        with self._executor_futures_lock:
            self._executor_futures.add(future)

    def cleanup_executor(self) -> None:
        with self._executor_futures_lock:
            done = {fut for fut in self._executor_futures if fut.done()}
            self._executor_futures -= done
        self._executor.shutdown(wait=False)
