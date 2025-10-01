"""
ClStock プロセス管理システム
複数のサービスの起動・停止・監視を統合管理
"""

import os
import sys
import psutil
import subprocess
import threading
import time
import signal
import shlex
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.logger_config import get_logger
from config.settings import get_settings
from systems.resource_monitor import ResourceMonitor

logger = get_logger(__name__)
settings = get_settings()


class ProcessStatus(Enum):
    """プロセス状態"""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ProcessInfo:
    """プロセス情報"""

    name: str
    command: Union[str, Sequence[str]]
    working_dir: str = str(PROJECT_ROOT)
    env_vars: Dict[str, str] = field(default_factory=dict)
    auto_restart: bool = True
    max_restart_attempts: int = 3
    restart_delay: int = 5
    timeout: int = 300
    priority: int = 5  # プロセス優先度 (0-10, 10が最高)
    max_memory_mb: int = 1000  # 最大メモリ使用量 (MB)
    max_cpu_percent: float = 80  # 最大CPU使用率 (%)

    # 実行時情報
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


class ProcessManager:
    """プロセス管理クラス"""

    def __init__(self):
        self.processes: Dict[str, ProcessInfo] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._shutdown_lock = threading.Lock()  # シャットダウン処理の排他制御用
        self._shutdown_thread: Optional[threading.Thread] = (
            None  # シャットダウンスレッドの参照
        )
        # プロセスプールの追加
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self._executor_futures_lock = threading.Lock()
        self._executor_futures: Set[concurrent.futures.Future] = set()
        # リソース制限の追加
        self._resource_limit_lock = threading.Lock()
        self._current_cpu_usage = 0.0
        self._current_memory_usage = 0.0
        self._max_system_cpu_percent = 80  # システム全体の最大CPU使用率
        self._max_system_memory_percent = 80  # システム全体の最大メモリ使用率
        self.resource_monitor = ResourceMonitor()

        # デフォルトサービス定義
        self._define_default_services()

        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _define_default_services(self):
        """デフォルトサービスの定義"""
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

        for name, process_info in services.items():
            self.register_service(process_info)

    def register_service(self, process_info: ProcessInfo) -> bool:
        """サービスの登録"""
        try:
            self.processes[process_info.name] = process_info
            logger.info(f"サービス登録: {process_info.name}")
            return True
        except KeyError as e:
            logger.error(f"サービス登録失敗 {process_info.name}: キーエラー - {e}")
            return False
        except Exception as e:
            logger.error(f"サービス登録失敗 {process_info.name}: {e}")
            return False

    def _check_resource_limits(self, process_info: ProcessInfo) -> bool:
        """リソース制限チェック"""
        with self._resource_limit_lock:
            # システム全体のリソース使用状況を取得
            system_usage = self.resource_monitor.get_system_usage()
            system_cpu_percent = system_usage.cpu_percent
            system_memory_percent = system_usage.memory_percent
            
            # 新しいプロセスのリソース要件が制限を超えるかチェック
            if system_cpu_percent + process_info.max_cpu_percent > self._max_system_cpu_percent:
                logger.warning(f"CPUリソース不足のため {process_info.name} の起動を延期: "
                              f"現在 {system_cpu_percent:.1f}% + 要求 {process_info.max_cpu_percent}% > 制限 {self._max_system_cpu_percent}%")
                return False
                
            total_memory = system_usage.memory_total or 1
            requested_memory_percent = (process_info.max_memory_mb * 1024 * 1024) / total_memory * 100
            if system_memory_percent + requested_memory_percent > self._max_system_memory_percent:
                logger.warning(f"メモリリソース不足のため {process_info.name} の起動を延期: "
                              f"現在 {system_memory_percent:.1f}% + 要求 {process_info.max_memory_mb}MB > 制限 {self._max_system_memory_percent}%")
                return False
        
        return True

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

            capture_output = bool(
                getattr(settings.process, "log_process_output", False)
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
        """サービスの停止"""
        if name not in self.processes:
            logger.error(f"未登録のサービス: {name}")
            return False

        process_info = self.processes[name]

        if process_info.status != ProcessStatus.RUNNING:
            logger.warning(f"サービスは実行中ではありません: {name}")
            return True

        try:
            process_info.status = ProcessStatus.STOPPING
            logger.info(f"サービス停止: {name}")

            if process_info.process:
                if force:
                    process_info.process.kill()
                else:
                    process_info.process.terminate()

                # 停止待機（タイムアウトをサービス定義のタイムアウト値に設定）
                try:
                    process_info.process.wait(timeout=process_info.timeout)
                except subprocess.TimeoutExpired:
                    logger.warning(f"タイムアウト、強制終了: {name}")
                    process_info.process.kill()
                    process_info.process.wait(
                        timeout=10
                    )  # 強制終了後もタイムアウトを設定

            process_info.status = ProcessStatus.STOPPED
            process_info.pid = None
            process_info.process = None
            if process_info.stdout_thread and process_info.stdout_thread.is_alive():
                process_info.stdout_thread.join(timeout=1)
            if process_info.stderr_thread and process_info.stderr_thread.is_alive():
                process_info.stderr_thread.join(timeout=1)
            process_info.stdout_thread = None
            process_info.stderr_thread = None

            logger.info(f"サービス停止完了: {name}")
            return True

        except ProcessLookupError as e:
            logger.warning(f"プロセスが既に終了しています: {name} - {e}")
            process_info.status = ProcessStatus.STOPPED
            process_info.pid = None
            process_info.process = None
            return True
        except subprocess.SubprocessError as e:
            logger.error(f"サービス停止失敗 {name}: サブプロセスエラー - {e}")
            return False
        except Exception as e:
            logger.error(f"サービス停止失敗 {name}: 予期しないエラー - {e}")
            return False

    def start_multiple_services(self, names: List[str], max_parallel: int = 3) -> Dict[str, bool]:
        """複数サービスの並列起動"""
        results = {}
        
        # 優先度に基づいてサービスをソート (高い順)
        sorted_services = sorted(names, key=lambda x: self.processes[x].priority if x in self.processes else 0, reverse=True)
        
        # 最大並列数に基づいてサービスをグループ化
        service_groups = [sorted_services[i:i + max_parallel] for i in range(0, len(sorted_services), max_parallel)]
        
        for group in service_groups:
            # 各グループ内のサービスを並列に起動
            futures = {}
            for name in group:
                if name in self.processes:
                    # リソース制限チェック
                    if self._check_resource_limits(self.processes[name]):
                        future = self._executor.submit(self.start_service, name)
                        self._register_executor_future(future)
                        futures[future] = name
                    else:
                        results[name] = False
                        logger.warning(f"リソース不足のためサービス {name} の起動をスキップ")
            
            # 各グループの完了を待機
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    result = future.result(timeout=10)  # 各サービスの起動に最大10秒まで待機
                    results[name] = result
                except concurrent.futures.TimeoutError:
                    logger.error(f"サービス {name} の起動がタイムアウト")
                    results[name] = False
                except Exception as e:
                    logger.error(f"サービス {name} の起動中にエラー: {e}")
                    results[name] = False
            
            # グループ間の待機時間（リソース使用量の落ち着きのため）
            time.sleep(1)
        
        return results

    def restart_service(self, name: str) -> bool:
        """サービスの再起動"""
        logger.info(f"サービス再起動: {name}")

        if not self.stop_service(name):
            return False

        time.sleep(2)  # 再起動間隔

        return self.start_service(name)

    def get_service_status(self, name: str) -> Optional[ProcessInfo]:
        """サービス状態の取得"""
        return self.processes.get(name)

    def list_services(self) -> List[ProcessInfo]:
        """全サービスのリスト"""
        return list(self.processes.values())

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
            system_usage = self.resource_monitor.get_system_usage()
            system_cpu_percent = system_usage.cpu_percent
            system_memory_percent = system_usage.memory_percent

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
                    original_settings = settings.process  # 設定から元の値を取得
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

    def stop_all_services(self, force: bool = False):
        """全サービスの停止"""
        logger.info("全サービス停止開始")

        # 監視停止
        self.stop_monitoring()

        # 全サービス停止（エラーが発生しても他のサービスの停止を続行）
        success_count = 0
        failure_count = 0

        for name in list(self.processes.keys()):
            try:
                if self.stop_service(name, force=force):
                    success_count += 1
                else:
                    failure_count += 1
            except Exception as e:
                logger.error(f"サービス停止中にエラー発生 {name}: {e}")
                failure_count += 1

        logger.info(
            f"全サービス停止完了 (成功: {success_count}, 失敗: {failure_count})"
        )

        # シャットダウンイベントが設定されている場合、プロセス終了を待機
        if self._shutdown_event.is_set():
            logger.info("シャットダウンイベント検出、プロセス終了準備完了")

    def get_system_status(self) -> Dict:
        """システム全体の状態"""
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

    def _signal_handler(self, signum):
        """シグナルハンドラー"""
        logger.info(f"シグナル受信: {signum}")

        # 排他制御で複数のシャットダウン処理を防止
        if not self._shutdown_lock.acquire(blocking=False):
            logger.info("シャットダウンは既に進行中")
            return

        try:
            # 既にシャットダウンが進行中の場合は何もしない
            if self._shutdown_event.is_set():
                logger.info("シャットダウンは既に進行中")
                return

            # シャットダウンイベントを設定
            self._shutdown_event.set()

            # 別スレッドで非同期シャットダウン処理を実行
            self._shutdown_thread = threading.Thread(
                target=self._graceful_shutdown, daemon=True, name="ShutdownThread"
            )
            self._shutdown_thread.start()
        finally:
            self._shutdown_lock.release()

    def _graceful_shutdown(self):
        """グレースフルシャットダウン処理"""
        try:
            logger.info("グレースフルシャットダウン開始")
            self.stop_all_services(force=True)

            # 並列実行プールのシャットダウン
            logger.info("並列実行プールをシャットダウン (最大10秒待機)")
            shutdown_timeout = 10
            deadline = time.monotonic() + shutdown_timeout
            pending_futures = self._collect_executor_futures()
            all_completed = True

            for future in pending_futures:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    all_completed = False
                    logger.warning("並列タスクの完了待機がタイムアウトしました")
                    break

                try:
                    future.result(timeout=remaining)
                except concurrent.futures.TimeoutError:
                    logger.warning("並列タスクの完了待機がタイムアウトしました")
                    all_completed = False
                    break
                except Exception as future_error:
                    logger.error(f"並列タスクの実行中にエラー: {future_error}")

            if all_completed:
                self._executor.shutdown(wait=True)
            else:
                self._executor.shutdown(wait=False)
                # タイムアウトした未完了タスクはキャンセルを試みる
                self._cancel_pending_futures()

            logger.info("全サービス停止完了、プロセス終了")
            # シャットダウンイベントをクリア
            self._shutdown_event.clear()
            # プロセス終了
            os._exit(0)
        except Exception as e:
            logger.error(f"シャットダウン中にエラー発生: {e}")
            os._exit(1)

    def _register_executor_future(self, future: concurrent.futures.Future) -> None:
        """シャットダウン時に待機するため、実行中の Future を登録する"""

        with self._executor_futures_lock:
            self._executor_futures.add(future)

        def _remove_completed(fut: concurrent.futures.Future) -> None:
            with self._executor_futures_lock:
                self._executor_futures.discard(fut)

        future.add_done_callback(_remove_completed)

    def _collect_executor_futures(self) -> List[concurrent.futures.Future]:
        """現在登録されている Future のスナップショットを取得"""

        with self._executor_futures_lock:
            return list(self._executor_futures)

    def _cancel_pending_futures(self) -> None:
        """未完了の Future をキャンセル"""

        with self._executor_futures_lock:
            for future in list(self._executor_futures):
                future.cancel()


# グローバルインスタンス
process_manager = ProcessManager()


def get_process_manager() -> ProcessManager:
    """プロセスマネージャーの取得"""
    return process_manager


if __name__ == "__main__":
    # デモ実行
    manager = get_process_manager()

    print("=== ClStock プロセス管理システム ===")
    print("利用可能なサービス:")

    for service in manager.list_services():
        print(f"  - {service.name}: {service.command}")

    print("\n監視開始...")
    manager.start_monitoring()

    try:
        # 対話モード
        while True:
            command = (
                input("\nコマンド (start/stop/restart/status/quit): ").strip().lower()
            )

            if command == "quit":
                break
            elif command == "status":
                status = manager.get_system_status()
                print(f"サービス数: {status['total_services']}")
                print(f"実行中: {status['running']}")
                print(f"失敗: {status['failed']}")

                for service in manager.list_services():
                    print(f"  {service.name}: {service.status.value}")

            elif command.startswith(("start", "stop", "restart")):
                parts = command.split()
                if len(parts) == 2:
                    action, service_name = parts

                    if action == "start":
                        manager.start_service(service_name)
                    elif action == "stop":
                        manager.stop_service(service_name)
                    elif action == "restart":
                        manager.restart_service(service_name)
                else:
                    print("使用法: <action> <service_name>")

    except KeyboardInterrupt:
        print("\n終了...")

    finally:
        manager.stop_all_services()