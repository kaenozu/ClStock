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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.logger_config import get_logger
from config.settings import get_settings

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
    command: str
    working_dir: str = str(PROJECT_ROOT)
    env_vars: Dict[str, str] = field(default_factory=dict)
    auto_restart: bool = True
    max_restart_attempts: int = 3
    restart_delay: int = 5
    timeout: int = 300

    # 実行時情報
    pid: Optional[int] = None
    process: Optional[subprocess.Popen] = None
    status: ProcessStatus = ProcessStatus.STOPPED
    start_time: Optional[datetime] = None
    restart_count: int = 0
    last_error: Optional[str] = None


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
                command="python investment_system.py",
                working_dir=str(PROJECT_ROOT),
            ),
            "deep_learning": ProcessInfo(
                name="deep_learning",
                command="python big_data_deep_learning.py",
                working_dir=str(PROJECT_ROOT),
            ),
            "ensemble_test": ProcessInfo(
                name="ensemble_test",
                command="python advanced_ensemble_test.py",
                working_dir=str(PROJECT_ROOT),
                auto_restart=False,
            ),
            "clstock_main": ProcessInfo(
                name="clstock_main",
                command="python clstock_main.py --integrated 7203",
                working_dir=str(PROJECT_ROOT),
                auto_restart=False,
            ),
            "optimized_system": ProcessInfo(
                name="optimized_system",
                command="python optimized_investment_system.py",
                working_dir=str(PROJECT_ROOT),
            ),
            "selective_system": ProcessInfo(
                name="selective_system",
                command="python super_selective_system.py",
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

    def start_service(self, name: str) -> bool:
        """サービスの開始"""
        if name not in self.processes:
            logger.error(f"未登録のサービス: {name}")
            return False

        process_info = self.processes[name]

        if process_info.status == ProcessStatus.RUNNING:
            logger.warning(f"サービスは既に実行中: {name}")
            return True

        try:
            process_info.status = ProcessStatus.STARTING
            logger.info(f"サービス開始: {name}")

            # 環境変数設定
            env = os.environ.copy()
            env.update(process_info.env_vars)

            # プロセス開始
            process_info.process = subprocess.Popen(
                process_info.command.split(),
                cwd=process_info.working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            process_info.pid = process_info.process.pid
            process_info.start_time = datetime.now()
            process_info.status = ProcessStatus.RUNNING
            process_info.last_error = None

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

        logger.info("プロセス監視停止")

    def wait_for_shutdown(self, timeout: int = 60):
        """シャットダウン完了を待機"""
        if self._shutdown_thread and self._shutdown_thread.is_alive():
            self._shutdown_thread.join(timeout=timeout)
            if self._shutdown_thread.is_alive():
                logger.warning("シャットダウンスレッドの終了待機タイムアウト")

    def _monitor_processes(self):
        """プロセス監視メインループ"""
        while self.monitoring_active and not self._shutdown_event.is_set():
            try:
                for name, process_info in self.processes.items():
                    if process_info.status == ProcessStatus.RUNNING:
                        self._check_process_health(process_info)

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

            # メモリ使用量チェック
            if process_info.pid:
                try:
                    process = psutil.Process(process_info.pid)
                    memory_mb = process.memory_info().rss / 1024 / 1024

                    if memory_mb > 1000:  # 1GB超過
                        logger.warning(
                            f"高メモリ使用量検出: {process_info.name} ({memory_mb:.1f}MB)"
                        )
                except psutil.NoSuchProcess:
                    logger.warning(f"プロセス消失: {process_info.name}")
                    process_info.status = ProcessStatus.FAILED

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

    def _signal_handler(self, signum, frame):
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
            logger.info("全サービス停止完了、プロセス終了")
            # シャットダウンイベントをクリア
            self._shutdown_event.clear()
            # プロセス終了
            os._exit(0)
        except Exception as e:
            logger.error(f"シャットダウン中にエラー発生: {e}")
            os._exit(1)


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
