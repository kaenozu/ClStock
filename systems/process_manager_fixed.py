#!/usr/bin/env python3
"""
プロセス管理システム（セキュリティ修正版）
- コマンドインジェクション対策
- デッドロック対策
"""

import subprocess
import threading
import queue
import time
import os
import json
import psutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessStatus(Enum):
    """プロセス状態"""
    STOPPED = "停止"
    STARTING = "開始中"
    RUNNING = "実行中"
    STOPPING = "停止中"
    FAILED = "失敗"
    UNKNOWN = "不明"


@dataclass
class ProcessInfo:
    """プロセス情報"""
    name: str
    command: List[str]  # コマンドを文字列のリストとして保持（セキュリティ対策）
    working_dir: str
    status: ProcessStatus = ProcessStatus.STOPPED
    pid: Optional[int] = None
    process: Optional[subprocess.Popen] = None
    start_time: Optional[datetime] = None
    last_error: Optional[str] = None
    restart_count: int = 0
    auto_restart: bool = True
    max_restarts: int = 3
    env_vars: Dict[str, str] = None
    log_file: Optional[str] = None
    output_reader: Optional[threading.Thread] = None

    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}


class OutputReader(threading.Thread):
    """非同期で出力を読み取るスレッド（デッドロック対策）"""

    def __init__(self, pipe, log_file=None, pipe_name="output"):
        super().__init__(daemon=True)
        self.pipe = pipe
        self.log_file = log_file
        self.pipe_name = pipe_name
        self.lines = queue.Queue()
        self._stop_event = threading.Event()

    def run(self):
        """出力を非同期で読み取る"""
        try:
            while not self._stop_event.is_set():
                if self.pipe:
                    line = self.pipe.readline()
                    if line:
                        # キューに追加
                        self.lines.put(line)

                        # ログファイルに書き込み
                        if self.log_file:
                            with open(self.log_file, 'a', encoding='utf-8') as f:
                                f.write(f"[{self.pipe_name}] {line}")
                    else:
                        # パイプが閉じられた
                        break
                else:
                    break
        except Exception as e:
            logger.error(f"出力読み取りエラー: {e}")

    def stop(self):
        """スレッドを停止"""
        self._stop_event.set()

    def get_recent_lines(self, n=100):
        """最近のn行を取得"""
        lines = []
        while not self.lines.empty() and len(lines) < n:
            try:
                lines.append(self.lines.get_nowait())
            except queue.Empty:
                break
        return lines


class SecureProcessManager:
    """セキュアなプロセス管理クラス"""

    def __init__(self, config_file: str = "process_config.json"):
        self.config_file = config_file
        self.processes: Dict[str, ProcessInfo] = {}
        self.monitoring_active = False
        self.monitor_thread = None
        self._load_config()
        self._setup_default_services()

    def _validate_command(self, command: List[str]) -> bool:
        """コマンドの安全性を検証"""
        # 危険な文字やコマンドをチェック
        dangerous_chars = [';', '&', '|', '>', '<', '`', '$', '(', ')', '{', '}']
        dangerous_commands = ['rm', 'del', 'format', 'shutdown', 'reboot']

        for part in command:
            # 危険な文字のチェック
            for char in dangerous_chars:
                if char in part:
                    logger.warning(f"危険な文字が検出されました: {char} in {part}")
                    return False

            # 危険なコマンドのチェック
            base_command = os.path.basename(part).lower()
            for dangerous in dangerous_commands:
                if dangerous in base_command:
                    logger.warning(f"危険なコマンドが検出されました: {dangerous}")
                    return False

        return True

    def _sanitize_input(self, input_str: str) -> str:
        """入力文字列をサニタイズ"""
        # 英数字、日本語、一部の記号のみを許可
        import re
        # 危険な文字を削除
        sanitized = re.sub(r'[;&|><`$(){}]', '', input_str)
        # 連続するスペースを単一スペースに
        sanitized = re.sub(r'\s+', ' ', sanitized)
        return sanitized.strip()

    def _setup_default_services(self):
        """デフォルトサービスの設定"""
        default_services = {
            "dashboard": ProcessInfo(
                name="dashboard",
                command=["python", "app/personal_dashboard.py"],
                working_dir=str(Path.cwd()),
                log_file="logs/dashboard.log"
            ),
            "demo_trading": ProcessInfo(
                name="demo_trading",
                command=["python", "demo_trading.py"],
                working_dir=str(Path.cwd()),
                log_file="logs/demo_trading.log"
            ),
            "optimized_system": ProcessInfo(
                name="optimized_system",
                command=["python", "optimized_investment_system.py"],
                working_dir=str(Path.cwd()),
                log_file="logs/optimized_system.log"
            ),
        }

        for name, info in default_services.items():
            self.processes[name] = info

    def start_service(self, name: str, symbol: Optional[str] = None) -> bool:
        """サービスの安全な開始"""
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

            # コマンドの構築（リスト形式で安全）
            command = process_info.command.copy()

            # シンボルが指定されている場合、サニタイズして追加
            if symbol:
                sanitized_symbol = self._sanitize_input(symbol)
                # 数字のみを許可（株式コードの場合）
                import re
                if re.match(r'^[0-9]+$', sanitized_symbol):
                    command.append(sanitized_symbol)
                else:
                    logger.error(f"無効なシンボル: {symbol}")
                    return False

            # コマンドの検証
            if not self._validate_command(command):
                logger.error(f"安全でないコマンド: {command}")
                process_info.status = ProcessStatus.FAILED
                return False

            # 環境変数設定
            env = os.environ.copy()
            env.update(process_info.env_vars)

            # ログファイルの準備
            if process_info.log_file:
                log_path = Path(process_info.log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)

            # プロセス開始（subprocess.runではなくPopenを使用、ただし安全に）
            process_info.process = subprocess.Popen(
                command,  # リスト形式で渡す（シェルインジェクション対策）
                cwd=process_info.working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False  # シェルを使用しない（セキュリティ対策）
            )

            # 非同期出力リーダーを開始（デッドロック対策）
            stdout_reader = OutputReader(
                process_info.process.stdout,
                process_info.log_file,
                "stdout"
            )
            stderr_reader = OutputReader(
                process_info.process.stderr,
                process_info.log_file,
                "stderr"
            )

            stdout_reader.start()
            stderr_reader.start()

            process_info.output_reader = stdout_reader
            process_info.pid = process_info.process.pid
            process_info.start_time = datetime.now()
            process_info.status = ProcessStatus.RUNNING
            process_info.last_error = None

            logger.info(f"サービス開始完了: {name} (PID: {process_info.pid})")
            return True

        except Exception as e:
            process_info.status = ProcessStatus.FAILED
            process_info.last_error = str(e)
            logger.error(f"サービス開始失敗 {name}: {e}")
            return False

    def stop_service(self, name: str, force: bool = False) -> bool:
        """サービスの安全な停止"""
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

            if process_info.output_reader:
                process_info.output_reader.stop()

            if process_info.process:
                if force:
                    process_info.process.kill()
                else:
                    process_info.process.terminate()

                # プロセスの終了を待つ
                try:
                    process_info.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"プロセス終了タイムアウト、強制終了: {name}")
                    process_info.process.kill()
                    process_info.process.wait()

            process_info.status = ProcessStatus.STOPPED
            process_info.pid = None
            process_info.process = None
            process_info.output_reader = None

            logger.info(f"サービス停止完了: {name}")
            return True

        except Exception as e:
            process_info.status = ProcessStatus.FAILED
            process_info.last_error = str(e)
            logger.error(f"サービス停止失敗 {name}: {e}")
            return False

    def execute_safe_command(self, command: List[str], timeout: int = 30) -> tuple[bool, str, str]:
        """安全なコマンド実行（同期的）"""
        try:
            # コマンドの検証
            if not self._validate_command(command):
                return False, "", "安全でないコマンドです"

            # subprocess.runを使用（タイムアウト付き）
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False  # シェルを使用しない
            )

            return result.returncode == 0, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, "", "コマンドタイムアウト"
        except Exception as e:
            return False, "", str(e)

    def predict_stock_safe(self, symbol: str) -> Optional[Dict[str, Any]]:
        """安全な株価予測実行"""
        # シンボルのサニタイズと検証
        sanitized_symbol = self._sanitize_input(symbol)

        # 数字のみを許可
        import re
        if not re.match(r'^[0-9]{4}$', sanitized_symbol):
            logger.error(f"無効な銘柄コード: {symbol}")
            return None

        try:
            # 安全なコマンド構築
            command = [
                "python",
                "models_new/precision/precision_87_system.py",
                "--symbol",
                sanitized_symbol
            ]

            success, stdout, stderr = self.execute_safe_command(command, timeout=60)

            if success:
                # 結果をパース
                return {"status": "success", "output": stdout}
            else:
                return {"status": "error", "error": stderr}

        except Exception as e:
            logger.error(f"予測実行エラー: {e}")
            return None

    def get_system_status(self) -> Dict[str, Any]:
        """システム状態の取得"""
        running = sum(1 for p in self.processes.values()
                     if p.status == ProcessStatus.RUNNING)
        failed = sum(1 for p in self.processes.values()
                    if p.status == ProcessStatus.FAILED)

        return {
            "total_services": len(self.processes),
            "running": running,
            "stopped": len(self.processes) - running - failed,
            "failed": failed,
            "monitoring_active": self.monitoring_active,
            "timestamp": datetime.now()
        }

    def _load_config(self):
        """設定ファイルの読み込み"""
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 設定を安全に適用
                    logger.info("設定ファイルを読み込みました")
            except Exception as e:
                logger.error(f"設定ファイル読み込みエラー: {e}")

    def _save_config(self):
        """設定ファイルの保存"""
        try:
            config = {
                "services": {
                    name: {
                        "command": info.command,
                        "working_dir": info.working_dir,
                        "auto_restart": info.auto_restart,
                        "max_restarts": info.max_restarts
                    }
                    for name, info in self.processes.items()
                }
            }

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"設定ファイル保存エラー: {e}")


# グローバルインスタンス
_process_manager = None


def get_secure_process_manager() -> SecureProcessManager:
    """セキュアなプロセスマネージャーのシングルトンインスタンスを取得"""
    global _process_manager
    if _process_manager is None:
        _process_manager = SecureProcessManager()
    return _process_manager


if __name__ == "__main__":
    # デモ実行
    manager = get_secure_process_manager()

    print("=== セキュアプロセス管理システム ===")

    # 安全なコマンド実行テスト
    print("\n1. 安全なコマンド実行:")
    success, stdout, stderr = manager.execute_safe_command(["echo", "Hello, World!"])
    print(f"結果: {success}, 出力: {stdout.strip()}")

    # 危険なコマンドのブロックテスト
    print("\n2. 危険なコマンドのブロック:")
    dangerous_command = ["echo", "test; rm -rf /"]  # セミコロンを含む
    success, stdout, stderr = manager.execute_safe_command(dangerous_command)
    print(f"結果: {success}, エラー: {stderr}")

    # 安全な予測実行
    print("\n3. 安全な株価予測:")
    result = manager.predict_stock_safe("7203")
    if result:
        print(f"予測結果: {result['status']}")

    # システム状態
    status = manager.get_system_status()
    print(f"\n4. システム状態:")
    print(f"総サービス数: {status['total_services']}")
    print(f"実行中: {status['running']}")
    print(f"停止中: {status['stopped']}")