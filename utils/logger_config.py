#!/usr/bin/env python3
"""
ログ設定の一元管理モジュール
logging.basicConfigの複数回呼び出し問題を解決
"""

import logging
import sys
import threading
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta


def setup_logger(
    name: str = None, level: int = logging.INFO, format_string: str = None
) -> logging.Logger:
    """
    安全なログ設定

    Args:
        name: ロガー名（Noneの場合はルートロガー）
        level: ログレベル
        format_string: フォーマット文字列

    Returns:
        設定済みロガー
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # ロガーを取得
    logger = logging.getLogger(name)

    # 既にハンドラーが設定されている場合は重複を避ける
    if logger.handlers:
        return logger

    # レベル設定
    logger.setLevel(level)

    # コンソールハンドラーを作成
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # フォーマッターを作成
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)

    # ハンドラーをロガーに追加
    logger.addHandler(console_handler)

    # 親ロガーへの伝播を防ぐ（重複ログを避ける）
    if name:
        logger.propagate = False

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    ロガーを取得（既存の場合はそのまま返す）

    Args:
        name: ロガー名

    Returns:
        ロガー
    """
    logger = logging.getLogger(name)

    # まだ設定されていない場合は設定
    if not logger.handlers:
        return setup_logger(name)

    return logger


def set_log_level(level: int, logger_name: str = None):
    """
    安全にログレベルを変更

    Args:
        level: 新しいログレベル
        logger_name: ロガー名（Noneの場合はルートロガー）
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # ハンドラーのレベルも更新
    for handler in logger.handlers:
        handler.setLevel(level)


class CentralizedLogger:
    """集約ログ管理クラス"""

    def __init__(self):
        self.log_collectors: Dict[str, List[str]] = {}
        self.log_files: Dict[str, Path] = {}
        self.active_services: Set[str] = set()
        self._lock = threading.Lock()

        # ログディレクトリ作成
        self.log_dir = Path(__file__).parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # 集約ログファイル
        self.centralized_log = self.log_dir / "clstock_centralized.log"

    def register_service(self, service_name: str, log_file_path: Optional[str] = None):
        """サービスのログ収集登録"""
        with self._lock:
            self.active_services.add(service_name)
            self.log_collectors[service_name] = []

            if log_file_path:
                self.log_files[service_name] = Path(log_file_path)

    def collect_service_logs(
        self, service_name: str, max_lines: int = 100
    ) -> List[str]:
        """サービスのログ収集"""
        logs = []

        if service_name in self.log_files:
            log_file = self.log_files[service_name]
            if log_file.exists():
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        logs.extend(lines[-max_lines:])
                except Exception as e:
                    logs.append(f"[ERROR] ログ読み込み失敗: {e}")

        return logs

    def collect_all_logs(self) -> Dict[str, List[str]]:
        """全サービスのログ集約"""
        all_logs = {}

        for service_name in self.active_services:
            all_logs[service_name] = self.collect_service_logs(service_name)

        return all_logs

    def write_centralized_log(
        self, message: str, level: str = "INFO", service: str = "SYSTEM"
    ):
        """集約ログへの書き込み"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] [{service}] {message}\n"

        try:
            with open(self.centralized_log, "a", encoding="utf-8") as f:
                f.write(formatted_message)
        except Exception as e:
            # フォールバック：標準エラー出力
            print(f"集約ログ書き込み失敗: {e}", file=sys.stderr)

    def get_recent_logs(
        self, hours: int = 1, service_filter: Optional[str] = None
    ) -> List[str]:
        """最近のログ取得"""
        recent_logs = []
        cutoff_time = datetime.now() - timedelta(hours=hours)

        if not self.centralized_log.exists():
            return recent_logs

        try:
            with open(self.centralized_log, "r", encoding="utf-8") as f:
                for line in f:
                    # タイムスタンプ解析
                    try:
                        timestamp_str = line.split("]")[0][1:]  # [timestamp] の中身
                        log_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

                        if log_time >= cutoff_time:
                            if service_filter:
                                if f"[{service_filter}]" in line:
                                    recent_logs.append(line.strip())
                            else:
                                recent_logs.append(line.strip())
                    except (ValueError, IndexError):
                        # タイムスタンプ解析失敗時はスキップ
                        continue

        except Exception as e:
            recent_logs.append(f"[ERROR] ログ読み込み失敗: {e}")

        return recent_logs

    def analyze_log_patterns(self) -> Dict[str, Any]:
        """ログパターン分析"""
        analysis = {
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0,
            "services": {},
            "error_patterns": [],
            "recent_errors": [],
        }

        recent_logs = self.get_recent_logs(hours=24)

        for log_line in recent_logs:
            # レベル統計
            if "[ERROR]" in log_line:
                analysis["error_count"] += 1
                analysis["recent_errors"].append(log_line)
            elif "[WARNING]" in log_line:
                analysis["warning_count"] += 1
            elif "[INFO]" in log_line:
                analysis["info_count"] += 1

            # サービス別統計
            for service in self.active_services:
                if f"[{service}]" in log_line:
                    if service not in analysis["services"]:
                        analysis["services"][service] = {"logs": 0, "errors": 0}
                    analysis["services"][service]["logs"] += 1
                    if "[ERROR]" in log_line:
                        analysis["services"][service]["errors"] += 1

        # エラーパターン分析（簡易版）
        error_messages = [log for log in recent_logs if "[ERROR]" in log]
        error_keywords = ["connection", "timeout", "failed", "exception", "error"]

        for keyword in error_keywords:
            count = sum(1 for msg in error_messages if keyword.lower() in msg.lower())
            if count > 0:
                analysis["error_patterns"].append({"keyword": keyword, "count": count})

        return analysis

    def cleanup_old_logs(self, days: int = 7):
        """古いログファイルのクリーンアップ"""
        cutoff_time = datetime.now() - timedelta(days=days)
        cleaned_files = []

        for log_file in self.log_dir.glob("*.log"):
            try:
                if log_file.stat().st_mtime < cutoff_time.timestamp():
                    log_file.unlink()
                    cleaned_files.append(str(log_file))
            except Exception as e:
                self.write_centralized_log(
                    f"ログクリーンアップ失敗 {log_file}: {e}", "WARNING"
                )

        if cleaned_files:
            self.write_centralized_log(
                f"古いログファイル削除: {len(cleaned_files)}件", "INFO"
            )

        return cleaned_files

    def generate_log_report(self) -> Dict[str, Any]:
        """ログレポート生成"""
        report = {
            "report_time": datetime.now(),
            "active_services": len(self.active_services),
            "log_analysis": self.analyze_log_patterns(),
            "log_files": {},
            "disk_usage": {},
        }

        # ログファイル情報
        for service, log_file in self.log_files.items():
            if log_file.exists():
                stat = log_file.stat()
                report["log_files"][service] = {
                    "path": str(log_file),
                    "size_mb": stat.st_size / 1024 / 1024,
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                }

        # ディスク使用量
        total_size = sum(
            f.stat().st_size for f in self.log_dir.glob("*.log") if f.exists()
        )
        report["disk_usage"] = {
            "total_mb": total_size / 1024 / 1024,
            "file_count": len(list(self.log_dir.glob("*.log"))),
        }

        return report


# グローバルインスタンス
centralized_logger = CentralizedLogger()


def get_centralized_logger() -> CentralizedLogger:
    """集約ログ管理インスタンス取得"""
    return centralized_logger
