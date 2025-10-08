"""ClStock システム監視・パフォーマンス監視統合モジュール
"""

import os
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import get_settings
from utils.logger_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class SystemMetrics:
    """システムメトリクス"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    active_processes: int
    load_average: Optional[float] = None  # Unix系のみ


@dataclass
class ProcessMetrics:
    """プロセスメトリクス"""

    timestamp: datetime
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    status: str
    create_time: datetime
    num_threads: int
    num_fds: Optional[int] = None  # Unix系のみ


@dataclass
class PerformanceAlert:
    """パフォーマンスアラート"""

    timestamp: datetime
    level: str  # INFO, WARNING, CRITICAL
    category: str  # CPU, MEMORY, DISK, PROCESS
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class SystemMonitor:
    """システム監視クラス"""

    def __init__(self, max_history_points: int = 1000):
        self.max_history_points = max_history_points
        self.system_metrics_history: deque = deque(maxlen=max_history_points)
        self.process_metrics_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=100),
        )
        self.alerts: deque = deque(maxlen=500)

        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # パフォーマンス閾値設定
        self.cpu_warning_threshold = settings.process.cpu_warning_threshold_percent
        self.memory_warning_threshold = settings.process.memory_warning_threshold_mb
        self.disk_warning_threshold = 85.0  # %

        # 統計情報
        self.start_time = datetime.now()
        self.total_alerts = 0

    def start_monitoring(self, interval_seconds: int = 5):
        """監視開始"""
        if self.monitoring_active:
            logger.warning("監視は既に有効です")
            return

        self.monitoring_active = True
        self._shutdown_event.clear()

        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True,
        )
        self.monitor_thread.start()

        logger.info(f"システム監視開始 (間隔: {interval_seconds}秒)")

    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        self._shutdown_event.set()

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)

        logger.info("システム監視停止")

    def _monitoring_loop(self, interval_seconds: int):
        """監視メインループ"""
        while self.monitoring_active and not self._shutdown_event.is_set():
            try:
                # システムメトリクス収集
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)

                # プロセスメトリクス収集
                self._collect_process_metrics()

                # アラートチェック
                self._check_system_health(system_metrics)

                # 待機
                self._shutdown_event.wait(interval_seconds)

            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                self._shutdown_event.wait(interval_seconds * 2)

    def _collect_system_metrics(self) -> SystemMetrics:
        """システムメトリクス収集"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)

            # メモリ情報
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / 1024 / 1024

            # ディスク情報
            disk = psutil.disk_usage("/")
            disk_usage_percent = disk.percent
            disk_free_gb = disk.free / 1024 / 1024 / 1024

            # ネットワーク情報
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / 1024 / 1024
            network_recv_mb = network.bytes_recv / 1024 / 1024

            # プロセス数
            active_processes = len(psutil.pids())

            # ロードアベレージ（Unix系のみ）
            load_average = None
            if hasattr(os, "getloadavg"):
                load_average = os.getloadavg()[0]

            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                active_processes=active_processes,
                load_average=load_average,
            )

        except Exception as e:
            logger.error(f"システムメトリクス収集エラー: {e}")
            # 障害時でも監視ループを継続できるように安全なデフォルト値を返す
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                active_processes=0,
                load_average=None,
            )

    def _collect_process_metrics(self):
        """プロセスメトリクス収集"""
        try:
            current_time = datetime.now()

            for proc in psutil.process_iter(
                [
                    "pid",
                    "name",
                    "cpu_percent",
                    "memory_percent",
                    "memory_info",
                    "status",
                    "create_time",
                    "num_threads",
                ],
            ):
                try:
                    proc_info = proc.info
                    pid = proc_info["pid"]

                    # メモリ情報
                    memory_info = proc_info["memory_info"]
                    memory_rss_mb = memory_info.rss / 1024 / 1024
                    memory_vms_mb = memory_info.vms / 1024 / 1024

                    # 作成時間
                    create_time = datetime.fromtimestamp(proc_info["create_time"])

                    # ファイルディスクリプタ数（Unix系のみ）
                    num_fds = None
                    try:
                        if hasattr(proc, "num_fds"):
                            num_fds = proc.num_fds()
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        pass

                    metrics = ProcessMetrics(
                        timestamp=current_time,
                        pid=pid,
                        name=proc_info["name"],
                        cpu_percent=proc_info["cpu_percent"] or 0.0,
                        memory_percent=proc_info["memory_percent"] or 0.0,
                        memory_rss_mb=memory_rss_mb,
                        memory_vms_mb=memory_vms_mb,
                        status=proc_info["status"],
                        create_time=create_time,
                        num_threads=proc_info["num_threads"],
                        num_fds=num_fds,
                    )

                    self.process_metrics_history[pid].append(metrics)

                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    continue

        except Exception as e:
            logger.error(f"プロセスメトリクス収集エラー: {e}")

    def _check_system_health(self, metrics: SystemMetrics):
        """システムヘルスチェック"""
        alerts = []

        # CPU使用率チェック
        if metrics.cpu_percent > self.cpu_warning_threshold:
            level = "CRITICAL" if metrics.cpu_percent > 90 else "WARNING"
            alerts.append(
                PerformanceAlert(
                    timestamp=metrics.timestamp,
                    level=level,
                    category="CPU",
                    message=f"高CPU使用率: {metrics.cpu_percent:.1f}%",
                    details={"cpu_percent": metrics.cpu_percent},
                ),
            )

        # メモリ使用率チェック
        if metrics.memory_percent > 80:
            level = "CRITICAL" if metrics.memory_percent > 95 else "WARNING"
            alerts.append(
                PerformanceAlert(
                    timestamp=metrics.timestamp,
                    level=level,
                    category="MEMORY",
                    message=f"高メモリ使用率: {metrics.memory_percent:.1f}%",
                    details={
                        "memory_percent": metrics.memory_percent,
                        "memory_available_mb": metrics.memory_available_mb,
                    },
                ),
            )

        # ディスク使用率チェック
        if metrics.disk_usage_percent > self.disk_warning_threshold:
            level = "CRITICAL" if metrics.disk_usage_percent > 95 else "WARNING"
            alerts.append(
                PerformanceAlert(
                    timestamp=metrics.timestamp,
                    level=level,
                    category="DISK",
                    message=f"高ディスク使用率: {metrics.disk_usage_percent:.1f}%",
                    details={
                        "disk_usage_percent": metrics.disk_usage_percent,
                        "disk_free_gb": metrics.disk_free_gb,
                    },
                ),
            )

        # アラート登録
        for alert in alerts:
            self._add_alert(alert)

    def _add_alert(self, alert: PerformanceAlert):
        """アラート追加"""
        self.alerts.append(alert)
        self.total_alerts += 1

        # ログ出力
        if alert.level == "CRITICAL":
            logger.error(f"🚨 {alert.category}: {alert.message}")
        elif alert.level == "WARNING":
            logger.warning(f"⚠️  {alert.category}: {alert.message}")
        else:
            logger.info(f"ℹ️  {alert.category}: {alert.message}")

    def get_current_system_status(self) -> Dict[str, Any]:
        """現在のシステム状態取得"""
        if not self.system_metrics_history:
            return {"status": "no_data", "message": "監視データなし"}

        latest = self.system_metrics_history[-1]
        recent_alerts = [
            a
            for a in self.alerts
            if (datetime.now() - a.timestamp).total_seconds() < 300
        ]  # 5分以内

        return {
            "status": "ok" if not recent_alerts else "warning",
            "timestamp": latest.timestamp,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "monitoring_active": self.monitoring_active,
            "system": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "memory_available_mb": latest.memory_available_mb,
                "disk_usage_percent": latest.disk_usage_percent,
                "disk_free_gb": latest.disk_free_gb,
                "active_processes": latest.active_processes,
                "load_average": latest.load_average,
            },
            "alerts": {
                "total": self.total_alerts,
                "recent": len(recent_alerts),
                "latest": [
                    {
                        "level": a.level,
                        "category": a.category,
                        "message": a.message,
                        "timestamp": a.timestamp,
                    }
                    for a in list(self.alerts)[-5:]  # 最新5件
                ],
            },
        }

    def get_process_summary(self) -> List[Dict[str, Any]]:
        """プロセスサマリー取得"""
        process_summary = []

        # 最新のプロセス情報をCPU使用率順で取得
        latest_processes = {}
        for pid, metrics_list in self.process_metrics_history.items():
            if metrics_list:
                latest_processes[pid] = metrics_list[-1]

        # CPU使用率順でソート
        sorted_processes = sorted(
            latest_processes.values(),
            key=lambda x: x.cpu_percent,
            reverse=True,
        )

        for proc in sorted_processes[:20]:  # トップ20
            process_summary.append(
                {
                    "pid": proc.pid,
                    "name": proc.name,
                    "cpu_percent": proc.cpu_percent,
                    "memory_percent": proc.memory_percent,
                    "memory_rss_mb": proc.memory_rss_mb,
                    "status": proc.status,
                    "num_threads": proc.num_threads,
                    "uptime_hours": (datetime.now() - proc.create_time).total_seconds()
                    / 3600,
                },
            )

        return process_summary

    def get_performance_trends(self, hours: int = 1) -> Dict[str, List]:
        """パフォーマンストレンド取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # 指定時間内のメトリクスをフィルタ
        recent_metrics = [
            m for m in self.system_metrics_history if m.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {"cpu": [], "memory": [], "disk": [], "timestamps": []}

        return {
            "cpu": [m.cpu_percent for m in recent_metrics],
            "memory": [m.memory_percent for m in recent_metrics],
            "disk": [m.disk_usage_percent for m in recent_metrics],
            "timestamps": [m.timestamp.isoformat() for m in recent_metrics],
        }

    def generate_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポート生成"""
        if not self.system_metrics_history:
            return {"error": "監視データなし"}

        # 統計計算
        cpu_values = [m.cpu_percent for m in self.system_metrics_history]
        memory_values = [m.memory_percent for m in self.system_metrics_history]

        def stats(values):
            if not values:
                return {"min": 0, "max": 0, "avg": 0}
            return {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }

        return {
            "report_time": datetime.now(),
            "monitoring_duration_hours": (
                datetime.now() - self.start_time
            ).total_seconds()
            / 3600,
            "data_points": len(self.system_metrics_history),
            "cpu_stats": stats(cpu_values),
            "memory_stats": stats(memory_values),
            "total_alerts": self.total_alerts,
            "alert_breakdown": {
                level: len([a for a in self.alerts if a.level == level])
                for level in ["INFO", "WARNING", "CRITICAL"]
            },
            "top_processes": self.get_process_summary()[:10],
        }


# グローバルインスタンス
system_monitor = SystemMonitor()


def get_system_monitor() -> SystemMonitor:
    """システム監視インスタンス取得"""
    return system_monitor


if __name__ == "__main__":
    # デモ実行
    monitor = get_system_monitor()

    print("=== ClStock システム監視 ===")
    print("監視開始...")

    monitor.start_monitoring(interval_seconds=2)

    try:
        while True:
            time.sleep(10)

            # 現在の状態表示
            status = monitor.get_current_system_status()
            print(f"\n--- {status['timestamp'].strftime('%H:%M:%S')} ---")
            print(f"CPU: {status['system']['cpu_percent']:.1f}%")
            print(f"メモリ: {status['system']['memory_percent']:.1f}%")
            print(f"ディスク: {status['system']['disk_usage_percent']:.1f}%")
            print(f"プロセス数: {status['system']['active_processes']}")
            print(f"アラート: {status['alerts']['recent']}件")

    except KeyboardInterrupt:
        print("\n監視停止...")
        monitor.stop_monitoring()

        # レポート生成
        report = monitor.generate_performance_report()
        print("\n=== パフォーマンスレポート ===")
        print(f"監視時間: {report['monitoring_duration_hours']:.1f}時間")
        print(f"データポイント: {report['data_points']}件")
        print(f"CPU平均: {report['cpu_stats']['avg']:.1f}%")
        print(f"メモリ平均: {report['memory_stats']['avg']:.1f}%")
        print(f"総アラート数: {report['total_alerts']}件")
