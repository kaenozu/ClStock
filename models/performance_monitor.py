"""モデル実行時のリソース使用量を監視するモジュール"""

import functools
import time

import psutil


class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.end_time = None

    def start_monitoring(self):
        self.start_time = time.time()

    def stop_monitoring(self):
        self.end_time = time.time()

    def get_cpu_usage(self):
        """CPU使用率を取得"""
        return self.process.cpu_percent(interval=1)

    def get_memory_usage(self):
        """メモリ使用量を取得 (MB)"""
        memory_info = self.process.memory_info()
        return memory_info.rss / 1024 / 1024  # MBに変換

    def get_execution_time(self):
        """実行時間を取得 (秒)"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


def monitor_resources(func):
    """関数の実行時にリソース使用量を監視するデコレーター"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        result = func(*args, **kwargs)
        monitor.stop_monitoring()

        print(f"Function {func.__name__} resource usage:")
        print(f"  CPU Usage: {monitor.get_cpu_usage()}%")
        print(f"  Memory Usage: {monitor.get_memory_usage():.2f} MB")
        print(f"  Execution Time: {monitor.get_execution_time():.2f} seconds")

        return result

    return wrapper
