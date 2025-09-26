"""
キャッシュ機能
データの重複取得を防ぎパフォーマンスを向上
"""

import pickle
import hashlib
import time
import threading
from pathlib import Path
from typing import Any, Optional, Callable, Dict
from functools import wraps
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataCache:
    """ファイルベースのデータキャッシュ"""

    def __init__(
        self,
        cache_dir: str = "cache",
        default_ttl: int = 3600,
        max_size: Optional[int] = None,
        auto_cleanup: bool = True,
        cleanup_interval: int = 3600,
    ):
        """
        Args:
            cache_dir: キャッシュディレクトリ
            default_ttl: デフォルトTTL（秒）
            max_size: 最大キャッシュサイズ（エントリ数）
            auto_cleanup: 自動クリーンアップを有効化
            cleanup_interval: 自動クリーンアップ間隔（秒）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval = cleanup_interval
        self._shutdown_event = None
        self.cleanup_thread = None
        self._lock = threading.RLock()  # スレッドセーフにするためのロック

        if self.auto_cleanup:
            self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """自動クリーンアップスレッドを開始"""

        self._shutdown_event = threading.Event()
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        logger.info(
            f"Automatic cache cleanup thread started with interval {self.cleanup_interval}s"
        )

    def _cleanup_worker(self):
        """クリーンアップワーカースレッド"""
        import time

        while self._shutdown_event and not self._shutdown_event.is_set():
            try:
                # クリーンアップ間隔待機
                if self._shutdown_event.wait(self.cleanup_interval):
                    break

                # 自動クリーンアップ実行
                self.cleanup_expired()

            except Exception as e:
                logger.error(f"Cache cleanup worker error: {e}")
                # エラーでも継続

        logger.info("Cache cleanup worker stopped")

    def shutdown(self):
        """キャッシュをシャットダウン"""
        if self._shutdown_event:
            self._shutdown_event.set()

        # クリーンアップスレッドの終了を待機
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)  # 5秒でタイムアウト
            if self.cleanup_thread.is_alive():
                logger.warning("Cache cleanup thread did not terminate gracefully")

        logger.info("DataCache shutdown completed")

    def _get_cache_key(self, *args, **kwargs) -> str:
        """キャッシュキーを生成"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def _get_cache_path(self, cache_key: str) -> Path:
        """キャッシュファイルパスを取得"""
        return self.cache_dir / f"{cache_key}.cache"
    
    def _validate_cache_entry(self, cache_data: Dict[str, Any], cache_key: str) -> bool:
        """キャッシュエントリーの検証"""
        required_keys = ["value", "expires_at", "created_at"]
        for key in required_keys:
            if key not in cache_data:
                logger.warning(f"Cache entry missing required key '{key}' for key {cache_key}")
                return False
        
        if not isinstance(cache_data["expires_at"], (int, float)):
            logger.warning(f"Invalid expires_at type for key {cache_key}")
            return False
        
        return True

    def get(self, cache_key: str) -> Optional[Any]:
        """キャッシュから値を取得"""
        cache_path = self._get_cache_path(cache_key)

        with self._lock:  # スレッドセーフにアクセス
            if not cache_path.exists():
                return None

            try:
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)

                # キャッシュエントリーの検証
                if not self._validate_cache_entry(cache_data, cache_key):
                    cache_path.unlink()
                    return None

                # TTLチェック
                if time.time() > cache_data["expires_at"]:
                    cache_path.unlink()  # 期限切れファイルを削除
                    return None

                logger.debug(f"Cache hit: {cache_key}")
                return cache_data["value"]

            except Exception as e:
                logger.warning(f"Cache read error for {cache_key}: {e}")
                # 破損したキャッシュファイルを削除
                if cache_path.exists():
                    cache_path.unlink()
                return None

    def set(self, cache_key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """値をキャッシュに保存"""
        if ttl is None:
            ttl = self.default_ttl

        cache_data = {
            "value": value,
            "expires_at": time.time() + ttl,
            "created_at": time.time(),
            "last_access": time.time(),
        }

        cache_path = self._get_cache_path(cache_key)

        with self._lock:  # スレッドセーフにアクセス
            try:
                # 最大サイズ制限の確認
                if self.max_size:
                    current_files = list(self.cache_dir.glob("*.cache"))
                    if len(current_files) >= self.max_size:
                        # LRU (Least Recently Used) で削除
                        oldest_file = min(current_files, key=lambda f: f.stat().st_mtime)
                        oldest_file.unlink()
                        logger.debug(f"Removed oldest cache entry: {oldest_file.name}")

                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                logger.debug(f"Cache set: {cache_key} (TTL: {ttl}s)")
                return True
            except Exception as e:
                logger.warning(f"Cache write error for {cache_key}: {e}")
                return False

    def delete(self, cache_key: str) -> bool:
        """キャッシュを削除"""
        cache_path = self._get_cache_path(cache_key)
        with self._lock:  # スレッドセーフにアクセス
            if cache_path.exists():
                cache_path.unlink()
                logger.debug(f"Cache deleted: {cache_key}")
                return True
            return False

    def clear(self) -> int:
        """全キャッシュを削除"""
        count = 0
        with self._lock:  # スレッドセーフにアクセス
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
                count += 1
        logger.info(f"All cache cleared: {count} files")
        return count

    def cleanup_expired(self) -> int:
        """期限切れキャッシュを削除"""
        current_time = time.time()
        deleted_count = 0

        with self._lock:  # スレッドセーフにアクセス
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, "rb") as f:
                        cache_data = pickle.load(f)

                    # キャッシュエントリーの検証
                    if not self._validate_cache_entry(cache_data, cache_file.stem):
                        cache_file.unlink()
                        deleted_count += 1
                        continue

                    if current_time > cache_data["expires_at"]:
                        cache_file.unlink()
                        deleted_count += 1

                except Exception:
                    # 破損ファイルも削除
                    cache_file.unlink()
                    deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} expired/invalid cache files")

        return deleted_count
    
    def get_stats(self) -> Dict[str, Union[int, float, bool]]:
        """キャッシュ統計情報を取得"""
        with self._lock:  # スレッドセーフにアクセス
            files = list(self.cache_dir.glob("*.cache"))
            total_size = sum(f.stat().st_size for f in files)
            
            # 有効期限切れのファイル数をカウント
            expired_count = 0
            current_time = time.time()
            for f in files:
                try:
                    with open(f, "rb") as file:
                        cache_data = pickle.load(file)
                    if current_time > cache_data.get("expires_at", 0):
                        expired_count += 1
                except:
                    # 読み取りエラーも無効なファイルとしてカウント
                    expired_count += 1
            
            return {
                "total_files": len(files),
                "valid_files": len(files) - expired_count,
                "expired_files": expired_count,
                "total_size_bytes": total_size,
                "max_size_limit": self.max_size,
                "auto_cleanup_enabled": self.auto_cleanup,
            }


# グローバルキャッシュインスタンス（30分間隔で自動クリーンアップ）
_cache = DataCache(auto_cleanup=True, cleanup_interval=1800, max_size=1000)


def cached(ttl: Optional[int] = None, cache_instance: Optional[DataCache] = None):
    """
    関数結果をキャッシュするデコレータ

    Args:
        ttl: TTL（秒）
        cache_instance: 使用するキャッシュインスタンス
    """

    def decorator(func: Callable) -> Callable:
        cache = cache_instance or _cache

        @wraps(func)
        def wrapper(*args, **kwargs):
            # キャッシュキーを生成
            func_name = f"{func.__module__}.{func.__name__}"
            cache_key = cache._get_cache_key(func_name, *args, **kwargs)

            # キャッシュから取得を試行
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # 関数を実行してキャッシュに保存
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


def cache_dataframe(ttl: int = 1800):  # 30分
    """
    DataFrameのキャッシュ用デコレータ
    """
    return cached(ttl=ttl)


def get_cache() -> DataCache:
    """グローバルキャッシュインスタンスを取得"""
    return _cache


def clear_cache() -> int:
    """グローバルキャッシュをクリア"""
    return _cache.clear()


def cleanup_cache() -> int:
    """期限切れキャッシュをクリーンアップ"""
    return _cache.cleanup_expired()


def get_cache_stats() -> Dict[str, Union[int, float, bool]]:
    """キャッシュ統計情報を取得"""
    return _cache.get_stats()


def shutdown_cache():
    """キャッシュをシャットダウン"""
    _cache.shutdown()
