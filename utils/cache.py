"""
キャッシュ機能
データの重複取得を防ぎパフォーマンスを向上
"""

import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Callable
from functools import wraps
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataCache:
    """シンプルなファイルベースキャッシュ"""

    def __init__(self, cache_dir: str = "cache", default_ttl: int = 3600):
        """
        Args:
            cache_dir: キャッシュディレクトリ
            default_ttl: デフォルトTTL（秒）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl

    def _get_cache_key(self, *args, **kwargs) -> str:
        """キャッシュキーを生成"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """キャッシュファイルパスを取得"""
        return self.cache_dir / f"{cache_key}.cache"

    def get(self, cache_key: str) -> Optional[Any]:
        """キャッシュから値を取得"""
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # TTLチェック
            if time.time() > cache_data['expires_at']:
                cache_path.unlink()  # 期限切れファイルを削除
                return None

            logger.debug(f"Cache hit: {cache_key}")
            return cache_data['value']

        except Exception as e:
            logger.warning(f"Cache read error for {cache_key}: {e}")
            # 破損したキャッシュファイルを削除
            if cache_path.exists():
                cache_path.unlink()
            return None

    def set(self, cache_key: str, value: Any, ttl: Optional[int] = None) -> None:
        """値をキャッシュに保存"""
        if ttl is None:
            ttl = self.default_ttl

        cache_data = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }

        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.debug(f"Cache set: {cache_key} (TTL: {ttl}s)")
        except Exception as e:
            logger.warning(f"Cache write error for {cache_key}: {e}")

    def delete(self, cache_key: str) -> None:
        """キャッシュを削除"""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"Cache deleted: {cache_key}")

    def clear(self) -> None:
        """全キャッシュを削除"""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
        logger.info("All cache cleared")

    def cleanup_expired(self) -> int:
        """期限切れキャッシュを削除"""
        current_time = time.time()
        deleted_count = 0

        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)

                if current_time > cache_data['expires_at']:
                    cache_file.unlink()
                    deleted_count += 1

            except Exception:
                # 破損ファイルも削除
                cache_file.unlink()
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} expired cache files")

        return deleted_count


# グローバルキャッシュインスタンス
_cache = DataCache()


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


def clear_cache():
    """グローバルキャッシュをクリア"""
    _cache.clear()


def cleanup_cache():
    """期限切れキャッシュをクリーンアップ"""
    return _cache.cleanup_expired()