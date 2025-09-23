"""Cache management for models."""

import logging
import pickle
import json
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis互換のキャッシュシステム（ローカル実装）"""

    def __init__(self, cache_dir: str = "cache", default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl  # デフォルトTTL（秒）
        self.memory_cache: Dict[str, Dict[str, Any]] = {}

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """キーに値を設定"""
        try:
            ttl = ttl or self.default_ttl
            expiry = datetime.now() + timedelta(seconds=ttl)

            # メモリキャッシュに保存
            self.memory_cache[key] = {
                'value': value,
                'expiry': expiry
            }

            # ディスクにも永続化
            cache_file = self.cache_dir / f"{key}.cache"
            cache_data = {
                'value': value,
                'expiry': expiry.isoformat(),
                'created': datetime.now().isoformat()
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            return True

        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """キーから値を取得"""
        try:
            # メモリキャッシュから確認
            if key in self.memory_cache:
                cache_item = self.memory_cache[key]
                if datetime.now() < cache_item['expiry']:
                    return cache_item['value']
                else:
                    # 期限切れの場合は削除
                    del self.memory_cache[key]

            # ディスクキャッシュから確認
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)

                expiry = datetime.fromisoformat(cache_data['expiry'])
                if datetime.now() < expiry:
                    # メモリキャッシュにも復元
                    self.memory_cache[key] = {
                        'value': cache_data['value'],
                        'expiry': expiry
                    }
                    return cache_data['value']
                else:
                    # 期限切れファイルを削除
                    cache_file.unlink()

            return None

        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """キーを削除"""
        try:
            # メモリキャッシュから削除
            if key in self.memory_cache:
                del self.memory_cache[key]

            # ディスクキャッシュから削除
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                cache_file.unlink()

            return True

        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """キーが存在するかチェック"""
        return self.get(key) is not None

    def expire(self, key: str, ttl: int) -> bool:
        """キーのTTLを設定"""
        try:
            value = self.get(key)
            if value is not None:
                return self.set(key, value, ttl)
            return False

        except Exception as e:
            logger.error(f"Failed to set expiry for key {key}: {e}")
            return False

    def ttl(self, key: str) -> int:
        """キーの残りTTLを取得"""
        try:
            if key in self.memory_cache:
                expiry = self.memory_cache[key]['expiry']
                remaining = (expiry - datetime.now()).total_seconds()
                return max(0, int(remaining))

            # ディスクキャッシュから確認
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)

                expiry = datetime.fromisoformat(cache_data['expiry'])
                remaining = (expiry - datetime.now()).total_seconds()
                return max(0, int(remaining))

            return -1  # キーが存在しない

        except Exception as e:
            logger.error(f"Failed to get TTL for key {key}: {e}")
            return -1

    def keys(self, pattern: str = "*") -> list:
        """パターンにマッチするキー一覧を取得"""
        try:
            all_keys: set[str] = set()

            # メモリキャッシュから
            all_keys.update(self.memory_cache.keys())

            # ディスクキャッシュから
            for cache_file in self.cache_dir.glob("*.cache"):
                key = cache_file.stem
                all_keys.add(key)

            # パターンマッチング（簡単な実装）
            if pattern == "*":
                return list(all_keys)
            else:
                # 基本的なワイルドカード対応
                import fnmatch
                return [key for key in all_keys if fnmatch.fnmatch(key, pattern)]

        except Exception as e:
            logger.error(f"Failed to get keys with pattern {pattern}: {e}")
            return []

    def flushall(self) -> bool:
        """全てのキーを削除"""
        try:
            # メモリキャッシュをクリア
            self.memory_cache.clear()

            # ディスクキャッシュをクリア
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()

            return True

        except Exception as e:
            logger.error(f"Failed to flush all cache: {e}")
            return False

    def info(self) -> Dict[str, Any]:
        """キャッシュ情報を取得"""
        try:
            memory_keys = len(self.memory_cache)
            disk_files = len(list(self.cache_dir.glob("*.cache")))

            # キャッシュサイズ計算
            total_size = 0
            for cache_file in self.cache_dir.glob("*.cache"):
                total_size += cache_file.stat().st_size

            return {
                'memory_keys': memory_keys,
                'disk_files': disk_files,
                'total_size_bytes': total_size,
                'cache_directory': str(self.cache_dir),
                'default_ttl': self.default_ttl
            }

        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return {}

    def cleanup_expired(self) -> int:
        """期限切れキャッシュをクリーンアップ"""
        cleaned_count = 0

        try:
            # メモリキャッシュのクリーンアップ
            expired_memory_keys = []
            for key, cache_item in self.memory_cache.items():
                if datetime.now() >= cache_item['expiry']:
                    expired_memory_keys.append(key)

            for key in expired_memory_keys:
                del self.memory_cache[key]
                cleaned_count += 1

            # ディスクキャッシュのクリーンアップ
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)

                    expiry = datetime.fromisoformat(cache_data['expiry'])
                    if datetime.now() >= expiry:
                        cache_file.unlink()
                        cleaned_count += 1

                except Exception:
                    # 破損ファイルも削除
                    cache_file.unlink()
                    cleaned_count += 1

            logger.info(f"Cleaned up {cleaned_count} expired cache entries")
            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
            return cleaned_count

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        info = self.info()

        # ヒット率計算（簡易版）
        total_operations = getattr(self, '_total_operations', 0)
        cache_hits = getattr(self, '_cache_hits', 0)
        hit_rate = cache_hits / total_operations if total_operations > 0 else 0

        return {
            **info,
            'hit_rate': hit_rate,
            'total_operations': total_operations,
            'cache_hits': cache_hits
        }

    def _increment_stats(self, hit: bool = False):
        """統計カウンタを更新"""
        if not hasattr(self, '_total_operations'):
            self._total_operations = 0
            self._cache_hits = 0

        self._total_operations += 1
        if hit:
            self._cache_hits += 1

    def get_with_stats(self, key: str) -> Optional[Any]:
        """統計付きでキーから値を取得"""
        value = self.get(key)
        self._increment_stats(hit=(value is not None))
        return value