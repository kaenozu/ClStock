"""Cache management utilities."""

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class AdvancedCacheManager:
    """高度なキャッシュ管理システム"""

    def __init__(self):
        self.feature_cache = {}
        self.prediction_cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "feature_cache_size": 0,
            "prediction_cache_size": 0,
        }

    def get_cached_features(
        self, symbol: str, data_hash: str,
    ) -> Optional[pd.DataFrame]:
        """特徴量キャッシュから取得"""
        cache_key = f"{symbol}_{data_hash}"
        if cache_key in self.feature_cache:
            self.cache_stats["hits"] += 1
            return self.feature_cache[cache_key]
        self.cache_stats["misses"] += 1
        return None

    def cache_features(self, symbol: str, data_hash: str, features: pd.DataFrame):
        """特徴量をキャッシュ"""
        cache_key = f"{symbol}_{data_hash}"
        self.feature_cache[cache_key] = features
        self.cache_stats["feature_cache_size"] = len(self.feature_cache)

    def get_cached_prediction(self, symbol: str, features_hash: str) -> Optional[float]:
        """予測結果キャッシュから取得"""
        cache_key = f"{symbol}_{features_hash}"
        if cache_key in self.prediction_cache:
            self.cache_stats["hits"] += 1
            return self.prediction_cache[cache_key]
        self.cache_stats["misses"] += 1
        return None

    def cache_prediction(self, symbol: str, features_hash: str, prediction: float):
        """予測結果をキャッシュ"""
        cache_key = f"{symbol}_{features_hash}"
        self.prediction_cache[cache_key] = prediction
        self.cache_stats["prediction_cache_size"] = len(self.prediction_cache)

    def cleanup_old_cache(self, max_size: int = 1000):
        """古いキャッシュをクリーンアップ"""
        if len(self.feature_cache) > max_size:
            # 最も古いエントリを削除（簡単なLRU実装）
            keys_to_remove = list(self.feature_cache.keys())[:-max_size]
            for key in keys_to_remove:
                del self.feature_cache[key]
        if len(self.prediction_cache) > max_size:
            keys_to_remove = list(self.prediction_cache.keys())[:-max_size]
            for key in keys_to_remove:
                del self.prediction_cache[key]
        self.cache_stats["feature_cache_size"] = len(self.feature_cache)
        self.cache_stats["prediction_cache_size"] = len(self.prediction_cache)

    def get_cache_stats(self) -> Dict:
        """キャッシュ統計を取得"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        )
        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }


class RedisCache:
    """Redis高速キャッシュシステム"""

    def __init__(self, host="localhost", port=6379, db=0):
        try:
            import redis

            self.redis_client = redis.Redis(
                host=host, port=port, db=db, decode_responses=True, socket_timeout=5,
            )
            # 接続テスト
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.warning(
                f"Redis not available, falling back to memory cache: {e!s}",
            )
            self.redis_available = False
            self.memory_cache = {}

    def get(self, key: str) -> Optional[str]:
        """キャッシュ取得"""
        try:
            if self.redis_available:
                return self.redis_client.get(key)
            return self.memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e!s}")
            return None

    def set(self, key: str, value: str, ttl: int = 3600):
        """キャッシュ設定"""
        try:
            if self.redis_available:
                self.redis_client.setex(key, ttl, value)
            else:
                self.memory_cache[key] = value
                # メモリキャッシュのTTL管理は簡略化
        except Exception as e:
            logger.error(f"Cache set error: {e!s}")

    def get_json(self, key: str) -> Optional[Dict]:
        """JSON形式でキャッシュ取得"""
        try:
            data = self.get(key)
            if data:
                import json

                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Cache JSON get error: {e!s}")
            return None

    def set_json(self, key: str, value: Dict, ttl: int = 3600):
        """JSON形式でキャッシュ設定"""
        try:
            import json

            self.set(key, json.dumps(value), ttl)
        except Exception as e:
            logger.error(f"Cache JSON set error: {e!s}")
