#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
インテリジェント予測キャッシュシステム
90%の予測で即座応答を実現する高度キャッシングシステム
"""

import time
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple
from functools import lru_cache
import numpy as np
import pandas as pd
import threading

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..base.interfaces import PredictionResult
from .prediction_modes import PredictionMode


class MarketVolatilityCalculator:
    """市場ボラティリティ計算器"""

    def __init__(self):
        self.volatility_cache = {}
        self.cache_timeout = 300  # 5分キャッシュ

    def get_market_volatility(self, symbol: str = None) -> float:
        """市場ボラティリティ取得"""
        cache_key = symbol or "market_overall"
        current_time = time.time()

        # キャッシュチェック
        if cache_key in self.volatility_cache:
            cached_data, timestamp = self.volatility_cache[cache_key]
            if current_time - timestamp < self.cache_timeout:
                return cached_data

        # ボラティリティ計算
        volatility = self._calculate_volatility(symbol)
        self.volatility_cache[cache_key] = (volatility, current_time)

        return volatility

    def _calculate_volatility(self, symbol: str = None) -> float:
        """ボラティリティ計算（簡易版）"""
        try:
            # 現在時刻ベースの疑似ボラティリティ
            current_hour = datetime.now().hour

            # 市場時間帯でボラティリティが高い
            if 9 <= current_hour <= 15:  # 市場時間
                base_volatility = 0.8
            elif 15 < current_hour <= 17:  # アフターマーケット
                base_volatility = 0.6
            else:  # 夜間・早朝
                base_volatility = 0.3

            # ランダム要素追加（実際は実データを使用）
            noise = np.random.normal(0, 0.1)
            volatility = max(0.1, min(1.0, base_volatility + noise))

            return volatility

        except Exception:
            return 0.5  # デフォルトボラティリティ


class AdaptiveCacheStrategy:
    """アダプティブキャッシュ戦略"""

    def __init__(self):
        self.volatility_calculator = MarketVolatilityCalculator()
        self.base_ttl = {
            PredictionMode.SPEED_PRIORITY: 30,  # 30秒
            PredictionMode.ACCURACY_PRIORITY: 300,  # 5分
            PredictionMode.BALANCED: 120,  # 2分
            PredictionMode.AUTO: 60,  # 1分
        }

    def calculate_ttl(self, symbol: str, mode: PredictionMode) -> int:
        """動的TTL計算"""
        # 基本TTL取得
        base_ttl = self.base_ttl.get(mode, 60)

        # 市場ボラティリティ取得
        volatility = self.volatility_calculator.get_market_volatility(symbol)

        # ボラティリティに基づくTTL調整
        # 高ボラティリティ → 短いTTL
        # 低ボラティリティ → 長いTTL
        volatility_factor = 2.0 - volatility  # 0.8-1.2の範囲
        adjusted_ttl = int(base_ttl * volatility_factor)

        # TTL範囲制限
        min_ttl = 10  # 最小10秒
        max_ttl = base_ttl * 3  # 最大3倍

        return max(min_ttl, min(max_ttl, adjusted_ttl))


class CacheInvalidationEngine:
    """キャッシュ無効化エンジン"""

    def __init__(self):
        self.last_market_check = {}
        self.significant_change_threshold = 0.02  # 2%の変化で無効化

    def should_invalidate(self, symbol: str) -> bool:
        """キャッシュ無効化判定"""
        try:
            # 市場大幅変動検出（簡易版）
            current_time = time.time()

            # 30秒以内のチェックはスキップ
            if symbol in self.last_market_check:
                if current_time - self.last_market_check[symbol] < 30:
                    return False

            self.last_market_check[symbol] = current_time

            # 疑似的な市場変動チェック
            # 実際は価格データAPIから取得
            random_change = np.random.normal(0, 0.01)
            significant_change = abs(random_change) > self.significant_change_threshold

            return significant_change

        except Exception:
            return False  # エラー時は無効化しない


class IntelligentPredictionCache:
    """
    インテリジェント予測キャッシュシステム

    特徴:
    - Redis + インメモリ二重キャッシュ
    - 市場ボラティリティ連動TTL
    - 市場変動による動的無効化
    - 90%以上のキャッシュヒット率目標
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        cleanup_interval: int = 300,
        max_cache_size: int = 1000,
    ):
        self.logger = logging.getLogger(__name__)

        # Redis接続（利用可能な場合）
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                # 接続テスト
                self.redis_client.ping()
                self.logger.info("Redis cache connected successfully")
            except Exception as e:
                self.logger.warning(
                    f"Redis connection failed: {str(e)}, using in-memory cache"
                )
                self.redis_client = None

        # インメモリキャッシュ（フォールバック）
        self.memory_cache = {}
        self.cache_timestamps = {}

        # キャッシュ戦略
        self.cache_strategy = AdaptiveCacheStrategy()
        self.invalidation_engine = CacheInvalidationEngine()

        # 統計情報
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "invalidations": 0,
            "redis_hits": 0,
            "memory_hits": 0,
        }

        # 自動クリーンアップ設定
        self.auto_cleanup_enabled = True
        self.cleanup_interval = (
            cleanup_interval  # 5分ごとにクリーンアップ（デフォルト）
        )
        self.max_cache_size = max_cache_size  # 最大キャッシュサイズ
        self._shutdown_event = threading.Event()
        self.cleanup_thread = None
        self._start_cleanup_thread()

        self.logger.info(
            f"IntelligentPredictionCache initialized with cleanup_interval={cleanup_interval}s, max_cache_size={max_cache_size}"
        )

    def _start_cleanup_thread(self):
        """自動クリーンアップスレッドを開始"""
        if self.auto_cleanup_enabled and self.cleanup_thread is None:
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_worker, daemon=True
            )
            self.cleanup_thread.start()
            self.logger.info("Automatic cache cleanup thread started")

    def _cleanup_worker(self):
        """クリーンアップワーカースレッド"""
        while not self._shutdown_event.is_set():
            try:
                # クリーンアップ間隔待機
                if self._shutdown_event.wait(self.cleanup_interval):
                    break

                # 自動クリーンアップ実行
                self._perform_auto_cleanup()

            except Exception as e:
                self.logger.error(f"Cache cleanup worker error: {e}")
                # エラーでも継続

        self.logger.info("Cache cleanup worker stopped")

    def _perform_auto_cleanup(self):
        """自動クリーンアップを実行"""
        try:
            # 期限切れキャッシュのクリーンアップ
            self._cleanup_memory_cache()

            # サイズ制限による追加クリーンアップ
            if len(self.memory_cache) > self.max_cache_size:
                self._cleanup_by_size_limit()

            self.logger.debug(
                f"Cache cleanup completed. Memory cache size: {len(self.memory_cache)}"
            )

        except Exception as e:
            self.logger.error(f"Auto cache cleanup failed: {e}")

    def shutdown(self):
        """キャッシュをシャットダウン"""
        self.logger.info("Shutting down intelligent prediction cache")
        self._shutdown_event.set()

        # クリーンアップスレッドの終了を待機
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)  # 5秒でタイムアウト
            if self.cleanup_thread.is_alive():
                self.logger.warning("Cache cleanup thread did not terminate gracefully")

        # Redisキャッシュをクリア
        if self.redis_client:
            try:
                keys = self.redis_client.keys("pred:*")
                if keys:
                    self.redis_client.delete(*keys)
                self.logger.info("Redis cache cleared during shutdown")
            except Exception as e:
                self.logger.error(f"Error clearing Redis cache during shutdown: {e}")

    def _cleanup_by_size_limit(self):
        """サイズ制限に基づくキャッシュクリーンアップ"""
        # タイムスタンプでソートして古いものから削除
        sorted_items = sorted(
            self.cache_timestamps.items(), key=lambda x: x[1][0]  # timestampでソート
        )

        # サイズ制限に合わせて削除
        items_to_remove = len(self.memory_cache) - self.max_cache_size
        removed_count = 0
        for i in range(min(items_to_remove, len(sorted_items))):
            key_to_remove = sorted_items[i][0]
            self.memory_cache.pop(key_to_remove, None)
            self.cache_timestamps.pop(key_to_remove, None)
            removed_count += 1

        self.logger.info(
            f"Removed {removed_count} oldest cache entries due to size limit"
        )
        return removed_count

    def get_cache_key(
        self, symbol: str, mode: PredictionMode, data_hash: str = None
    ) -> str:
        """キャッシュキー生成"""
        if data_hash is None:
            data_hash = self._generate_data_hash(symbol)

        return f"pred:{symbol}:{mode.value}:{data_hash}"

    def _generate_data_hash(self, symbol: str) -> str:
        """データハッシュ生成（簡易版）"""
        # 実際は最新の市場データのハッシュを使用
        current_minute = datetime.now().strftime("%Y%m%d%H%M")
        hash_input = f"{symbol}:{current_minute}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:8]

    def get_cached_prediction(
        self, symbol: str, mode: PredictionMode
    ) -> Optional[PredictionResult]:
        """キャッシュから予測結果取得"""
        cache_key = self.get_cache_key(symbol, mode)

        # 無効化チェック
        if self.invalidation_engine.should_invalidate(symbol):
            self._invalidate_symbol(symbol)
            self.cache_stats["invalidations"] += 1
            return None

        # Redis優先でキャッシュチェック
        cached_result = self._get_from_redis(cache_key)
        if cached_result is not None:
            self.cache_stats["hits"] += 1
            self.cache_stats["redis_hits"] += 1
            return cached_result

        # インメモリキャッシュチェック
        cached_result = self._get_from_memory(cache_key)
        if cached_result is not None:
            self.cache_stats["hits"] += 1
            self.cache_stats["memory_hits"] += 1
            return cached_result

        self.cache_stats["misses"] += 1
        return None

    def cache_prediction(
        self, symbol: str, mode: PredictionMode, result: PredictionResult
    ):
        """予測結果をキャッシュ"""
        cache_key = self.get_cache_key(symbol, mode)
        ttl = self.cache_strategy.calculate_ttl(symbol, mode)

        # Redisにキャッシュ
        self._set_to_redis(cache_key, result, ttl)

        # インメモリにもキャッシュ
        self._set_to_memory(cache_key, result, ttl)

        self.logger.debug(
            f"Cached prediction for {symbol} ({mode.value}) with TTL {ttl}s"
        )

    def _get_from_redis(self, cache_key: str) -> Optional[PredictionResult]:
        """Redisからデータ取得"""
        if not self.redis_client:
            return None

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            self.logger.warning(f"Redis get error: {str(e)}")

        return None

    def _set_to_redis(self, cache_key: str, result: PredictionResult, ttl: int):
        """Redisにデータ設定"""
        if not self.redis_client:
            return

        try:
            serialized_data = pickle.dumps(result)
            self.redis_client.setex(cache_key, ttl, serialized_data)
        except Exception as e:
            self.logger.warning(f"Redis set error: {str(e)}")

    def _get_from_memory(self, cache_key: str) -> Optional[PredictionResult]:
        """インメモリからデータ取得"""
        if cache_key not in self.memory_cache:
            return None

        # TTLチェック
        if cache_key in self.cache_timestamps:
            cached_time, ttl = self.cache_timestamps[cache_key]
            if time.time() - cached_time > ttl:
                # 期限切れ
                del self.memory_cache[cache_key]
                del self.cache_timestamps[cache_key]
                return None

        return self.memory_cache[cache_key]

    def _set_to_memory(self, cache_key: str, result: PredictionResult, ttl: int):
        """インメモリにデータ設定"""
        self.memory_cache[cache_key] = result
        self.cache_timestamps[cache_key] = (time.time(), ttl)

        # メモリキャッシュサイズ制限
        if len(self.memory_cache) > 1000:
            self._cleanup_memory_cache()

    def _cleanup_memory_cache(self):
        """メモリキャッシュクリーンアップ"""
        current_time = time.time()
        expired_keys = []

        # 期限切れエントリを特定
        for cache_key, (cached_time, ttl) in self.cache_timestamps.items():
            if current_time - cached_time > ttl:
                expired_keys.append(cache_key)

        # 期限切れエントリを削除
        removed_count = 0
        for key in expired_keys:
            self.memory_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
            removed_count += 1

        if removed_count > 0:
            self.logger.debug(f"Cleaned up {removed_count} expired cache entries")

        return removed_count

    def _invalidate_symbol(self, symbol: str):
        """特定銘柄のキャッシュ無効化"""
        # Redisパターン削除
        if self.redis_client:
            try:
                pattern = f"pred:{symbol}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                self.logger.warning(f"Redis invalidation error: {str(e)}")

        # インメモリ削除
        keys_to_delete = [
            key for key in self.memory_cache.keys() if key.startswith(f"pred:{symbol}:")
        ]
        for key in keys_to_delete:
            self.memory_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)

    def get_cache_statistics(self) -> Dict[str, Any]:
        """キャッシュ統計取得"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        )

        return {
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "invalidations": self.cache_stats["invalidations"],
            "redis_hits": self.cache_stats["redis_hits"],
            "memory_hits": self.cache_stats["memory_hits"],
            "redis_available": self.redis_client is not None,
            "memory_cache_size": len(self.memory_cache),
        }

    def clear_cache(self):
        """キャッシュ全削除"""
        # Redis削除
        if self.redis_client:
            try:
                keys = self.redis_client.keys("pred:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                self.logger.warning(f"Redis clear error: {str(e)}")

        # インメモリ削除
        self.memory_cache.clear()
        self.cache_timestamps.clear()

        self.logger.info("Cache cleared")
