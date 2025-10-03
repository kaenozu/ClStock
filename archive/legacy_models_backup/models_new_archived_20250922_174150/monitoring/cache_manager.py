#!/usr/bin/env python3
"""
キャッシュ管理モジュール
高度なキャッシュシステムによる性能最適化
"""

import logging
from dataclasses import asdict
from typing import Optional, Dict, Any, List
import pandas as pd
from pathlib import Path
import hashlib
import json
import threading
from datetime import datetime, timedelta

from ..base.interfaces import CacheManager

logger = logging.getLogger(__name__)


class AdvancedCacheManager(CacheManager):
    """高度なキャッシュ管理システム"""

    def __init__(
        self,
        max_cache_size: int = 1000,
        ttl_hours: int = 24,
        cleanup_interval: int = 1800,
    ):
        self.feature_cache = {}
        self.prediction_cache = {}
        self.max_cache_size = max_cache_size
        self.ttl_hours = ttl_hours
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "feature_cache_size": 0,
            "prediction_cache_size": 0,
        }
        self.cache_file = Path("cache/advanced_cache.json")
        self.cache_file.parent.mkdir(exist_ok=True)

        # 自動クリーンアップ設定
        self.auto_cleanup_enabled = True
        self.cleanup_interval = (
            cleanup_interval  # 30分ごとにクリーンアップ（デフォルト）
        )
        self._shutdown_event = threading.Event()
        self.cleanup_thread = None
        self._start_cleanup_thread()

        logger.info(
            f"AdvancedCacheManager initialized with cleanup_interval={cleanup_interval}s, max_cache_size={max_cache_size}"
        )

    def _start_cleanup_thread(self):
        """自動クリーンアップスレッドを開始 - 安全性を向上"""
        if not self.auto_cleanup_enabled:
            logger.debug("Auto cleanup is disabled")
            return

        # 既存のスレッドが生きている場合は何もしない
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            logger.warning("Cleanup thread is already running")
            return

        try:
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_worker, daemon=True, name="CacheCleanupWorker"
            )
            self.cleanup_thread.start()
            logger.info(
                "Automatic cache cleanup thread started for AdvancedCacheManager"
            )
        except Exception as e:
            logger.error(f"Failed to start cleanup thread: {e}")
            self.cleanup_thread = None

    def _cleanup_worker(self):
        """クリーンアップワーカースレッド - より安全な実装"""
        logger.info("Cache cleanup worker started")

        while not self._shutdown_event.is_set():
            try:
                # シャットダウンイベントを待機（タイムアウト付き）
                if self._shutdown_event.wait(self.cleanup_interval):
                    logger.info("Shutdown signal received, terminating cleanup worker")
                    break

                # シャットダウン中でない場合のみクリーンアップ実行
                if not self._shutdown_event.is_set():
                    self.cleanup_old_cache()
                    logger.debug(
                        f"Periodic cache cleanup completed. Feature cache: {len(self.feature_cache)}, Prediction cache: {len(self.prediction_cache)}"
                    )

            except Exception as e:
                logger.error(f"Cache cleanup worker error: {e}")
                # エラーが発生しても継続するが、短い間隔で再試行
                if not self._shutdown_event.wait(5):  # 5秒待機
                    continue
                else:
                    break

        logger.info("Cache cleanup worker stopped gracefully")

    def shutdown(self):
        """キャッシュマネージャーをシャットダウン"""
        logger.info("Shutting down advanced cache manager")

        # シャットダウンイベントを設定
        self._shutdown_event.set()

        # クリーンアップスレッドの終了を待機
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            logger.info("Waiting for cache cleanup thread to terminate...")
            self.cleanup_thread.join(timeout=10.0)  # タイムアウトを10秒に延長

            if self.cleanup_thread.is_alive():
                logger.warning(
                    "Cache cleanup thread did not terminate gracefully within timeout"
                )
                # デーモンスレッドの場合は強制終了は避ける
            else:
                logger.info("Cache cleanup thread terminated successfully")

        # キャッシュをディスクに保存（エラー処理を強化）
        try:
            self.save_cache_to_disk()
            logger.info("Cache saved to disk during shutdown")
        except Exception as e:
            logger.error(f"Error saving cache during shutdown: {e}")

        # クリーンアップ処理の完了
        logger.info("Cache manager shutdown completed")

    def get(self, key: str) -> Optional[Any]:
        """キャッシュ値取得（汎用インターフェース）"""
        if key in self.feature_cache:
            cache_entry = self.feature_cache[key]
            if self._is_valid_cache_entry(cache_entry):
                self.cache_stats["hits"] += 1
                return cache_entry["data"]

        if key in self.prediction_cache:
            cache_entry = self.prediction_cache[key]
            if self._is_valid_cache_entry(cache_entry):
                self.cache_stats["hits"] += 1
                return cache_entry["data"]

        self.cache_stats["misses"] += 1
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """キャッシュ値設定（汎用インターフェース）"""
        cache_entry = {
            "data": value,
            "timestamp": datetime.now().isoformat(),
            "ttl_seconds": ttl,
        }

        # データ型に応じてキャッシュを分類
        if isinstance(value, pd.DataFrame):
            self.feature_cache[key] = cache_entry
        else:
            self.prediction_cache[key] = cache_entry

        self._update_cache_stats()
        self.cleanup_old_cache()

    def delete(self, key: str) -> None:
        """キャッシュ削除"""
        if key in self.feature_cache:
            del self.feature_cache[key]
        if key in self.prediction_cache:
            del self.prediction_cache[key]
        self._update_cache_stats()

    def clear(self) -> None:
        """全キャッシュ削除"""
        self.feature_cache.clear()
        self.prediction_cache.clear()
        self._update_cache_stats()

    def get_cached_features(
        self, symbol: str, data_hash: str
    ) -> Optional[pd.DataFrame]:
        """特徴量キャッシュから取得"""
        cache_key = f"features_{symbol}_{data_hash}"
        cached_data = self.get(cache_key)
        return cached_data if isinstance(cached_data, pd.DataFrame) else None

    def cache_features(self, symbol: str, data_hash: str, features: pd.DataFrame):
        """特徴量をキャッシュ"""
        cache_key = f"features_{symbol}_{data_hash}"
        self.set(cache_key, features, ttl=self.ttl_hours * 3600)

    def get_cached_prediction(self, symbol: str, features_hash: str) -> Optional[float]:
        """予測結果キャッシュから取得"""
        cache_key = f"prediction_{symbol}_{features_hash}"
        cached_data = self.get(cache_key)
        return cached_data if isinstance(cached_data, (int, float)) else None

    def cache_prediction(self, symbol: str, features_hash: str, prediction: float):
        """予測結果をキャッシュ"""
        cache_key = f"prediction_{symbol}_{features_hash}"
        self.set(cache_key, prediction, ttl=self.ttl_hours * 3600)

    def _is_valid_cache_entry(self, cache_entry: Dict) -> bool:
        """キャッシュエントリの有効性チェック"""
        try:
            timestamp = datetime.fromisoformat(cache_entry["timestamp"])
            ttl_seconds = cache_entry.get("ttl_seconds", self.ttl_hours * 3600)
            expiry_time = timestamp + timedelta(seconds=ttl_seconds)
            return datetime.now() < expiry_time
        except (KeyError, ValueError):
            return False

    def cleanup_old_cache(self):
        """古いキャッシュをクリーンアップ"""
        # TTL期限切れエントリを削除
        removed_feature = self._remove_expired_entries(self.feature_cache)
        removed_prediction = self._remove_expired_entries(self.prediction_cache)

        # サイズ制限によるLRU削除
        feature_removed_lru = 0
        prediction_removed_lru = 0

        if len(self.feature_cache) > self.max_cache_size:
            feature_removed_lru = self._apply_lru_cleanup(self.feature_cache)

        if len(self.prediction_cache) > self.max_cache_size:
            prediction_removed_lru = self._apply_lru_cleanup(self.prediction_cache)

        total_removed = (
            removed_feature
            + removed_prediction
            + feature_removed_lru
            + prediction_removed_lru
        )
        if total_removed > 0:
            self._update_cache_stats()
            logger.debug(
                f"Cache cleanup completed. Removed {removed_feature} expired features, "
                f"{removed_prediction} expired predictions, {feature_removed_lru} LRU features, "
                f"{prediction_removed_lru} LRU predictions"
            )

    def _remove_expired_entries(self, cache: Dict) -> int:
        """期限切れエントリを削除"""
        expired_keys = []
        for key, entry in cache.items():
            if not self._is_valid_cache_entry(entry):
                expired_keys.append(key)

        for key in expired_keys:
            del cache[key]

        return len(expired_keys)

    def _apply_lru_cleanup(self, cache: Dict) -> int:
        """LRUアルゴリズムでキャッシュクリーンアップ"""
        # タイムスタンプでソートして古いものから削除
        sorted_items = sorted(cache.items(), key=lambda x: x[1]["timestamp"])

        items_to_remove = len(cache) - self.max_cache_size
        removed_count = 0
        for i in range(min(items_to_remove, len(sorted_items))):
            key_to_remove = sorted_items[i][0]
            del cache[key_to_remove]
            removed_count += 1

        return removed_count

    def _update_cache_stats(self):
        """キャッシュ統計を更新"""
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
            "memory_usage_mb": self._estimate_memory_usage(),
        }

    def _estimate_memory_usage(self) -> float:
        """メモリ使用量を推定（MB）"""
        # 簡易的な推定計算
        feature_size = len(self.feature_cache) * 0.1  # 仮定: 1エントリ0.1MB
        prediction_size = len(self.prediction_cache) * 0.001  # 仮定: 1エントリ0.001MB
        return feature_size + prediction_size

    def generate_data_hash(self, data: pd.DataFrame) -> str:
        """データハッシュ生成"""
        try:
            # データフレームの形状とサンプル値でハッシュ生成
            shape_str = f"{data.shape[0]}x{data.shape[1]}"
            sample_values = str(data.iloc[0].values) if not data.empty else ""
            hash_input = f"{shape_str}_{sample_values}"
            return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16]

    def save_cache_to_disk(self):
        """キャッシュをディスクに保存"""
        try:
            cache_data = {
                "feature_cache": self._serialize_cache(self.feature_cache),
                "prediction_cache": self._serialize_cache(self.prediction_cache),
                "cache_stats": self.cache_stats,
                "saved_at": datetime.now().isoformat(),
            }

            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Cache saved to {self.cache_file}")

        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def load_cache_from_disk(self) -> bool:
        """ディスクからキャッシュを読み込み"""
        try:
            if not self.cache_file.exists():
                return False

            with open(self.cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            self.feature_cache = self._deserialize_cache(
                cache_data.get("feature_cache", {})
            )
            self.prediction_cache = self._deserialize_cache(
                cache_data.get("prediction_cache", {})
            )
            self.cache_stats = cache_data.get("cache_stats", self.cache_stats)

            # 期限切れキャッシュをクリーンアップ
            self.cleanup_old_cache()

            logger.info(f"Cache loaded from {self.cache_file}")
            return True

        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return False

    def _serialize_cache(self, cache: Dict) -> Dict:
        """キャッシュをシリアライズ（DataFrameを除く）"""
        serializable_cache = {}
        for key, entry in cache.items():
            if isinstance(entry["data"], pd.DataFrame):
                # DataFrameは保存しない（大きすぎるため）
                continue
            serializable_cache[key] = entry
        return serializable_cache

    def _deserialize_cache(self, cache_data: Dict) -> Dict:
        """キャッシュをデシリアライズ"""
        return cache_data


class RealTimeCacheManager(AdvancedCacheManager):
    """リアルタイムデータ専用キャッシュマネージャー"""

    def __init__(
        self,
        max_cache_size: int = 5000,
        ttl_hours: int = 1,
        cleanup_interval: int = 600,
    ):
        super().__init__(max_cache_size, ttl_hours)

        # リアルタイムデータ専用キャッシュ
        self.tick_cache = {}
        self.order_book_cache = {}
        self.index_cache = {}
        self.news_cache = {}

        # 時系列データ保持設定
        self.max_tick_history = 1000  # 銘柄あたり最大1000ティック
        self.max_order_book_history = 100  # 板情報履歴

        # データ統計
        self.real_time_stats = {
            "total_ticks_cached": 0,
            "total_order_books_cached": 0,
            "cache_evictions": 0,
            "oldest_data_timestamp": None,
            "newest_data_timestamp": None,
        }

        # 自動クリーンアップ設定（リアルタイム用）
        self.realtime_cleanup_interval = (
            cleanup_interval  # 10分ごとにクリーンアップ（デフォルト）
        )
        self._start_realtime_cleanup_thread()

        logger.info(
            f"RealTimeCacheManager initialized with cleanup_interval={cleanup_interval}s, max_cache_size={max_cache_size}"
        )

    def _start_realtime_cleanup_thread(self):
        """リアルタイムキャッシュ用自動クリーンアップスレッドを開始"""
        if self.auto_cleanup_enabled:
            realtime_cleanup_thread = threading.Thread(
                target=self._realtime_cleanup_worker, daemon=True
            )
            realtime_cleanup_thread.start()
            logger.info("Automatic realtime cache cleanup thread started")

    def _realtime_cleanup_worker(self):
        """リアルタイムキャッシュクリーンアップワーカー"""
        while not self._shutdown_event.is_set():
            try:
                # クリーンアップ間隔待機
                if self._shutdown_event.wait(self.realtime_cleanup_interval):
                    break

                # リアルタイムキャッシュクリーンアップ実行
                self.cleanup_real_time_cache(
                    older_than_hours=2
                )  # 2時間以上古いデータをクリーンアップ
                logger.debug(
                    f"Realtime cache cleanup completed. Tick cache symbols: {len(self.tick_cache)}, Order book cache symbols: {len(self.order_book_cache)}"
                )

            except Exception as e:
                logger.error(f"Realtime cache cleanup worker error: {e}")
                # エラーでも継続

        logger.info("Realtime cache cleanup worker stopped")

    def shutdown(self):
        """キャッシュマネージャーをシャットダウン"""
        logger.info("Shutting down realtime cache manager")
        self._shutdown_event.set()

        # クリーンアップスレッドの終了を待機
        # Note: The realtime cleanup thread shares the same shutdown event
        # We don't have a reference to it directly, but it will stop when the event is set

        # キャッシュをディスクに保存
        try:
            self.save_cache_to_disk()
            logger.info("Cache saved to disk during shutdown")
        except Exception as e:
            logger.error(f"Error saving cache during shutdown: {e}")

    def cache_tick_data(self, tick_data, max_history: Optional[int] = None) -> None:
        """ティックデータを時系列キャッシュに保存"""
        from models.base.interfaces import TickData

        if not isinstance(tick_data, TickData):
            logger.warning("Invalid tick data type for caching")
            return

        symbol = tick_data.symbol

        # 時系列データの管理
        if symbol not in self.tick_cache:
            self.tick_cache[symbol] = []

        self.tick_cache[symbol].append(
            {
                "data": tick_data,
                "timestamp": tick_data.timestamp,
                "cached_at": datetime.now(),
            }
        )

        # 履歴サイズ制限
        max_hist = max_history or self.max_tick_history
        if len(self.tick_cache[symbol]) > max_hist:
            self.tick_cache[symbol] = self.tick_cache[symbol][-max_hist:]
            self.real_time_stats["cache_evictions"] += 1

        # 統計更新
        self.real_time_stats["total_ticks_cached"] += 1
        self._update_real_time_stats(tick_data.timestamp)

        # 最新データとしてもキャッシュ
        latest_key = f"latest_tick_{symbol}"
        self.set(latest_key, tick_data, ttl=300)

    def cache_order_book_data(
        self, order_book_data, max_history: Optional[int] = None
    ) -> None:
        """板情報データを時系列キャッシュに保存"""
        from models.base.interfaces import OrderBookData

        if not isinstance(order_book_data, OrderBookData):
            logger.warning("Invalid order book data type for caching")
            return

        symbol = order_book_data.symbol

        if symbol not in self.order_book_cache:
            self.order_book_cache[symbol] = []

        self.order_book_cache[symbol].append(
            {
                "data": order_book_data,
                "timestamp": order_book_data.timestamp,
                "cached_at": datetime.now(),
            }
        )

        # 履歴サイズ制限
        max_hist = max_history or self.max_order_book_history
        if len(self.order_book_cache[symbol]) > max_hist:
            self.order_book_cache[symbol] = self.order_book_cache[symbol][-max_hist:]

        self.real_time_stats["total_order_books_cached"] += 1
        self._update_real_time_stats(order_book_data.timestamp)

        # 最新データとしてもキャッシュ
        latest_key = f"latest_orderbook_{symbol}"
        self.set(latest_key, order_book_data, ttl=60)

    def get_tick_history(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List:
        """指定期間のティック履歴を取得"""
        if symbol not in self.tick_cache:
            return []

        tick_history = self.tick_cache[symbol]

        # 時間範囲フィルタ
        if start_time or end_time:
            filtered_history = []
            for entry in tick_history:
                timestamp = entry["timestamp"]
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                filtered_history.append(entry)
            tick_history = filtered_history

        # 件数制限
        if limit and len(tick_history) > limit:
            tick_history = tick_history[-limit:]

        return [entry["data"] for entry in tick_history]

    def get_order_book_history(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List:
        """指定期間の板情報履歴を取得"""
        if symbol not in self.order_book_cache:
            return []

        ob_history = self.order_book_cache[symbol]

        # 時間範囲フィルタ
        if start_time or end_time:
            filtered_history = []
            for entry in ob_history:
                timestamp = entry["timestamp"]
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                filtered_history.append(entry)
            ob_history = filtered_history

        # 件数制限
        if limit and len(ob_history) > limit:
            ob_history = ob_history[-limit:]

        return [entry["data"] for entry in ob_history]

    def calculate_market_metrics(self, symbol: str) -> Dict[str, Any]:
        """市場メトリクスを算出"""
        tick_history = self.get_tick_history(symbol, limit=100)

        if not tick_history:
            return {}

        prices = [tick.price for tick in tick_history]
        volumes = [tick.volume for tick in tick_history]

        # 基本統計
        current_price = prices[-1] if prices else 0
        avg_price = sum(prices) / len(prices)
        price_volatility = self._calculate_volatility(prices)
        total_volume = sum(volumes)

        # 価格変動
        if len(prices) >= 2:
            price_change = prices[-1] - prices[0]
            price_change_pct = (price_change / prices[0]) * 100 if prices[0] != 0 else 0
        else:
            price_change = 0
            price_change_pct = 0

        return {
            "symbol": symbol,
            "current_price": current_price,
            "average_price": avg_price,
            "price_volatility": price_volatility,
            "total_volume": total_volume,
            "price_change": price_change,
            "price_change_percent": price_change_pct,
            "tick_count": len(tick_history),
            "last_update": (
                tick_history[-1].timestamp.isoformat() if tick_history else None
            ),
        }

    def _calculate_volatility(self, prices: List[float]) -> float:
        """価格のボラティリティを計算"""
        if len(prices) < 2:
            return 0.0

        # 標準偏差を計算
        mean_price = sum(prices) / len(prices)
        variance = sum((price - mean_price) ** 2 for price in prices) / len(prices)
        volatility = variance**0.5

        # パーセンテージボラティリティ
        return (volatility / mean_price) * 100 if mean_price != 0 else 0.0

    def _update_real_time_stats(self, timestamp: datetime) -> None:
        """リアルタイム統計を更新"""
        if (
            self.real_time_stats["oldest_data_timestamp"] is None
            or timestamp < self.real_time_stats["oldest_data_timestamp"]
        ):
            self.real_time_stats["oldest_data_timestamp"] = timestamp

        if (
            self.real_time_stats["newest_data_timestamp"] is None
            or timestamp > self.real_time_stats["newest_data_timestamp"]
        ):
            self.real_time_stats["newest_data_timestamp"] = timestamp

    def get_real_time_cache_stats(self) -> Dict[str, Any]:
        """リアルタイムキャッシュ統計を取得"""
        base_stats = self.get_cache_stats()

        # リアルタイム専用統計を追加
        base_stats.update(
            {
                "real_time_stats": self.real_time_stats,
                "cached_symbols": {
                    "tick_symbols": list(self.tick_cache.keys()),
                    "order_book_symbols": list(self.order_book_cache.keys()),
                    "total_tick_symbols": len(self.tick_cache),
                    "total_order_book_symbols": len(self.order_book_cache),
                },
                "cache_sizes": {
                    "tick_cache_entries": sum(
                        len(entries) for entries in self.tick_cache.values()
                    ),
                    "order_book_cache_entries": sum(
                        len(entries) for entries in self.order_book_cache.values()
                    ),
                },
            }
        )

        return base_stats

    def cleanup_real_time_cache(self, older_than_hours: int = 24) -> int:
        """古いリアルタイムデータをクリーンアップ"""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        total_removed = 0

        # ティックキャッシュのクリーンアップ
        for symbol in list(self.tick_cache.keys()):
            original_count = len(self.tick_cache[symbol])
            self.tick_cache[symbol] = [
                entry
                for entry in self.tick_cache[symbol]
                if entry["timestamp"] > cutoff_time
            ]
            removed_count = original_count - len(self.tick_cache[symbol])
            total_removed += removed_count

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old tick entries for {symbol}")

            # 空になったキャッシュを削除
            if not self.tick_cache[symbol]:
                del self.tick_cache[symbol]

        # 板情報キャッシュのクリーンアップ
        for symbol in list(self.order_book_cache.keys()):
            original_count = len(self.order_book_cache[symbol])
            self.order_book_cache[symbol] = [
                entry
                for entry in self.order_book_cache[symbol]
                if entry["timestamp"] > cutoff_time
            ]
            removed_count = original_count - len(self.order_book_cache[symbol])
            total_removed += removed_count

            if removed_count > 0:
                logger.info(
                    f"Cleaned up {removed_count} old order book entries for {symbol}"
                )

            if not self.order_book_cache[symbol]:
                del self.order_book_cache[symbol]

        if total_removed > 0:
            logger.debug(
                f"Realtime cache cleanup completed. Removed {total_removed} entries"
            )

        return total_removed

    def export_real_time_data(self, symbol: str, format: str = "dataframe") -> Any:
        """リアルタイムデータをエクスポート"""
        tick_history = self.get_tick_history(symbol)

        if format == "dataframe":
            import pandas as pd

            if not tick_history:
                return pd.DataFrame()

            data = []
            for tick in tick_history:
                data.append(
                    {
                        "timestamp": tick.timestamp,
                        "symbol": tick.symbol,
                        "price": tick.price,
                        "volume": tick.volume,
                        "bid_price": tick.bid_price,
                        "ask_price": tick.ask_price,
                        "trade_type": tick.trade_type,
                    }
                )

            return pd.DataFrame(data)

        elif format == "json":
            return [asdict(tick) for tick in tick_history]
        else:
            return tick_history
