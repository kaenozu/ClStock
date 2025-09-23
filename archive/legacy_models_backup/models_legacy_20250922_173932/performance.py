"""Performance optimization models and utilities."""

import logging
import os
import pandas as pd
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .base import StockPredictor, PredictionResult, CacheablePredictor
from .core import EnsembleStockPredictor

logger = logging.getLogger(__name__)


class ParallelStockPredictor(StockPredictor):
    """並列処理対応の高速株価予測器"""

    def __init__(self, ensemble_predictor: EnsembleStockPredictor, n_jobs: int = -1):
        super().__init__("parallel")
        self.ensemble_predictor = ensemble_predictor
        self.n_jobs = n_jobs if n_jobs != -1 else os.cpu_count()
        self.batch_cache: Dict[str, float] = {}

    def predict_multiple_stocks_parallel(self, symbols: List[str]) -> Dict[str, float]:
        """複数銘柄の並列予測"""
        results = {}

        # キャッシュチェック
        uncached_symbols = []
        for symbol in symbols:
            if symbol in self.batch_cache:
                results[symbol] = self.batch_cache[symbol]
            else:
                uncached_symbols.append(symbol)

        if not uncached_symbols:
            return results

        logger.info(f"Predicting {len(uncached_symbols)} stocks in parallel...")

        # 並列実行
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_symbol = {
                executor.submit(self._safe_predict_score, symbol): symbol
                for symbol in uncached_symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    score = future.result()
                    results[symbol] = score
                    self.batch_cache[symbol] = score
                except Exception as e:
                    logger.error(f"Error predicting {symbol}: {str(e)}")
                    results[symbol] = 50.0

        return results

    def _safe_predict_score(self, symbol: str) -> float:
        """安全な予測スコア取得"""
        try:
            result = self.ensemble_predictor.predict(symbol)
            return result.prediction
        except Exception as e:
            logger.error(f"Error predicting score for {symbol}: {str(e)}")
            return 50.0

    def batch_data_preparation(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """バッチデータ準備（並列）"""
        data_results = {}

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_symbol = {
                executor.submit(self._get_stock_data_safe, symbol): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty:
                        data_results[symbol] = data
                except Exception as e:
                    logger.error(f"Error getting data for {symbol}: {str(e)}")

        return data_results

    def _get_stock_data_safe(self, symbol: str) -> pd.DataFrame:
        """安全なデータ取得"""
        try:
            from data.stock_data import StockDataProvider

            data_provider = StockDataProvider()
            return data_provider.get_stock_data(symbol, "1y")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def clear_batch_cache(self):
        """バッチキャッシュをクリア"""
        self.batch_cache.clear()
        logger.info("Batch cache cleared")

    def train(self, data: pd.DataFrame, target: pd.Series) -> None:
        """Train the underlying ensemble predictor"""
        self.ensemble_predictor.train(data, target)
        self._is_trained = True

    def predict(
        self, symbol: str, data: Optional[pd.DataFrame] = None
    ) -> PredictionResult:
        """Single prediction using ensemble predictor"""
        if not self.is_trained():
            raise ValueError("Model must be trained before making predictions")

        try:
            result = self.ensemble_predictor.predict(symbol, data)
            return PredictionResult(
                prediction=result.prediction,
                confidence=result.confidence,
                timestamp=datetime.now(),
                metadata={
                    "model_type": self.model_type,
                    "symbol": symbol,
                    "parallel_enabled": True,
                    "n_jobs": self.n_jobs,
                },
            )
        except Exception as e:
            logger.error(f"Parallel prediction error for {symbol}: {str(e)}")
            return PredictionResult(
                prediction=50.0,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"error": str(e)},
            )


class AdvancedCacheManager:
    """高度なキャッシュ管理システム"""

    def __init__(self, max_size: int = 1000):
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        self.prediction_cache: Dict[str, float] = {}
        self.max_size = max_size
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "feature_cache_size": 0,
            "prediction_cache_size": 0,
        }

    def get_cached_features(
        self, symbol: str, data_hash: str
    ) -> Optional[pd.DataFrame]:
        """特徴量キャッシュから取得"""
        cache_key = f"{symbol}_{data_hash}"
        if cache_key in self.feature_cache:
            self.cache_stats["hits"] += 1
            return self.feature_cache[cache_key]
        else:
            self.cache_stats["misses"] += 1
            return None

    def cache_features(self, symbol: str, data_hash: str, features: pd.DataFrame):
        """特徴量をキャッシュ"""
        cache_key = f"{symbol}_{data_hash}"

        # キャッシュサイズ制限チェック
        if len(self.feature_cache) >= self.max_size:
            self._evict_oldest_feature()

        self.feature_cache[cache_key] = features.copy()
        self.cache_stats["feature_cache_size"] = len(self.feature_cache)

    def get_cached_prediction(self, symbol: str, features_hash: str) -> Optional[float]:
        """予測結果キャッシュから取得"""
        cache_key = f"{symbol}_{features_hash}"
        if cache_key in self.prediction_cache:
            self.cache_stats["hits"] += 1
            return self.prediction_cache[cache_key]
        else:
            self.cache_stats["misses"] += 1
            return None

    def cache_prediction(self, symbol: str, features_hash: str, prediction: float):
        """予測結果をキャッシュ"""
        cache_key = f"{symbol}_{features_hash}"

        # キャッシュサイズ制限チェック
        if len(self.prediction_cache) >= self.max_size:
            self._evict_oldest_prediction()

        self.prediction_cache[cache_key] = prediction
        self.cache_stats["prediction_cache_size"] = len(self.prediction_cache)

    def _evict_oldest_feature(self):
        """最も古い特徴量キャッシュを削除"""
        if self.feature_cache:
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]

    def _evict_oldest_prediction(self):
        """最も古い予測キャッシュを削除"""
        if self.prediction_cache:
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]

    def cleanup_old_cache(self, max_size: Optional[int] = None):
        """古いキャッシュをクリーンアップ"""
        if max_size is None:
            max_size = self.max_size

        # 特徴量キャッシュのクリーンアップ
        if len(self.feature_cache) > max_size:
            keys_to_remove = list(self.feature_cache.keys())[:-max_size]
            for key in keys_to_remove:
                del self.feature_cache[key]

        # 予測キャッシュのクリーンアップ
        if len(self.prediction_cache) > max_size:
            keys_to_remove = list(self.prediction_cache.keys())[:-max_size]
            for key in keys_to_remove:
                del self.prediction_cache[key]

        self.cache_stats["feature_cache_size"] = len(self.feature_cache)
        self.cache_stats["prediction_cache_size"] = len(self.prediction_cache)

    def clear_all_cache(self):
        """全てのキャッシュをクリア"""
        self.feature_cache.clear()
        self.prediction_cache.clear()
        self.cache_stats["feature_cache_size"] = 0
        self.cache_stats["prediction_cache_size"] = 0
        logger.info("All caches cleared")

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
            "memory_efficiency": self._calculate_memory_efficiency(),
        }

    def _calculate_memory_efficiency(self) -> float:
        """メモリ効率を計算"""
        total_cache_size = (
            self.cache_stats["feature_cache_size"]
            + self.cache_stats["prediction_cache_size"]
        )
        max_total_size = self.max_size * 2  # 両方のキャッシュの最大サイズ
        return total_cache_size / max_total_size if max_total_size > 0 else 0.0

    def get_data_hash(self, data: pd.DataFrame) -> str:
        """データのハッシュ値を生成"""
        if data is None or data.empty:
            return "empty"

        # データの最後の行とサイズでハッシュを生成
        try:
            last_row_values = data.iloc[-1].values
            data_signature = f"{len(data)}_{hash(tuple(last_row_values))}"
            return str(abs(hash(data_signature)))
        except Exception:
            return str(hash(str(data.shape)))


class UltraHighPerformancePredictor(CacheablePredictor):
    """超高性能予測器 - キャッシュと並列処理の統合"""

    def __init__(
        self,
        base_predictor: StockPredictor,
        cache_manager: AdvancedCacheManager,
        n_jobs: int = -1,
    ):
        super().__init__("ultra_performance", cache_manager.max_size)
        self.base_predictor = base_predictor
        self.cache_manager = cache_manager
        self.parallel_predictor = ParallelStockPredictor(
            (
                base_predictor
                if isinstance(base_predictor, EnsembleStockPredictor)
                else None
            ),
            n_jobs,
        )

    def train(self, data: pd.DataFrame, target: pd.Series) -> None:
        """Train the base predictor"""
        self.base_predictor.train(data, target)
        self._is_trained = True

    def predict(
        self, symbol: str, data: Optional[pd.DataFrame] = None
    ) -> PredictionResult:
        """Ultra-fast prediction with caching"""
        if not self.is_trained():
            raise ValueError("Model must be trained before making predictions")

        try:
            if data is None:
                from data.stock_data import StockDataProvider

                data_provider = StockDataProvider()
                data = data_provider.get_stock_data(symbol, "1y")

            # データハッシュを生成
            data_hash = self.cache_manager.get_data_hash(data)

            # キャッシュから予測結果を確認
            cached_prediction = self.cache_manager.get_cached_prediction(
                symbol, data_hash
            )
            if cached_prediction is not None:
                return PredictionResult(
                    prediction=cached_prediction,
                    confidence=0.9,  # キャッシュヒット時の高い信頼度
                    timestamp=datetime.now(),
                    metadata={
                        "model_type": self.model_type,
                        "symbol": symbol,
                        "cache_hit": True,
                        "data_hash": data_hash,
                    },
                )

            # キャッシュミスの場合、新しい予測を実行
            result = self.base_predictor.predict(symbol, data)

            # 結果をキャッシュに保存
            self.cache_manager.cache_prediction(symbol, data_hash, result.prediction)

            # メタデータを更新
            result.metadata.update(
                {"cache_hit": False, "data_hash": data_hash, "ultra_performance": True}
            )

            return result

        except Exception as e:
            logger.error(f"Ultra performance prediction error for {symbol}: {str(e)}")
            return PredictionResult(
                prediction=50.0,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"error": str(e)},
            )

    def predict_multiple(self, symbols: List[str]) -> Dict[str, PredictionResult]:
        """並列予測とキャッシュの組み合わせ"""
        results = {}

        # キャッシュヒットをチェック
        uncached_symbols = []
        for symbol in symbols:
            try:
                from data.stock_data import StockDataProvider

                data_provider = StockDataProvider()
                data = data_provider.get_stock_data(symbol, "1y")
                data_hash = self.cache_manager.get_data_hash(data)

                cached_prediction = self.cache_manager.get_cached_prediction(
                    symbol, data_hash
                )
                if cached_prediction is not None:
                    results[symbol] = PredictionResult(
                        prediction=cached_prediction,
                        confidence=0.9,
                        timestamp=datetime.now(),
                        metadata={"cache_hit": True, "symbol": symbol},
                    )
                else:
                    uncached_symbols.append(symbol)
            except Exception:
                uncached_symbols.append(symbol)

        # キャッシュミスしたシンボルを並列処理
        if uncached_symbols:
            with ThreadPoolExecutor(
                max_workers=self.parallel_predictor.n_jobs
            ) as executor:
                future_to_symbol = {
                    executor.submit(self.predict, symbol): symbol
                    for symbol in uncached_symbols
                }

                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        results[symbol] = result
                    except Exception as e:
                        logger.error(f"Error in parallel prediction for {symbol}: {e}")
                        results[symbol] = PredictionResult(
                            prediction=50.0,
                            confidence=0.0,
                            timestamp=datetime.now(),
                            metadata={"error": str(e)},
                        )

        return results

    def get_performance_stats(self) -> Dict:
        """パフォーマンス統計を取得"""
        cache_stats = self.cache_manager.get_cache_stats()
        return {
            "cache_stats": cache_stats,
            "parallel_jobs": self.parallel_predictor.n_jobs,
            "model_type": self.model_type,
            "base_predictor_type": self.base_predictor.model_type,
            "is_trained": self.is_trained(),
        }

    def optimize_cache(self):
        """キャッシュを最適化"""
        self.cache_manager.cleanup_old_cache()
        logger.info("Cache optimization completed")

    def warm_up_cache(self, symbols: List[str]):
        """キャッシュのウォームアップ"""
        logger.info(f"Warming up cache for {len(symbols)} symbols...")
        self.predict_multiple(symbols)
        logger.info("Cache warm-up completed")
