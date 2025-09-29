"""
統合ベース予測クラス - 全ての予測器の共通実装
重複コードを排除し、統一されたベース機能を提供
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from .interfaces import (
    StockPredictor,
    PredictionResult,
    ModelConfiguration,
    ModelType,
    PredictionMode,
    PerformanceMetrics,
    DataProvider,
    CacheProvider,
)


class BaseStockPredictor(StockPredictor):
    """
    統合ベース予測クラス
    全てのPredictorクラスで共通する機能を実装
    """

    def __init__(
        self,
        config: ModelConfiguration,
        data_provider: Optional[DataProvider] = None,
        cache_provider: Optional[CacheProvider] = None,
    ):
        super().__init__(config)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_provider = data_provider
        self.cache_provider = cache_provider

        # 統計情報
        self.prediction_count = 0
        self.total_execution_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

        # モデル固有データ
        self.model_data: Dict[str, Any] = {}

        self.logger.info(
            f"Initialized {self.__class__.__name__} with config: {config.model_type.value}"
        )

    def predict(self, symbol: str) -> PredictionResult:
        """統一された予測実行フロー"""
        start_time = time.time()

        try:
            # 1. 入力検証
            if not self.validate_symbol(symbol):
                raise ValueError(f"Invalid symbol: {symbol}")

            if not self.is_ready():
                raise RuntimeError("Model is not ready for prediction")

            # 2. キャッシュ確認
            if self.cache_provider and self.config.cache_enabled:
                cache_key = self._generate_cache_key(symbol)
                cached_result = self.cache_provider.get(cache_key)
                if cached_result is not None:
                    self.cache_hits += 1
                    self.logger.debug(f"Cache hit for {symbol}")
                    return cached_result
                self.cache_misses += 1

            # 3. 実際の予測実行（サブクラスで実装）
            prediction_value = self._predict_implementation(symbol)
            confidence = self._calculate_confidence(symbol, prediction_value)

            # 4. 結果作成
            execution_time = time.time() - start_time
            result = PredictionResult(
                prediction=float(np.clip(prediction_value, 0, 100)),
                confidence=float(np.clip(confidence, 0, 1)),
                accuracy=self._estimate_accuracy(),
                timestamp=datetime.now(),
                symbol=symbol,
                model_type=self.config.model_type,
                execution_time=execution_time,
                metadata=self._get_prediction_metadata(symbol),
            )

            # 5. キャッシュ保存
            if self.cache_provider and self.config.cache_enabled:
                cache_ttl = 3600  # 1時間
                self.cache_provider.put(cache_key, result, cache_ttl)

            # 6. 統計更新
            self._update_statistics(execution_time)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Prediction failed for {symbol}: {str(e)}")

            # フォールバック結果
            return self._create_fallback_result(symbol, execution_time, str(e))

    def predict_batch(self, symbols: List[str]) -> List[PredictionResult]:
        """バッチ予測の統一実装"""
        if not symbols:
            return []

        results = []

        # 並列処理が有効な場合
        if self.config.parallel_enabled and len(symbols) > 1:
            results = self._predict_batch_parallel(symbols)
        else:
            # シーケンシャル処理
            for symbol in symbols:
                try:
                    result = self.predict(symbol)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch prediction failed for {symbol}: {e}")
                    fallback = self._create_fallback_result(symbol, 0.0, str(e))
                    results.append(fallback)

        return results

    def get_performance_metrics(self) -> PerformanceMetrics:
        """性能指標の取得"""
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / cache_total if cache_total > 0 else 0.0

        return PerformanceMetrics(
            accuracy=self._estimate_accuracy(),
            precision=self._calculate_precision(),
            recall=self._calculate_recall(),
            f1_score=self._calculate_f1_score(),
            execution_time=self.total_execution_time / max(1, self.prediction_count),
            memory_usage=self._get_memory_usage(),
            cache_hit_rate=cache_hit_rate,
            total_predictions=self.prediction_count,
            successful_predictions=self.prediction_count,  # 簡略化
        )

    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報の統一取得"""
        return {
            "name": self.__class__.__name__,
            "type": self.config.model_type.value,
            "version": self._model_version,
            "is_trained": self.is_trained,
            "prediction_mode": self.config.prediction_mode.value,
            "model_data": {
                "num_models": len(getattr(self, "models", {})),
                "num_features": len(getattr(self, "feature_names", [])),
                "models": list(getattr(self, "models", {}).keys()),
            },
            "config": {
                "cache_enabled": self.config.cache_enabled,
                "parallel_enabled": self.config.parallel_enabled,
                "max_workers": self.config.max_workers,
                "timeout_seconds": self.config.timeout_seconds,
            },
            "statistics": {
                "prediction_count": self.prediction_count,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "avg_execution_time": self.total_execution_time
                / max(1, self.prediction_count),
            },
        }

    # 抽象メソッド（サブクラスで実装必須）

    def _predict_implementation(self, symbol: str) -> float:
        """実際の予測ロジック（サブクラスで実装）"""
        raise NotImplementedError("Subclasses must implement _predict_implementation")

    def train(self, data: pd.DataFrame) -> bool:
        """モデル訓練（サブクラスで実装）"""
        raise NotImplementedError("Subclasses must implement train")

    # 内部ヘルパーメソッド

    def _generate_cache_key(self, symbol: str) -> str:
        """キャッシュキー生成"""
        date_str = datetime.now().strftime("%Y%m%d_%H")
        return f"{self.__class__.__name__}_{symbol}_{date_str}_{self.config.prediction_mode.value}"

    def _calculate_confidence(self, symbol: str, prediction: float) -> float:
        """信頼度計算のデフォルト実装"""
        # 基本的な信頼度計算（サブクラスでオーバーライド可能）
        if hasattr(self, "model_data") and "confidence_scores" in self.model_data:
            return self.model_data["confidence_scores"].get(symbol, 0.5)
        return 0.7  # デフォルト信頼度

    def _estimate_accuracy(self) -> float:
        """精度推定のデフォルト実装"""
        return 75.0  # サブクラスでオーバーライド

    def _calculate_precision(self) -> float:
        """適合率計算のデフォルト実装"""
        return 0.75

    def _calculate_recall(self) -> float:
        """再現率計算のデフォルト実装"""
        return 0.75

    def _calculate_f1_score(self) -> float:
        """F1スコア計算"""
        precision = self._calculate_precision()
        recall = self._calculate_recall()
        return (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    def _get_memory_usage(self) -> float:
        """メモリ使用量取得"""
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    def _get_prediction_metadata(self, symbol: str) -> Dict[str, Any]:
        """予測メタデータ生成"""
        return {
            "model_version": self._model_version,
            "prediction_mode": self.config.prediction_mode.value,
            "data_timestamp": datetime.now().isoformat(),
        }

    def _create_fallback_result(
        self, symbol: str, execution_time: float, error: str
    ) -> PredictionResult:
        """フォールバック結果作成"""
        return PredictionResult(
            prediction=50.0,  # 中性予測
            confidence=0.1,  # 低信頼度
            accuracy=30.0,  # 低精度
            timestamp=datetime.now(),
            symbol=symbol,
            model_type=self.config.model_type,
            execution_time=execution_time,
            metadata={"error": error, "fallback": True},
        )

    def _update_statistics(self, execution_time: float):
        """統計情報更新"""
        self.prediction_count += 1
        self.total_execution_time += execution_time

    def _predict_batch_parallel(self, symbols: List[str]) -> List[PredictionResult]:
        """並列バッチ予測"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(symbols)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_index = {
                executor.submit(self.predict, symbol): idx
                for idx, symbol in enumerate(symbols)
            }

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result(timeout=self.config.timeout_seconds)
                except Exception as e:
                    symbol = symbols[idx]
                    self.logger.error(f"Parallel prediction failed for {symbol}: {e}")
                    results[idx] = self._create_fallback_result(symbol, 0.0, str(e))

        return [r for r in results if r is not None]

    # プロパティ

    def get_confidence(self, symbol: str) -> float:
        """信頼度取得（インターフェース準拠）"""
        return self._calculate_confidence(symbol, 50.0)  # デフォルト予測値で計算
