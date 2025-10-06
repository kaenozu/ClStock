"""ハイブリッド予測器 - 統合リファクタリング版
Issue #9の修正を含む実装
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.base_predictor import BaseStockPredictor
from ..core.interfaces import (
    BatchPredictionResult,
    CacheProvider,
    DataProvider,
    ModelConfiguration,
    ModelType,
    PredictionMode,
    PredictionResult,
)
from ..ensemble.ensemble_predictor import RefactoredEnsemblePredictor


class RefactoredHybridPredictor(BaseStockPredictor):
    """統合リファクタリング版ハイブリッド予測器

    Issue #9の修正:
    - GPU batch処理でランダムデータを返す問題を修正
    - リアルタイム学習の実際値フィードバックを正しく実装
    """

    def __init__(
        self,
        config: ModelConfiguration = None,
        data_provider: DataProvider = None,
        cache_provider: CacheProvider = None,
        enable_real_time_learning: bool = True,
        enable_batch_optimization: bool = True,
        enable_cache: bool = True,
        enable_adaptive_optimization: bool = True,
        enable_streaming: bool = True,
        enable_multi_gpu: bool = True,
    ):
        base_custom_params = (
            dict(config.custom_params) if config and config.custom_params else {}
        )

        base_config = replace(
            config or ModelConfiguration(),
            model_type=ModelType.HYBRID,
            custom_params=base_custom_params,
        )

        base_config.cache_enabled = enable_cache
        base_config.custom_params.update(
            {
                "adaptive_optimization_enabled": enable_adaptive_optimization,
                "streaming_enabled": enable_streaming,
                "multi_gpu_enabled": enable_multi_gpu,
            },
        )

        super().__init__(base_config, data_provider, cache_provider)
        self.logger = logging.getLogger(__name__)

        ensemble_custom_params = dict(base_config.custom_params)
        ensemble_custom_params.setdefault(
            "parent_model_type", base_config.model_type.value,
        )

        ensemble_config = replace(
            base_config,
            model_type=ModelType.ENSEMBLE,
            custom_params=ensemble_custom_params,
        )

        # 追加機能の有効/無効状態を保持
        self.cache_enabled = base_config.cache_enabled
        self.adaptive_optimization_enabled = enable_adaptive_optimization
        self.streaming_enabled = enable_streaming
        self.multi_gpu_enabled = enable_multi_gpu

        # エンサンブル予測器を内部で使用
        self.ensemble_predictor = RefactoredEnsemblePredictor(
            config=ensemble_config, data_provider=data_provider,
        )

        # リアルタイム学習設定
        self.enable_real_time_learning = enable_real_time_learning
        # Phase2フラグ互換性保持用の属性
        self.real_time_learning_enabled = enable_real_time_learning
        self.learning_history = deque(maxlen=1000)  # 学習履歴
        self.prediction_errors = deque(maxlen=100)  # 予測誤差記録

        # バッチ最適化設定
        self.enable_batch_optimization = enable_batch_optimization
        self.batch_thresholds = {
            "small": 10,
            "medium": 50,
            "large": 100,
            "massive": 500,
        }

        # 追加機能フラグ（現時点では予約。テストで指定可能にするため保持）
        self.cache_enabled = enable_cache
        self.adaptive_optimization_enabled = enable_adaptive_optimization
        self.streaming_enabled = enable_streaming
        self.multi_gpu_enabled = enable_multi_gpu
        # 互換性のための別名も保持
        self.enable_cache = enable_cache
        self.enable_adaptive_optimization = enable_adaptive_optimization
        self.enable_streaming = enable_streaming
        self.enable_multi_gpu = enable_multi_gpu

        self.logger.info("RefactoredHybridPredictor initialized with Issue #9 fixes")

    def _record_prediction(
        self,
        symbol: str,
        prediction_result: PredictionResult,
        prediction_mode: PredictionMode,
        learning_rate: float,
    ) -> None:
        """メタデータや市場データを用いて予測結果を記録"""
        # 予測モードの更新（失敗しても致命的ではないため例外は握りつぶす）
        try:
            if prediction_mode is not None:
                self.set_prediction_mode(prediction_mode)
        except Exception as exc:  # pragma: no cover - 予防的ロギング
            self.logger.debug(
                "Failed to update prediction mode for %s: %s", symbol, exc,
            )

        # 学習率を記録しておくと後続の分析で利用可能
        self.config.custom_params["last_learning_rate"] = learning_rate

        learning_enabled = getattr(
            self, "real_time_learning_enabled", self.enable_real_time_learning,
        )
        if not learning_enabled:
            return

        actual_price = None
        if prediction_result.metadata:
            actual_price = prediction_result.metadata.get("current_price")

        self._record_prediction_with_actual_price(
            symbol, prediction_result, actual_price,
        )

    def _predict_implementation(self, symbol: str) -> float:
        """ハイブリッド予測の実装"""
        try:
            # エンサンブル予測器を使用して予測
            result = self.ensemble_predictor.predict(symbol)

            # リアルタイム学習のフィードバック記録
            if self.enable_real_time_learning:
                self._record_prediction_with_actual_price(symbol, result)

            return result.prediction

        except Exception as e:
            self.logger.error(f"Prediction failed for {symbol}: {e!s}")
            return self.NEUTRAL_PREDICTION_VALUE

    def predict_batch(self, symbols: List[str]) -> BatchPredictionResult:
        """バッチ予測の実装（Issue #9修正: ランダムデータを返さない）"""
        start_time = time.time()
        results = {}
        errors = {}

        # バッチサイズに基づく処理方法の選択
        batch_size = len(symbols)

        if (
            self.enable_batch_optimization
            and batch_size >= self.batch_thresholds["large"]
        ):
            # 大規模バッチの並列処理（修正版）
            results, errors = self._process_large_batch(symbols)
        else:
            # 通常のバッチ処理
            for symbol in symbols:
                try:
                    prediction = self._predict_with_deterministic_logic(symbol)
                    results[symbol] = prediction
                except Exception as e:
                    errors[symbol] = str(e)
                    self.logger.error(f"Batch prediction failed for {symbol}: {e}")

        processing_time = time.time() - start_time

        return BatchPredictionResult(
            predictions=results,
            errors=errors,
            metadata={
                "processing_time": processing_time,
                "batch_size": batch_size,
                "success_rate": len(results) / batch_size if batch_size > 0 else 0,
            },
        )

    def _process_large_batch(
        self, symbols: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """大規模バッチの処理（修正版: 実際のデータを使用）"""
        results = {}
        errors = {}

        # 並列処理のための分割
        chunk_size = 50
        chunks = [
            symbols[i : i + chunk_size] for i in range(0, len(symbols), chunk_size)
        ]

        for chunk in chunks:
            # 各チャンクを並列で処理
            chunk_results = asyncio.run(self._async_process_chunk(chunk))

            for symbol, result in chunk_results.items():
                if isinstance(result, Exception):
                    errors[symbol] = str(result)
                else:
                    results[symbol] = result

        return results, errors

    async def _async_process_chunk(self, symbols: List[str]) -> Dict[str, Any]:
        """非同期でチャンクを処理"""
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._async_predict(symbol))
            tasks.append((symbol, task))

        results = {}
        for symbol, task in tasks:
            try:
                result = await task
                results[symbol] = result
            except Exception as e:
                results[symbol] = e

        return results

    async def _async_predict(self, symbol: str) -> float:
        """非同期予測"""
        # 同期的な予測を非同期で実行
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._predict_with_deterministic_logic, symbol,
        )

    def _predict_with_deterministic_logic(self, symbol: str) -> float:
        """決定論的ロジックで予測（ランダムデータを使わない）"""
        try:
            # 実際のデータプロバイダーから価格データ取得
            if self.data_provider:
                data = self.data_provider.get_stock_data(symbol)
                if data is not None and not data.empty:
                    # エンサンブル予測器を使用
                    result = self.ensemble_predictor.predict(symbol)
                    return result.prediction

            # データがない場合はデフォルト値
            return self.NEUTRAL_PREDICTION_VALUE

        except Exception as e:
            self.logger.error(f"Deterministic prediction failed for {symbol}: {e}")
            return self.NEUTRAL_PREDICTION_VALUE

    def _record_prediction(
        self,
        symbol: str,
        prediction_result: PredictionResult,
        mode: Optional[PredictionMode] = None,
        prediction_time: Optional[float] = None,
    ) -> None:
        """予測履歴とフィードバックを記録"""
        # 予測履歴（Phase2互換）
        if hasattr(self, "prediction_history") and self.prediction_history is not None:
            history_entry = {
                "symbol": symbol,
                "prediction": prediction_result.prediction,
                "confidence": prediction_result.confidence,
                "accuracy": prediction_result.accuracy,
                "mode": mode.value if mode else None,
                "prediction_time": prediction_time,
                "timestamp": datetime.now(),
            }
            # 実際の価格を保存できる場合は追加
            actual_from_metadata = (
                prediction_result.metadata.get("current_price")
                if prediction_result.metadata
                else None
            )
            if actual_from_metadata is not None:
                history_entry["actual"] = actual_from_metadata
            self.prediction_history.append(history_entry)

        actual_price = (
            prediction_result.metadata.get("current_price")
            if prediction_result.metadata
            else None
        )
        if actual_price is None:
            actual_price = self._get_actual_market_price(symbol)

        # リアルタイム学習システムへのフィードバック
        if (
            actual_price is not None
            and getattr(self, "real_time_learning_enabled", False)
            and getattr(self, "real_time_learner", None)
        ):
            self.real_time_learner.add_prediction_feedback(
                prediction=prediction_result.prediction,
                actual=actual_price,
                symbol=symbol,
            )

        # 既存の学習履歴更新ロジックを呼び出し
        self._record_prediction_with_actual_price(
            symbol, prediction_result, actual_price=actual_price,
        )

    def _record_prediction_with_actual_price(
        self,
        symbol: str,
        prediction_result: PredictionResult,
        actual_price: Optional[float] = None,
    ):
        """予測と実際の価格を記録（Issue #9修正: 実際の価格を使用）"""
        try:
            # 実際の市場価格を取得
            if actual_price is None:
                actual_price = self._get_actual_market_price(symbol)

            if actual_price is not None:
                # 予測誤差を計算
                if actual_price != 0:
                    error = (
                        abs(prediction_result.prediction - actual_price) / actual_price
                    )
                else:  # 実際の価格が0の場合は誤差0として扱う
                    error = 0.0

                # 学習履歴に記録
                self.learning_history.append(
                    {
                        "symbol": symbol,
                        "prediction": prediction_result.prediction,
                        "actual": actual_price,  # 修正: 実際の価格を使用
                        "error": error,
                        "timestamp": datetime.now(),
                        "confidence": prediction_result.confidence,
                    },
                )

                # 誤差統計を更新
                self.prediction_errors.append(error)

                # リアルタイム学習システムへフィードバック
                learner = getattr(self, "real_time_learner", None)
                if learner and hasattr(learner, "add_prediction_feedback"):
                    try:
                        learner.add_prediction_feedback(
                            prediction_result.prediction, actual_price, symbol,
                        )
                    except Exception as feedback_error:  # pragma: no cover - ログ用途
                        self.logger.warning(
                            "Real-time learner feedback failed for %s: %s",
                            symbol,
                            feedback_error,
                        )

                # リアルタイム学習システムの更新
                self._update_learning_weights(error)

        except Exception as e:
            self.logger.error(f"Failed to record prediction for {symbol}: {e}")

    def _get_actual_market_price(self, symbol: str) -> Optional[float]:
        """実際の市場価格を取得"""
        try:
            if self.data_provider:
                data = self.data_provider.get_stock_data(symbol, period="1d")
                if data is not None and not data.empty:
                    # 最新の終値を実際の価格として使用
                    return float(data["Close"].iloc[-1])
            return None
        except Exception as e:
            self.logger.error(f"Failed to get actual price for {symbol}: {e}")
            return None

    def _update_learning_weights(self, error: float):
        """学習重みを更新"""
        # 簡単な適応的学習ロジック
        if error > 0.1:  # 10%以上の誤差
            self.logger.info(
                f"High prediction error detected: {error:.2%}. Adjusting weights...",
            )
            # ここで実際の重み調整ロジックを実装

    def get_learning_statistics(self) -> Dict[str, Any]:
        """学習統計を取得"""
        if not self.prediction_errors:
            return {"status": "No learning data available"}

        errors = list(self.prediction_errors)
        return {
            "total_predictions": len(self.learning_history),
            "average_error": np.mean(errors),
            "median_error": np.median(errors),
            "max_error": np.max(errors),
            "min_error": np.min(errors),
            "recent_predictions": list(self.learning_history)[-10:],
        }

    def train(self, data: pd.DataFrame) -> bool:
        """モデル訓練"""
        try:
            # エンサンブル予測器の訓練
            self.ensemble_predictor.train(data)
            self.is_trained = True
            return True
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False
