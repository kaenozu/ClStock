"""Ultra high performance predictor module."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List

from .cache import RedisCache
from .deep_learning import DeepLearningPredictor
from .ensemble_predictor import EnsembleStockPredictor, ParallelStockPredictor
from .meta_learning import MetaLearningOptimizer
from .ml_stock_predictor import ModelPerformanceMonitor
from .sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)


class UltraHighPerformancePredictor:
    """Integrated ultra high performance prediction system."""

    def __init__(self) -> None:
        self.ensemble_predictor = EnsembleStockPredictor()
        self.deep_lstm = DeepLearningPredictor("lstm")
        self.deep_transformer = DeepLearningPredictor("transformer")
        self.sentiment_analyzer = SentimentAnalyzer()
        self.redis_cache = RedisCache()
        self.meta_optimizer = MetaLearningOptimizer()
        self.parallel_predictor: ParallelStockPredictor | None = None
        self.performance_monitor = ModelPerformanceMonitor()
        self.model_weights = {
            "ensemble": 0.4,
            "deep_lstm": 0.25,
            "deep_transformer": 0.25,
            "sentiment": 0.1,
        }

    def train_all_models(self, symbols: List[str]) -> None:
        from concurrent.futures import ThreadPoolExecutor

        logger.info("Training ultra-high performance prediction system...")

        def train_ensemble():
            self.ensemble_predictor.train_ensemble(symbols)

        def train_lstm():
            self.deep_lstm.train_deep_model(symbols)

        def train_transformer():
            self.deep_transformer.train_deep_model(symbols)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(train_ensemble),
                executor.submit(train_lstm),
                executor.submit(train_transformer),
            ]
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    logger.error(f"Training error: {exc}")
        self.parallel_predictor = ParallelStockPredictor(self.ensemble_predictor)
        logger.info("Ultra-high performance system training completed!")

    def ultra_predict(self, symbol: str) -> float:
        cache_key = f"ultra_pred_{symbol}_{datetime.now().strftime('%Y-%m-%d-%H')}"
        cached_result = self.redis_cache.get(cache_key)
        if cached_result:
            return float(cached_result)
        try:
            data = self.ensemble_predictor.data_provider.get_stock_data(symbol, "1y")
            best_model = self.meta_optimizer.select_best_model(symbol, data)
            predictions: Dict[str, float] = {}
            try:
                predictions["ensemble"] = self.ensemble_predictor.predict_score(symbol)
            except Exception:
                predictions["ensemble"] = 50.0
            try:
                predictions["deep_lstm"] = self.deep_lstm.predict_deep(symbol)
            except Exception:
                predictions["deep_lstm"] = 50.0
            try:
                predictions["deep_transformer"] = self.deep_transformer.predict_deep(
                    symbol,
                )
            except Exception:
                predictions["deep_transformer"] = 50.0
            try:
                sentiment_data = self.sentiment_analyzer.get_news_sentiment(symbol)
                macro_data = self.sentiment_analyzer.get_macro_economic_features()
                sentiment_score = (
                    50
                    + (
                        sentiment_data["positive_ratio"]
                        - sentiment_data["negative_ratio"]
                    )
                    * 50
                )
                sentiment_score += macro_data["gdp_growth"] * 100
                sentiment_score += (140 - macro_data["exchange_rate_usd_jpy"]) * 0.5
                predictions["sentiment"] = max(0, min(100, sentiment_score))
            except Exception:
                predictions["sentiment"] = 50.0
            if best_model in predictions:
                self.model_weights[best_model] *= 1.2
                total_weight = sum(self.model_weights.values())
                self.model_weights = {
                    k: v / total_weight for k, v in self.model_weights.items()
                }
            final_score = sum(
                predictions[model] * weight
                for model, weight in self.model_weights.items()
                if model in predictions
            )
            self.redis_cache.set(cache_key, str(final_score), ttl=3600)
            return float(final_score)
        except Exception as exc:
            logger.error(f"Ultra prediction error for {symbol}: {exc}")
            return 50.0
