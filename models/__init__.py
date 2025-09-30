"""Compatibility layer exposing core predictor utilities for tests and legacy modules."""

from .core import (
    CacheablePredictor,
    EnsembleStockPredictor,
    MLStockPredictor,
    PredictionResult,
    PredictorInterface,
    StockPredictor,
)
from .performance import ParallelStockPredictor, UltraHighPerformancePredictor
from .recommendation import StockRecommendation
from .cache import AdvancedCacheManager, RedisCache
from .deep_learning import DeepLearningPredictor, DQNReinforcementLearner
from .ensemble_predictor import (
    AdvancedEnsemblePredictor,
    EnsembleStockPredictor as LegacyEnsembleStockPredictor,
    ParallelStockPredictor as LegacyParallelStockPredictor,
)
from .meta_learning import MetaLearningOptimizer
from .ml_stock_predictor import (
    HyperparameterOptimizer,
    MLStockPredictor as AdvancedMLStockPredictor,
    ModelPerformanceMonitor,
)
from .sentiment import MacroEconomicDataProvider, SentimentAnalyzer

__all__ = [
    "PredictionResult",
    "PredictorInterface",
    "StockPredictor",
    "EnsembleStockPredictor",
    "CacheablePredictor",
    "MLStockPredictor",
    "ParallelStockPredictor",
    "UltraHighPerformancePredictor",
    "AdvancedMLStockPredictor",
    "HyperparameterOptimizer",
    "ModelPerformanceMonitor",
    "LegacyEnsembleStockPredictor",
    "AdvancedEnsemblePredictor",
    "LegacyParallelStockPredictor",
    "AdvancedCacheManager",
    "RedisCache",
    "DeepLearningPredictor",
    "DQNReinforcementLearner",
    "MetaLearningOptimizer",
    "SentimentAnalyzer",
    "MacroEconomicDataProvider",
]
