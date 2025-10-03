"""Unified models package providing lazy access to legacy and refactored APIs."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

__all__ = [
    # Legacy exports
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
    "AdvancedPrecisionBreakthrough87System",
    "Precision87BreakthroughSystem",
    "UltraHighPerformanceSystem",
    # Refactored exports
    "BasePredictionResult",
    "ModelPerformance",
    "BaseStockPredictorInterface",
    "BaseDataProvider",
    "ModelTrainer",
    "BaseCacheManagerInterface",
    "BasePerformanceMonitorInterface",
    "TickData",
    "OrderBookData",
    "IndexData",
    "NewsData",
    "RefactoredStockPredictor",
    "RefactoredPredictionResult",
    "RefactoredDataProvider",
    "ModelConfiguration",
    "PerformanceMetrics",
    "PredictorFactory",
    "ModelManager",
    "BaseStockPredictor",
    "RefactoredEnsemblePredictor",
    "AdvancedRefactoredEnsemblePredictor",
    "RefactoredDeepLearningPredictor",
    "MarketSentimentAnalyzer",
    "PredictionDashboard",
    "RiskManager",
    "RiskLevel",
    "RiskType",
    "PortfolioRisk",
    "AutoTradingStrategyGenerator",
    "RefactoredPerformanceMonitor",
    "RealTimeCacheManager",
    "RefactoredAdvancedCacheManager",
    "RefactoredPrecision87System",
    "HybridStockPredictor",
    "PredictionMode",
    "IntelligentPredictionCache",
    "AdaptivePerformanceOptimizer",
    "UltraFastStreamingPredictor",
    "MultiGPUParallelPredictor",
    "RealTimeLearningSystem",
    "RefactoredHybridPredictor",
]

_EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    # Legacy exports
    "PredictionResult": ("models.legacy_core", "PredictionResult"),
    "PredictorInterface": ("models.legacy_core", "PredictorInterface"),
    "StockPredictor": ("models.legacy_core", "StockPredictor"),
    "EnsembleStockPredictor": ("models.legacy_core", "EnsembleStockPredictor"),
    "CacheablePredictor": ("models.legacy_core", "CacheablePredictor"),
    "MLStockPredictor": ("models.legacy_core", "MLStockPredictor"),
    "ParallelStockPredictor": ("models.performance", "ParallelStockPredictor"),
    "UltraHighPerformancePredictor": ("models.performance", "UltraHighPerformancePredictor"),
    "AdvancedMLStockPredictor": ("models.ml_stock_predictor", "MLStockPredictor"),
    "HyperparameterOptimizer": ("models.ml_stock_predictor", "HyperparameterOptimizer"),
    "ModelPerformanceMonitor": ("models.ml_stock_predictor", "ModelPerformanceMonitor"),
    "LegacyEnsembleStockPredictor": ("models.legacy_ensemble_predictor", "EnsembleStockPredictor"),
    "AdvancedEnsemblePredictor": ("models.legacy_ensemble_predictor", "AdvancedEnsemblePredictor"),
    "LegacyParallelStockPredictor": ("models.legacy_ensemble_predictor", "ParallelStockPredictor"),
    "AdvancedCacheManager": ("models.cache", "AdvancedCacheManager"),
    "RedisCache": ("models.cache", "RedisCache"),
    "DeepLearningPredictor": ("models.legacy_deep_learning", "DeepLearningPredictor"),
    "DQNReinforcementLearner": ("models.legacy_deep_learning", "DQNReinforcementLearner"),
    "MetaLearningOptimizer": ("models.meta_learning", "MetaLearningOptimizer"),
    "SentimentAnalyzer": ("models.sentiment", "SentimentAnalyzer"),
    "MacroEconomicDataProvider": ("models.sentiment", "MacroEconomicDataProvider"),
    "AdvancedPrecisionBreakthrough87System": (
        "models.advanced_precision",
        "AdvancedPrecisionBreakthrough87System",
    ),
    "Precision87BreakthroughSystem": (
        "models.precision_breakthrough",
        "Precision87BreakthroughSystem",
    ),
    "UltraHighPerformanceSystem": (
        "models.ultra_high_performance",
        "UltraHighPerformancePredictor",
    ),
    # Refactored exports
    "BasePredictionResult": ("models.base.interfaces", "PredictionResult"),
    "ModelPerformance": ("models.base.interfaces", "ModelPerformance"),
    "BaseStockPredictorInterface": ("models.base.interfaces", "StockPredictor"),
    "BaseDataProvider": ("models.base.interfaces", "DataProvider"),
    "ModelTrainer": ("models.base.interfaces", "ModelTrainer"),
    "BaseCacheManagerInterface": ("models.base.interfaces", "CacheManager"),
    "BasePerformanceMonitorInterface": ("models.base.interfaces", "PerformanceMonitor"),
    "TickData": ("models.base.interfaces", "TickData"),
    "OrderBookData": ("models.base.interfaces", "OrderBookData"),
    "IndexData": ("models.base.interfaces", "IndexData"),
    "NewsData": ("models.base.interfaces", "NewsData"),
    "RefactoredStockPredictor": ("models.core", "StockPredictor"),
    "RefactoredPredictionResult": ("models.core", "PredictionResult"),
    "RefactoredDataProvider": ("models.core", "DataProvider"),
    "ModelConfiguration": ("models.core", "ModelConfiguration"),
    "PerformanceMetrics": ("models.core", "PerformanceMetrics"),
    "PredictorFactory": ("models.core", "PredictorFactory"),
    "ModelManager": ("models.core", "ModelManager"),
    "BaseStockPredictor": ("models.core", "BaseStockPredictor"),
    "RefactoredEnsemblePredictor": (
        "models.ensemble.ensemble_predictor",
        "RefactoredEnsemblePredictor",
    ),
    "AdvancedRefactoredEnsemblePredictor": (
        "models.ensemble.ensemble_predictor",
        "RefactoredEnsemblePredictor",
    ),
    "RefactoredDeepLearningPredictor": (
        "models.deep_learning.deep_predictor",
        "RefactoredDeepLearningPredictor",
    ),
    "MarketSentimentAnalyzer": ("models.advanced.market_sentiment_analyzer", "MarketSentimentAnalyzer"),
    "PredictionDashboard": ("models.advanced.prediction_dashboard", "PredictionDashboard"),
    "RiskManager": ("models.advanced.risk_management_framework", "RiskManager"),
    "RiskLevel": ("models.advanced.risk_management_framework", "RiskLevel"),
    "RiskType": ("models.advanced.risk_management_framework", "RiskType"),
    "PortfolioRisk": ("models.advanced.risk_management_framework", "PortfolioRisk"),
    "AutoTradingStrategyGenerator": (
        "models.advanced.trading_strategy_generator",
        "AutoTradingStrategyGenerator",
    ),
    "RefactoredPerformanceMonitor": (
        "models.monitoring.performance_monitor",
        "ModelPerformanceMonitor",
    ),
    "RealTimeCacheManager": ("models.monitoring.cache_manager", "RealTimeCacheManager"),
    "RefactoredAdvancedCacheManager": (
        "models.monitoring.cache_manager",
        "AdvancedCacheManager",
    ),
    "RefactoredPrecision87System": (
        "models.precision.precision_87_system",
        "Precision87BreakthroughSystem",
    ),
    "HybridStockPredictor": ("models.hybrid.hybrid_predictor", "HybridStockPredictor"),
    "PredictionMode": ("models.hybrid.prediction_modes", "PredictionMode"),
    "IntelligentPredictionCache": (
        "models.hybrid.intelligent_cache",
        "IntelligentPredictionCache",
    ),
    "AdaptivePerformanceOptimizer": (
        "models.hybrid.adaptive_optimizer",
        "AdaptivePerformanceOptimizer",
    ),
    "UltraFastStreamingPredictor": (
        "models.hybrid.ultra_fast_streaming",
        "UltraFastStreamingPredictor",
    ),
    "MultiGPUParallelPredictor": (
        "models.hybrid.multi_gpu_processor",
        "MultiGPUParallelPredictor",
    ),
    "RealTimeLearningSystem": (
        "models.hybrid.multi_gpu_processor",
        "RealTimeLearningSystem",
    ),
    "RefactoredHybridPredictor": (
        "models.hybrid.refactored_hybrid_predictor",
        "RefactoredHybridPredictor",
    ),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'models' has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
