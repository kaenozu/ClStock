"""
ClStock Models - 統合リファクタリング版
統一されたアーキテクチャによる高性能株価予測システム
"""

__version__ = "2.0.0"
__author__ = "ClStock Development Team"

# 統合されたメインインターfaces
from .core.interfaces import StockPredictor, PredictionResult, DataProvider
from .core.factory import PredictorFactory
from .core.manager import ModelManager

# 主要なモデル実装
from .ensemble.ensemble_predictor import RefactoredEnsemblePredictor
from .hybrid.hybrid_predictor import RefactoredHybridPredictor
from .deep_learning.deep_predictor import RefactoredDeepLearningPredictor


# Advanced modules from models_new integration
from .advanced.market_sentiment_analyzer import MarketSentimentAnalyzer
from .advanced.prediction_dashboard import PredictionDashboard
from .advanced.risk_management_framework import RiskManager
from .advanced.trading_strategy_generator import AutoTradingStrategyGenerator as TradingStrategyGenerator

# Monitoring modules
from .monitoring.performance_monitor import ModelPerformanceMonitor as PerformanceMonitor
from .monitoring.cache_manager import RealTimeCacheManager as CacheManager

# Precision modules
from .precision.precision_87_system import Precision87BreakthroughSystem as Precision87System

__all__ = [
    'StockPredictor',
    'PredictionResult',
    'DataProvider',
    'PredictorFactory',
    'ModelManager',
    'RefactoredEnsemblePredictor',
    'RefactoredHybridPredictor',
    'RefactoredDeepLearningPredictor',
    # Advanced features
    'MarketSentimentAnalyzer',
    'PredictionDashboard',
    'RiskManager',
    'TradingStrategyGenerator',
    # Monitoring features
    'PerformanceMonitor',
    'CacheManager',
    # Precision features
    'Precision87System'
]