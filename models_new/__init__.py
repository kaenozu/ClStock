#!/usr/bin/env python3
"""
ClStock新世代モジュラーMLモデルシステム
高精度予測・アンサンブル・深層学習・監視システム統合パッケージ
"""

# 基底インターフェース
from .base.interfaces import (
    StockPredictor,
    PredictionResult,
    ModelPerformance,
    DataProvider,
    ModelTrainer,
    CacheManager,
    PerformanceMonitor
)

# 精密予測システム
from .precision.precision_87_system import Precision87BreakthroughSystem

# アンサンブル予測システム
from .ensemble.ensemble_predictor import EnsembleStockPredictor

# 深層学習予測システム
from .deep_learning.deep_predictor import DeepLearningPredictor

# 監視・管理システム
from .monitoring.cache_manager import AdvancedCacheManager
from .monitoring.performance_monitor import ModelPerformanceMonitor

__version__ = "2.0.0"
__author__ = "ClStock Development Team"

__all__ = [
    # インターフェース
    'StockPredictor',
    'PredictionResult',
    'ModelPerformance',
    'DataProvider',
    'ModelTrainer',
    'CacheManager',
    'PerformanceMonitor',

    # 予測システム
    'Precision87BreakthroughSystem',
    'EnsembleStockPredictor',
    'DeepLearningPredictor',

    # 監視・管理
    'AdvancedCacheManager',
    'ModelPerformanceMonitor'
]


def get_available_predictors():
    """利用可能な予測器一覧を取得"""
    return {
        'precision_87': Precision87BreakthroughSystem,
        'ensemble': EnsembleStockPredictor,
        'deep_learning': DeepLearningPredictor
    }


def create_predictor(predictor_type: str, **kwargs):
    """予測器ファクトリー関数"""
    predictors = get_available_predictors()

    if predictor_type not in predictors:
        raise ValueError(f"Unknown predictor type: {predictor_type}. Available: {list(predictors.keys())}")

    return predictors[predictor_type](**kwargs)


def create_monitoring_suite():
    """監視システム一式を作成"""
    return {
        'cache_manager': AdvancedCacheManager(),
        'performance_monitor': ModelPerformanceMonitor()
    }