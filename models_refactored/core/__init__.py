"""
Core interfaces and base classes for ClStock Models
"""

from .interfaces import (
    StockPredictor,
    PredictionResult,
    DataProvider,
    ModelConfiguration,
    PerformanceMetrics,
)

from .factory import PredictorFactory
from .manager import ModelManager
from .base_predictor import BaseStockPredictor

__all__ = [
    "StockPredictor",
    "PredictionResult",
    "DataProvider",
    "ModelConfiguration",
    "PerformanceMetrics",
    "PredictorFactory",
    "ModelManager",
    "BaseStockPredictor",
]
