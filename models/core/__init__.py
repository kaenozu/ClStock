"""Core interfaces and base classes for ClStock Models
"""

from .base_predictor import BaseStockPredictor
from .factory import PredictorFactory
from .interfaces import (
    DataProvider,
    ModelConfiguration,
    PerformanceMetrics,
    PredictionResult,
    StockPredictor,
)
from .manager import ModelManager

__all__ = [
    "BaseStockPredictor",
    "DataProvider",
    "ModelConfiguration",
    "ModelManager",
    "PerformanceMetrics",
    "PredictionResult",
    "PredictorFactory",
    "StockPredictor",
]
