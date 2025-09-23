"""Compatibility layer exposing core predictor utilities for tests and legacy modules."""

from .core import (
    PredictionResult,
    PredictorInterface,
    StockPredictor,
    EnsembleStockPredictor,
    CacheablePredictor,
    MLStockPredictor,
)
from .performance import ParallelStockPredictor, UltraHighPerformancePredictor
from .recommendation import StockRecommendation

__all__ = [
    "PredictionResult",
    "PredictorInterface",
    "StockPredictor",
    "EnsembleStockPredictor",
    "CacheablePredictor",
    "MLStockPredictor",
    "ParallelStockPredictor",
    "UltraHighPerformancePredictor",
]

