"""Compatibility wrapper around :mod:`models.core.interfaces`."""

from __future__ import annotations

from models.core.interfaces import (  # type: ignore F401
    BatchPredictionResult,
    CacheProvider,
    DataProvider,
    ModelConfiguration,
    ModelType,
    PerformanceMetrics,
    PredictionMode,
    PredictionResult,
    StockPredictor,
)

__all__ = [
    "BatchPredictionResult",
    "CacheProvider",
    "DataProvider",
    "ModelConfiguration",
    "ModelType",
    "PerformanceMetrics",
    "PredictionMode",
    "PredictionResult",
    "StockPredictor",
]
