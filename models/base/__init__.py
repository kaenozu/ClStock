"""Base interfaces and data structures for the consolidated models package."""

from .interfaces import (
    PredictionResult,
    ModelPerformance,
    StockPredictor,
    DataProvider,
    ModelTrainer,
    CacheManager,
    PerformanceMonitor,
    TickData,
    OrderBookData,
    IndexData,
    NewsData,
)

__all__ = [
    "PredictionResult",
    "ModelPerformance",
    "StockPredictor",
    "DataProvider",
    "ModelTrainer",
    "CacheManager",
    "PerformanceMonitor",
    "TickData",
    "OrderBookData",
    "IndexData",
    "NewsData",
]
