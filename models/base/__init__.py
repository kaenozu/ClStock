"""Base interfaces and data structures for the consolidated models package."""

from .interfaces import (
    CacheManager,
    DataProvider,
    IndexData,
    ModelPerformance,
    ModelTrainer,
    NewsData,
    OrderBookData,
    PerformanceMonitor,
    PredictionResult,
    StockPredictor,
    TickData,
)

__all__ = [
    "CacheManager",
    "DataProvider",
    "IndexData",
    "ModelPerformance",
    "ModelTrainer",
    "NewsData",
    "OrderBookData",
    "PerformanceMonitor",
    "PredictionResult",
    "StockPredictor",
    "TickData",
]
