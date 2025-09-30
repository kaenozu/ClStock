"""Compatibility shim for legacy imports of advanced precision predictors."""

from .advanced_precision import AdvancedPrecisionBreakthrough87System
from .precision_breakthrough import Precision87BreakthroughSystem
from .ultra_high_performance import UltraHighPerformancePredictor

__all__ = [
    "AdvancedPrecisionBreakthrough87System",
    "Precision87BreakthroughSystem",
    "UltraHighPerformancePredictor",
]
