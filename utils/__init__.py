"""
ユーティリティパッケージ
"""

from .exceptions import (
    ClStockException,
    DataFetchError,
    InsufficientDataError,
    ModelNotTrainedError,
    InvalidSymbolError,
    ModelTrainingError,
    PredictionError,
    BacktestError,
    ConfigurationError,
    APIError,
    ValidationError,
    NetworkError,
    FileOperationError,
)

__all__ = [
    "ClStockException",
    "DataFetchError",
    "InsufficientDataError",
    "ModelNotTrainedError",
    "InvalidSymbolError",
    "ModelTrainingError",
    "PredictionError",
    "BacktestError",
    "ConfigurationError",
    "APIError",
    "ValidationError",
    "NetworkError",
    "FileOperationError",
]

try:
    from .cache import (
        DataCache,
        cached,
        cache_dataframe,
        get_cache,
        clear_cache,
        cleanup_cache,
    )
except ModuleNotFoundError:  # pragma: no cover - optional pandas dependency
    DataCache = None
    cached = None
    cache_dataframe = None
    get_cache = None
    clear_cache = None
    cleanup_cache = None
else:
    __all__.extend(
        [
            "DataCache",
            "cached",
            "cache_dataframe",
            "get_cache",
            "clear_cache",
            "cleanup_cache",
        ]
    )
