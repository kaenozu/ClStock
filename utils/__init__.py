"""ユーティリティパッケージ
"""

from .exceptions import (
    APIError,
    BacktestError,
    ClStockException,
    ConfigurationError,
    DataFetchError,
    FileOperationError,
    InsufficientDataError,
    InvalidSymbolError,
    ModelNotTrainedError,
    ModelTrainingError,
    NetworkError,
    PredictionError,
    ValidationError,
)

__all__ = [
    "APIError",
    "BacktestError",
    "ClStockException",
    "ConfigurationError",
    "DataFetchError",
    "FileOperationError",
    "InsufficientDataError",
    "InvalidSymbolError",
    "ModelNotTrainedError",
    "ModelTrainingError",
    "NetworkError",
    "PredictionError",
    "ValidationError",
]

try:
    from .cache import (
        DataCache,
        cache_dataframe,
        cached,
        cleanup_cache,
        clear_cache,
        get_cache,
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
            "cache_dataframe",
            "cached",
            "cleanup_cache",
            "clear_cache",
            "get_cache",
        ],
    )
