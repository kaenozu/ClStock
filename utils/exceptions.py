"""
ClStock カスタム例外クラス
より具体的なエラーハンドリングのために定義
"""

from typing import Optional


class ClStockException(Exception):
    """ClStock基底例外クラス"""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class DataFetchError(ClStockException):
    """データ取得エラー"""
    def __init__(self, symbol: str, message: str = "Failed to fetch data", details: Optional[str] = None):
        self.symbol = symbol
        super().__init__(f"Data fetch error for {symbol}: {message}", details)


class InsufficientDataError(ClStockException):
    """データ不足エラー"""
    def __init__(self, symbol: str, available: int, required: int):
        self.symbol = symbol
        self.available = available
        self.required = required
        super().__init__(
            f"Insufficient data for {symbol}: {available} available, {required} required"
        )


class ModelNotTrainedError(ClStockException):
    """モデル未訓練エラー"""
    def __init__(self, model_type: str):
        self.model_type = model_type
        super().__init__(f"Model not trained: {model_type}")


class InvalidSymbolError(ClStockException):
    """無効な銘柄コードエラー"""
    def __init__(self, symbol: str, valid_symbols: Optional[list] = None):
        self.symbol = symbol
        self.valid_symbols = valid_symbols
        message = f"Invalid symbol: {symbol}"
        if valid_symbols:
            message += f". Valid symbols: {', '.join(valid_symbols[:5])}..."
        super().__init__(message)


class ModelTrainingError(ClStockException):
    """モデル訓練エラー"""
    def __init__(self, model_type: str, details: Optional[str] = None):
        self.model_type = model_type
        super().__init__(f"Model training failed for {model_type}", details)


class PredictionError(ClStockException):
    """予測エラー"""
    def __init__(self, symbol: str, model_type: str, details: Optional[str] = None):
        self.symbol = symbol
        self.model_type = model_type
        super().__init__(
            f"Prediction failed for {symbol} using {model_type}",
            details
        )


class BacktestError(ClStockException):
    """バックテストエラー"""
    def __init__(self, period: str, details: Optional[str] = None):
        self.period = period
        super().__init__(f"Backtest failed for period {period}", details)


class ConfigurationError(ClStockException):
    """設定エラー"""
    def __init__(self, config_name: str, details: Optional[str] = None):
        self.config_name = config_name
        super().__init__(f"Configuration error: {config_name}", details)


class APIError(ClStockException):
    """API関連エラー"""
    def __init__(self, endpoint: str, status_code: Optional[int] = None, details: Optional[str] = None):
        self.endpoint = endpoint
        self.status_code = status_code
        message = f"API error at {endpoint}"
        if status_code:
            message += f" (Status: {status_code})"
        super().__init__(message, details)


class ValidationError(ClStockException):
    """バリデーションエラー"""
    def __init__(self, field: str, value: any, constraint: str):
        self.field = field
        self.value = value
        self.constraint = constraint
        super().__init__(
            f"Validation failed for {field}: '{value}' {constraint}"
        )


class NetworkError(ClStockException):
    """ネットワークエラー"""
    def __init__(self, url: Optional[str] = None, details: Optional[str] = None):
        self.url = url
        message = "Network connection error"
        if url:
            message += f" for {url}"
        super().__init__(message, details)


class FileOperationError(ClStockException):
    """ファイル操作エラー"""
    def __init__(self, operation: str, file_path: str, details: Optional[str] = None):
        self.operation = operation
        self.file_path = file_path
        super().__init__(
            f"File {operation} failed for {file_path}",
            details
        )