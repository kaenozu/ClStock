"""
カスタム例外のテスト
"""

import pytest
from utils.exceptions import (
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


class TestClStockException:
    """ClStockException基底クラスのテスト"""

    def test_basic_exception(self):
        """基本的な例外のテスト"""
        message = "Test error message"
        exc = ClStockException(message)

        assert str(exc) == message
        assert exc.message == message
        assert exc.details is None

    def test_exception_with_details(self):
        """詳細付き例外のテスト"""
        message = "Test error"
        details = "Additional details"
        exc = ClStockException(message, details)

        assert exc.message == message
        assert exc.details == details
        assert str(exc) == f"{message} - Details: {details}"


class TestDataFetchError:
    """DataFetchErrorのテスト"""

    def test_data_fetch_error_basic(self):
        """基本的なDataFetchErrorのテスト"""
        symbol = "7203"
        exc = DataFetchError(symbol)

        assert exc.symbol == symbol
        assert "7203" in str(exc)
        assert "Failed to fetch data" in str(exc)

    def test_data_fetch_error_custom_message(self):
        """カスタムメッセージ付きDataFetchErrorのテスト"""
        symbol = "7203"
        message = "Network timeout"
        details = "Connection reset"
        exc = DataFetchError(symbol, message, details)

        assert exc.symbol == symbol
        assert symbol in str(exc)
        assert message in str(exc)


class TestInsufficientDataError:
    """InsufficientDataErrorのテスト"""

    def test_insufficient_data_error(self):
        """InsufficientDataErrorのテスト"""
        symbol = "7203"
        available = 50
        required = 100
        exc = InsufficientDataError(symbol, available, required)

        assert exc.symbol == symbol
        assert exc.available == available
        assert exc.required == required
        assert str(available) in str(exc)
        assert str(required) in str(exc)


class TestModelNotTrainedError:
    """ModelNotTrainedErrorのテスト"""

    def test_model_not_trained_error(self):
        """ModelNotTrainedErrorのテスト"""
        model_type = "xgboost"
        exc = ModelNotTrainedError(model_type)

        assert exc.model_type == model_type
        assert model_type in str(exc)
        assert "not trained" in str(exc)


class TestInvalidSymbolError:
    """InvalidSymbolErrorのテスト"""

    def test_invalid_symbol_error_basic(self):
        """基本的なInvalidSymbolErrorのテスト"""
        symbol = "INVALID"
        exc = InvalidSymbolError(symbol)

        assert exc.symbol == symbol
        assert symbol in str(exc)
        assert "Invalid symbol" in str(exc)

    def test_invalid_symbol_error_with_valid_list(self):
        """有効銘柄リスト付きInvalidSymbolErrorのテスト"""
        symbol = "INVALID"
        valid_symbols = ["7203", "6758", "9984"]
        exc = InvalidSymbolError(symbol, valid_symbols)

        assert exc.symbol == symbol
        assert exc.valid_symbols == valid_symbols
        assert symbol in str(exc)
        assert "7203" in str(exc)


class TestModelTrainingError:
    """ModelTrainingErrorのテスト"""

    def test_model_training_error(self):
        """ModelTrainingErrorのテスト"""
        model_type = "xgboost"
        details = "Insufficient training data"
        exc = ModelTrainingError(model_type, details)

        assert exc.model_type == model_type
        assert model_type in str(exc)
        assert "training failed" in str(exc)


class TestPredictionError:
    """PredictionErrorのテスト"""

    def test_prediction_error(self):
        """PredictionErrorのテスト"""
        symbol = "7203"
        model_type = "xgboost"
        details = "Feature mismatch"
        exc = PredictionError(symbol, model_type, details)

        assert exc.symbol == symbol
        assert exc.model_type == model_type
        assert symbol in str(exc)
        assert model_type in str(exc)
        assert "Prediction failed" in str(exc)


class TestAPIError:
    """APIErrorのテスト"""

    def test_api_error_basic(self):
        """基本的なAPIErrorのテスト"""
        endpoint = "/api/v1/recommendations"
        exc = APIError(endpoint)

        assert exc.endpoint == endpoint
        assert endpoint in str(exc)
        assert "API error" in str(exc)

    def test_api_error_with_status(self):
        """ステータスコード付きAPIErrorのテスト"""
        endpoint = "/api/v1/recommendations"
        status_code = 500
        exc = APIError(endpoint, status_code)

        assert exc.endpoint == endpoint
        assert exc.status_code == status_code
        assert str(status_code) in str(exc)


class TestValidationError:
    """ValidationErrorのテスト"""

    def test_validation_error(self):
        """ValidationErrorのテスト"""
        field = "top_n"
        value = 15
        constraint = "must be between 1 and 10"
        exc = ValidationError(field, value, constraint)

        assert exc.field == field
        assert exc.value == value
        assert exc.constraint == constraint
        assert field in str(exc)
        assert str(value) in str(exc)
        assert constraint in str(exc)


class TestNetworkError:
    """NetworkErrorのテスト"""

    def test_network_error_basic(self):
        """基本的なNetworkErrorのテスト"""
        exc = NetworkError()

        assert "Network connection error" in str(exc)

    def test_network_error_with_url(self):
        """URL付きNetworkErrorのテスト"""
        url = "https://query1.finance.yahoo.com"
        details = "Connection timeout"
        exc = NetworkError(url, details)

        assert exc.url == url
        assert url in str(exc)


class TestFileOperationError:
    """FileOperationErrorのテスト"""

    def test_file_operation_error(self):
        """FileOperationErrorのテスト"""
        operation = "write"
        file_path = "/path/to/model.pkl"
        details = "Permission denied"
        exc = FileOperationError(operation, file_path, details)

        assert exc.operation == operation
        assert exc.file_path == file_path
        assert operation in str(exc)
        assert file_path in str(exc)


class TestExceptionInheritance:
    """例外の継承関係のテスト"""

    def test_all_exceptions_inherit_from_base(self):
        """すべての例外が基底クラスを継承していることを確認"""
        exceptions = [
            DataFetchError("7203"),
            InsufficientDataError("7203", 50, 100),
            ModelNotTrainedError("xgboost"),
            InvalidSymbolError("INVALID"),
            ModelTrainingError("xgboost"),
            PredictionError("7203", "xgboost"),
            BacktestError("2023-2024"),
            ConfigurationError("api_key"),
            APIError("/api/test"),
            ValidationError("field", "value", "constraint"),
            NetworkError(),
            FileOperationError("read", "/test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, ClStockException)
            assert isinstance(exc, Exception)
