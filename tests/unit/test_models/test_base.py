"""Test base model classes."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

# 蜿､縺・odels繝・ぅ繝ｬ繧ｯ繝医Μ縺ｯ蟒・ｭ｢縺輔ｌ縺ｾ縺励◆
# from models.base import (
#     StockPredictor,
#     PredictorInterface,
#     EnsemblePredictor,
#     CacheablePredictor,
#     PredictionResult
# )

# 譁ｰ縺励＞繧､繝ｳ繝昴・繝医ヱ繧ｹ
from models.core import (
    CacheablePredictor,
    EnsembleStockPredictor,
    PredictionResult,
    PredictorInterface,
    StockPredictor,
)
from models_refactored.ensemble.ensemble_predictor import (
    RefactoredEnsemblePredictor as EnsemblePredictor,
)


class TestPredictionResult:
    """Test PredictionResult dataclass."""

    def test_prediction_result_creation(self):
        """Test creating a PredictionResult."""
        result = PredictionResult(
            prediction=75.5,
            confidence=0.85,
            timestamp=datetime.now(),
            metadata={"test": True},
        )

        assert result.prediction == 75.5
        assert result.confidence == 0.85
        assert isinstance(result.timestamp, datetime)
        assert result.metadata["test"] is True


class TestStockPredictor:
    """Test base StockPredictor class."""

    def test_initialization(self):
        """Test StockPredictor initialization."""
        predictor = StockPredictor("test_model")

        assert predictor.model_type == "test_model"
        assert not predictor.is_trained()
        assert predictor.model is None
        assert predictor.feature_names == []

    def test_validate_input_valid_data(self):
        """Test validate_input with valid data."""
        predictor = StockPredictor()
        data = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [102, 103],
                "low": [99, 100],
                "close": [101, 102],
                "volume": [1000, 1100],
            }
        )

        assert predictor.validate_input(data) is True

    def test_validate_input_invalid_data(self):
        """Test validate_input with invalid data."""
        predictor = StockPredictor()

        # Empty DataFrame
        assert predictor.validate_input(pd.DataFrame()) is False

        # None data
        assert predictor.validate_input(None) is False

        # Missing columns
        data = pd.DataFrame({"open": [100], "high": [102]})
        assert predictor.validate_input(data) is False

    def test_get_confidence(self):
        """Test get_confidence default implementation."""
        predictor = StockPredictor()
        assert predictor.get_confidence() == 0.5

    def test_predict_not_trained(self):
        """Test predict when model is not trained."""
        predictor = StockPredictor()
        data = pd.DataFrame(
            {
                "open": [100],
                "high": [102],
                "low": [99],
                "close": [101],
                "volume": [1000],
            }
        )

        with pytest.raises(ValueError, match="Model must be trained"):
            predictor.predict("TEST", data)

    def test_predict_invalid_data(self):
        """Test predict with invalid data."""
        predictor = StockPredictor()
        predictor._is_trained = True

        with pytest.raises(ValueError, match="Invalid input data"):
            predictor.predict("TEST", None)

    @patch.object(StockPredictor, "prepare_features")
    @patch.object(StockPredictor, "train")
    def test_predict_default_implementation(self, mock_train, mock_prepare_features):
        """Test default predict implementation."""
        predictor = StockPredictor()
        predictor._is_trained = True

        data = pd.DataFrame(
            {
                "open": [100],
                "high": [102],
                "low": [99],
                "close": [101],
                "volume": [1000],
            }
        )

        result = predictor.predict("TEST", data)

        assert isinstance(result, PredictionResult)
        assert result.prediction == 0.0
        assert result.confidence == 0.5
        assert result.metadata["model_type"] == "base"
        assert result.metadata["symbol"] == "TEST"


class TestEnsemblePredictor:
    """Test EnsemblePredictor class."""

    def test_initialization(self):
        """Test EnsemblePredictor initialization."""
        ensemble = EnsemblePredictor()

        assert ensemble.model_type == "ensemble"
        assert ensemble.models == []
        assert ensemble.weights == []

    def test_add_model(self):
        """Test adding models to ensemble."""
        ensemble = EnsemblePredictor()
        model1 = StockPredictor("model1")
        model2 = StockPredictor("model2")

        ensemble.add_model(model1, 0.6)
        ensemble.add_model(model2, 0.4)

        assert len(ensemble.models) == 2
        assert ensemble.weights == [0.6, 0.4]

    def test_get_confidence_empty_ensemble(self):
        """Test get_confidence with empty ensemble."""
        ensemble = EnsemblePredictor()
        assert ensemble.get_confidence() == 0.0

    def test_get_confidence_with_models(self):
        """Test get_confidence with models."""
        ensemble = EnsemblePredictor()

        model1 = Mock()
        model1.get_confidence.return_value = 0.8
        model2 = Mock()
        model2.get_confidence.return_value = 0.6

        ensemble.add_model(model1, 0.7)
        ensemble.add_model(model2, 0.3)

        expected = (0.8 * 0.7 + 0.6 * 0.3) / (0.7 + 0.3)
        assert ensemble.get_confidence() == expected

    def test_is_trained_empty(self):
        """Test is_trained with empty ensemble."""
        ensemble = EnsemblePredictor()
        assert ensemble.is_trained() is True  # Empty ensemble is considered trained

    def test_is_trained_with_models(self):
        """Test is_trained with models."""
        ensemble = EnsemblePredictor()

        model1 = Mock()
        model1.is_trained.return_value = True
        model2 = Mock()
        model2.is_trained.return_value = False

        ensemble.add_model(model1)
        ensemble.add_model(model2)

        assert ensemble.is_trained() is False  # One model not trained


class TestCacheablePredictor:
    """Test CacheablePredictor class."""

    def test_initialization(self):
        """Test CacheablePredictor initialization."""
        predictor = CacheablePredictor(cache_size=500)

        assert predictor.cache_size == 500
        assert predictor._prediction_cache == {}

    def test_get_cache_key(self):
        """Test cache key generation."""
        predictor = CacheablePredictor()
        key = predictor._get_cache_key("AAPL", "hash123")

        assert key == "AAPL_hash123_cacheable"

    def test_get_data_hash(self):
        """Test data hash generation."""
        predictor = CacheablePredictor()

        # Empty data
        assert predictor._get_data_hash(pd.DataFrame()) == "empty"
        assert predictor._get_data_hash(None) == "empty"

        # Valid data
        data = pd.DataFrame({"close": [100, 101, 102]})
        hash_value = predictor._get_data_hash(data)
        assert isinstance(hash_value, str)
        assert hash_value != "empty"

    def test_cache_prediction(self):
        """Test caching predictions."""
        predictor = CacheablePredictor(cache_size=2)
        data = pd.DataFrame({"close": [100]})

        result = PredictionResult(
            prediction=75.0, confidence=0.8, timestamp=datetime.now(), metadata={}
        )

        predictor.cache_prediction("AAPL", data, result)

        assert len(predictor._prediction_cache) == 1

        # Test cache overflow
        data2 = pd.DataFrame({"close": [101]})
        data3 = pd.DataFrame({"close": [102]})

        predictor.cache_prediction("GOOGL", data2, result)
        assert len(predictor._prediction_cache) == 2

        predictor.cache_prediction("MSFT", data3, result)
        assert len(predictor._prediction_cache) == 2  # Should evict oldest

    def test_get_cached_prediction(self):
        """Test retrieving cached predictions."""
        predictor = CacheablePredictor()
        data = pd.DataFrame({"close": [100]})

        result = PredictionResult(
            prediction=75.0, confidence=0.8, timestamp=datetime.now(), metadata={}
        )

        # No cache hit
        assert predictor.get_cached_prediction("AAPL", data) is None

        # Cache and retrieve
        predictor.cache_prediction("AAPL", data, result)
        cached_result = predictor.get_cached_prediction("AAPL", data)

        assert cached_result is not None
        assert cached_result.prediction == 75.0


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
    )


class TestPredictorInterface:
    """Test PredictorInterface abstract class."""

    def test_cannot_instantiate_interface(self):
        """Test that PredictorInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PredictorInterface()

    def test_concrete_implementation_must_implement_methods(self):
        """Test that concrete implementations must implement abstract methods."""

        class IncompletePredictor(PredictorInterface):
            pass

        with pytest.raises(TypeError):
            IncompletePredictor()

    def test_complete_implementation(self):
        """Test a complete implementation of PredictorInterface."""

        class CompletePredictor(PredictorInterface):
            def predict(self, symbol: str, data=None):
                return PredictionResult(
                    prediction=50.0,
                    confidence=0.5,
                    timestamp=datetime.now(),
                    metadata={},
                )

            def get_confidence(self):
                return 0.5

            def is_trained(self):
                return True

        predictor = CompletePredictor()
        assert predictor.is_trained() is True
        assert predictor.get_confidence() == 0.5

        result = predictor.predict("TEST")
        assert isinstance(result, PredictionResult)
