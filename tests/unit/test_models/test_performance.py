from pathlib import Path
import sys

# Resolve the project root and add it to the system path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""Test performance optimization models."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime
from concurrent.futures import Future

# 菫ｮ豁｣縺輔ｌ縺溘う繝ｳ繝昴・繝・
Mock = MagicMock
from ClStock.models.performance import AdvancedCacheManager, ParallelStockPredictor, UltraHighPerformancePredictor
from models.core import PredictionResult
from models_refactored.ensemble.ensemble_predictor import (
    EnsemblePredictor as EnsembleStockPredictor,
)


@pytest.fixture
def mock_ensemble_predictor():
    """Create a mock ensemble predictor."""
    predictor = Mock(spec=EnsembleStockPredictor)
    predictor.predict.return_value = PredictionResult(
        prediction=75.0,
        confidence=0.8,
        timestamp=datetime.now(),
        metadata={"test": True},
    )
    return predictor


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    return pd.DataFrame(
        {
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1100, 1200],
        }
    )


class TestParallelStockPredictor:
    """Test ParallelStockPredictor class."""

    def test_initialization_default_jobs(self, mock_ensemble_predictor):
        """Test initialization with default number of jobs."""
        predictor = ParallelStockPredictor(mock_ensemble_predictor)

        assert predictor.model_type == "parallel"
        assert predictor.ensemble_predictor == mock_ensemble_predictor
        assert predictor.n_jobs > 0  # Should be number of CPUs

    def test_initialization_custom_jobs(self, mock_ensemble_predictor):
        """Test initialization with custom number of jobs."""
        predictor = ParallelStockPredictor(mock_ensemble_predictor, n_jobs=4)

        assert predictor.n_jobs == 4

    def test_safe_predict_score_success(self, mock_ensemble_predictor):
        """Test _safe_predict_score with successful prediction."""
        predictor = ParallelStockPredictor(mock_ensemble_predictor)

        score = predictor._safe_predict_score("AAPL")

        assert score == 75.0
        mock_ensemble_predictor.predict.assert_called_once_with("AAPL")

    def test_safe_predict_score_failure(self, mock_ensemble_predictor):
        """Test _safe_predict_score with prediction failure."""
        predictor = ParallelStockPredictor(mock_ensemble_predictor)
        mock_ensemble_predictor.predict.side_effect = Exception("Prediction failed")

        score = predictor._safe_predict_score("AAPL")

        assert score == 50.0  # Default fallback value

    def test_predict_multiple_stocks_parallel_cached(self, mock_ensemble_predictor):
        """Test parallel prediction with cached results."""
        predictor = ParallelStockPredictor(mock_ensemble_predictor)
        predictor.batch_cache = {"AAPL": 80.0, "GOOGL": 70.0}

        results = predictor.predict_multiple_stocks_parallel(["AAPL", "GOOGL"])

        assert results == {"AAPL": 80.0, "GOOGL": 70.0}
        # Should not call the predictor due to cache
        mock_ensemble_predictor.predict.assert_not_called()

    @patch("models.performance.ThreadPoolExecutor")
    def test_predict_multiple_stocks_parallel_uncached(
        self, mock_executor, mock_ensemble_predictor
    ):
        """Test parallel prediction with uncached results."""
        predictor = ParallelStockPredictor(mock_ensemble_predictor, n_jobs=2)

        # Mock the executor and futures
        mock_future1 = Mock()
        mock_future1.result.return_value = 75.0
        mock_future2 = Mock()
        mock_future2.result.return_value = 80.0

        mock_executor_instance = Mock()
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]
        mock_executor.return_value = mock_executor_instance

        # Mock as_completed
        with patch("models.performance.as_completed") as mock_as_completed:
            mock_as_completed.return_value = [mock_future1, mock_future2]

            results = predictor.predict_multiple_stocks_parallel(["AAPL", "GOOGL"])

            assert len(results) == 2
            assert "AAPL" in results
            assert "GOOGL" in results

    def test_clear_batch_cache(self, mock_ensemble_predictor):
        """Test clearing batch cache."""
        predictor = ParallelStockPredictor(mock_ensemble_predictor)
        predictor.batch_cache = {"AAPL": 80.0}

        predictor.clear_batch_cache()

        assert predictor.batch_cache == {}

    @patch("data.stock_data.StockDataProvider")
    def test_get_stock_data_safe_success(
        self, mock_provider_class, mock_ensemble_predictor, sample_stock_data
    ):
        """Test _get_stock_data_safe with successful data retrieval."""
        predictor = ParallelStockPredictor(mock_ensemble_predictor)

        mock_provider = Mock()
        mock_provider.get_stock_data.return_value = sample_stock_data
        mock_provider_class.return_value = mock_provider

        result = predictor._get_stock_data_safe("AAPL")

        assert not result.empty
        assert len(result) == 3

    @patch("data.stock_data.StockDataProvider")
    def test_get_stock_data_safe_failure(
        self, mock_provider_class, mock_ensemble_predictor
    ):
        """Test _get_stock_data_safe with data retrieval failure."""
        predictor = ParallelStockPredictor(mock_ensemble_predictor)

        mock_provider = Mock()
        mock_provider.get_stock_data.side_effect = Exception("Data fetch failed")
        mock_provider_class.return_value = mock_provider

        result = predictor._get_stock_data_safe("AAPL")

        assert result.empty

    def test_train(self, mock_ensemble_predictor, sample_stock_data):
        """Test training the parallel predictor."""
        predictor = ParallelStockPredictor(mock_ensemble_predictor)

        target = pd.Series([1, 2, 3])
        predictor.train(sample_stock_data, target)

        mock_ensemble_predictor.train.assert_called_once_with(sample_stock_data, target)
        assert predictor.is_trained() is True

    def test_predict_single(self, mock_ensemble_predictor, sample_stock_data):
        """Test single prediction through parallel predictor."""
        predictor = ParallelStockPredictor(mock_ensemble_predictor)
        predictor._is_trained = True

        result = predictor.predict("AAPL", sample_stock_data)

        assert isinstance(result, PredictionResult)
        assert result.prediction == 75.0
        assert result.metadata["model_type"] == "parallel"
        assert result.metadata["parallel_enabled"] is True


class TestAdvancedCacheManager:
    """Test AdvancedCacheManager class."""

    def test_initialization(self):
        """Test AdvancedCacheManager initialization."""
        cache = AdvancedCacheManager(max_size=500)

        assert cache.max_size == 500
        assert cache.feature_cache == {}
        assert cache.prediction_cache == {}
        assert cache.cache_stats["hits"] == 0
        assert cache.cache_stats["misses"] == 0

    def test_cache_features(self, sample_stock_data):
        """Test caching features."""
        cache = AdvancedCacheManager()

        cache.cache_features("AAPL", "hash123", sample_stock_data)

        assert len(cache.feature_cache) == 1
        assert cache.cache_stats["feature_cache_size"] == 1

    def test_get_cached_features_hit(self, sample_stock_data):
        """Test getting cached features - cache hit."""
        cache = AdvancedCacheManager()
        cache.cache_features("AAPL", "hash123", sample_stock_data)

        result = cache.get_cached_features("AAPL", "hash123")

        assert result is not None
        assert len(result) == len(sample_stock_data)
        assert cache.cache_stats["hits"] == 1

    def test_get_cached_features_miss(self):
        """Test getting cached features - cache miss."""
        cache = AdvancedCacheManager()

        result = cache.get_cached_features("AAPL", "hash123")

        assert result is None
        assert cache.cache_stats["misses"] == 1

    def test_cache_prediction(self):
        """Test caching predictions."""
        cache = AdvancedCacheManager()

        cache.cache_prediction("AAPL", "hash123", 75.5)

        assert len(cache.prediction_cache) == 1
        assert cache.cache_stats["prediction_cache_size"] == 1

    def test_get_cached_prediction_hit(self):
        """Test getting cached prediction - cache hit."""
        cache = AdvancedCacheManager()
        cache.cache_prediction("AAPL", "hash123", 75.5)

        result = cache.get_cached_prediction("AAPL", "hash123")

        assert result == 75.5
        assert cache.cache_stats["hits"] == 1

    def test_get_cached_prediction_miss(self):
        """Test getting cached prediction - cache miss."""
        cache = AdvancedCacheManager()

        result = cache.get_cached_prediction("AAPL", "hash123")

        assert result is None
        assert cache.cache_stats["misses"] == 1

    def test_cache_size_limit_features(self, sample_stock_data):
        """Test cache size limit for features."""
        cache = AdvancedCacheManager(max_size=2)

        # Add 3 features, should evict the oldest
        cache.cache_features("AAPL", "hash1", sample_stock_data)
        cache.cache_features("GOOGL", "hash2", sample_stock_data)
        cache.cache_features("MSFT", "hash3", sample_stock_data)

        assert len(cache.feature_cache) == 2
        assert "AAPL_hash1" not in cache.feature_cache

    def test_cache_size_limit_predictions(self):
        """Test cache size limit for predictions."""
        cache = AdvancedCacheManager(max_size=2)

        # Add 3 predictions, should evict the oldest
        cache.cache_prediction("AAPL", "hash1", 75.0)
        cache.cache_prediction("GOOGL", "hash2", 80.0)
        cache.cache_prediction("MSFT", "hash3", 85.0)

        assert len(cache.prediction_cache) == 2
        assert "AAPL_hash1" not in cache.prediction_cache

    def test_cleanup_old_cache(self, sample_stock_data):
        """Test cleanup_old_cache method."""
        cache = AdvancedCacheManager(max_size=5)

        # Fill cache beyond cleanup threshold
        for i in range(8):
            cache.cache_features(f"STOCK{i}", f"hash{i}", sample_stock_data)
            cache.cache_prediction(f"STOCK{i}", f"hash{i}", float(i))

        cache.cleanup_old_cache(max_size=3)

        assert len(cache.feature_cache) <= 3
        assert len(cache.prediction_cache) <= 3

    def test_clear_all_cache(self, sample_stock_data):
        """Test clear_all_cache method."""
        cache = AdvancedCacheManager()
        cache.cache_features("AAPL", "hash1", sample_stock_data)
        cache.cache_prediction("AAPL", "hash1", 75.0)

        cache.clear_all_cache()

        assert len(cache.feature_cache) == 0
        assert len(cache.prediction_cache) == 0

    def test_get_cache_stats(self):
        """Test get_cache_stats method."""
        cache = AdvancedCacheManager()

        # Generate some cache activity
        cache.get_cached_prediction("AAPL", "hash1")  # miss
        cache.cache_prediction("AAPL", "hash1", 75.0)
        cache.get_cached_prediction("AAPL", "hash1")  # hit

        stats = cache.get_cache_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["total_requests"] == 2

    def test_get_data_hash_empty(self):
        """Test get_data_hash with empty data."""
        cache = AdvancedCacheManager()

        assert cache.get_data_hash(pd.DataFrame()) == "empty"
        assert cache.get_data_hash(None) == "empty"

    def test_get_data_hash_valid_data(self, sample_stock_data):
        """Test get_data_hash with valid data."""
        cache = AdvancedCacheManager()

        hash_value = cache.get_data_hash(sample_stock_data)

        assert isinstance(hash_value, str)
        assert hash_value != "empty"

        # Same data should produce same hash
        hash_value2 = cache.get_data_hash(sample_stock_data)
        assert hash_value == hash_value2

    def test_calculate_memory_efficiency(self):
        """Test _calculate_memory_efficiency method."""
        cache = AdvancedCacheManager(max_size=10)

        # Empty cache
        efficiency = cache._calculate_memory_efficiency()
        assert efficiency == 0.0

        # Partially filled cache
        cache.cache_prediction("AAPL", "hash1", 75.0)
        cache.cache_prediction("GOOGL", "hash2", 80.0)

        efficiency = cache._calculate_memory_efficiency()
        assert 0 < efficiency <= 1.0


class TestUltraHighPerformancePredictor:
    """Test UltraHighPerformancePredictor class."""

    def test_initialization(self):
        """Test UltraHighPerformancePredictor initialization."""
        base_predictor = Mock()
        cache_manager = AdvancedCacheManager(max_size=100)

        predictor = UltraHighPerformancePredictor(base_predictor, cache_manager)

        assert predictor.model_type == "ultra_performance"
        assert predictor.base_predictor == base_predictor
        assert predictor.cache_manager == cache_manager

    def test_train(self, sample_stock_data):
        """Test training the ultra performance predictor."""
        base_predictor = Mock()
        cache_manager = AdvancedCacheManager()
        predictor = UltraHighPerformancePredictor(base_predictor, cache_manager)

        target = pd.Series([1, 2, 3])
        predictor.train(sample_stock_data, target)

        base_predictor.train.assert_called_once_with(sample_stock_data, target)
        assert predictor.is_trained() is True

    @patch("models.performance.ThreadPoolExecutor")
    def test_predict_cache_hit(self, mock_provider_class, sample_stock_data):
        """Test predict with cache hit."""
        base_predictor = Mock()
        cache_manager = AdvancedCacheManager()
        predictor = UltraHighPerformancePredictor(base_predictor, cache_manager)
        predictor._is_trained = True

        # Setup mock data provider
        mock_provider = Mock()
        mock_provider.get_stock_data.return_value = sample_stock_data
        mock_provider_class.return_value = mock_provider

        # Pre-populate cache
        data_hash = cache_manager.get_data_hash(sample_stock_data)
        cache_manager.cache_prediction("AAPL", data_hash, 85.0)

        result = predictor.predict("AAPL")

        assert isinstance(result, PredictionResult)
        assert result.prediction == 85.0
        assert result.metadata["cache_hit"] is True
        # Base predictor should not be called due to cache hit
        base_predictor.predict.assert_not_called()

    @patch("models.performance.StockDataProvider")
    def test_predict_cache_miss(self, mock_provider_class, sample_stock_data):
        """Test predict with cache miss."""
        base_predictor = Mock()
        base_predictor.predict.return_value = PredictionResult(
            prediction=75.0,
            confidence=0.8,
            timestamp=datetime.now(),
            metadata={"base_model": True},
        )

        cache_manager = AdvancedCacheManager()
        predictor = UltraHighPerformancePredictor(base_predictor, cache_manager)
        predictor._is_trained = True

        # Setup mock data provider
        mock_provider = Mock()
        mock_provider.get_stock_data.return_value = sample_stock_data
        mock_provider_class.return_value = mock_provider

        result = predictor.predict("AAPL")

        assert isinstance(result, PredictionResult)
        assert result.prediction == 75.0
        assert result.metadata["cache_hit"] is False
        assert result.metadata["ultra_performance"] is True

        # Should call base predictor
        base_predictor.predict.assert_called_once()

    def test_get_performance_stats(self):
        """Test get_performance_stats method."""
        base_predictor = Mock()
        base_predictor.model_type = "test_base"
        cache_manager = AdvancedCacheManager()
        predictor = UltraHighPerformancePredictor(base_predictor, cache_manager)

        stats = predictor.get_performance_stats()

        assert "cache_stats" in stats
        assert stats["model_type"] == "ultra_performance"
        assert stats["base_predictor_type"] == "test_base"
        assert "parallel_jobs" in stats

    def test_optimize_cache(self):
        """Test optimize_cache method."""
        base_predictor = Mock()
        cache_manager = Mock()
        predictor = UltraHighPerformancePredictor(base_predictor, cache_manager)

        predictor.optimize_cache()

        cache_manager.cleanup_old_cache.assert_called_once()

    @patch("models.performance.ThreadPoolExecutor")
    @patch("models.performance.as_completed")
    def test_predict_multiple_mixed_cache(self, mock_as_completed, mock_executor):
        """Test predict_multiple with mixed cache hits and misses."""
        base_predictor = Mock()
        cache_manager = AdvancedCacheManager()
        predictor = UltraHighPerformancePredictor(base_predictor, cache_manager)

        # Mock some methods to avoid complex setup
        with patch.object(predictor, "predict") as mock_predict:
            mock_predict.side_effect = [
                PredictionResult(
                    prediction=75.0,
                    confidence=0.8,
                    timestamp=datetime.now(),
                    metadata={"cache_hit": False},
                ),
                PredictionResult(
                    prediction=80.0,
                    confidence=0.9,
                    timestamp=datetime.now(),
                    metadata={"cache_hit": False},
                ),
            ]

            # Mock executor
            mock_future1 = Mock()
            mock_future1.result.return_value = mock_predict.side_effect[0]
            mock_future2 = Mock()
            mock_future2.result.return_value = mock_predict.side_effect[1]

            mock_executor_instance = Mock()
            mock_executor_instance.__enter__.return_value = mock_executor_instance
            mock_executor_instance.__exit__.return_value = None
            mock_executor.return_value = mock_executor_instance

            mock_as_completed.return_value = [mock_future1, mock_future2]

            results = predictor.predict_multiple(["AAPL", "GOOGL"])

            assert len(results) == 2
            assert all(
                isinstance(result, PredictionResult) for result in results.values()
            )
