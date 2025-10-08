"""Comprehensive tests for the cache manager functionality."""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

import pandas as pd
from models.core.interfaces import (
    DataProvider as OrderBookData,
)

# AdvancedCacheManager縺ｯ邨ｱ蜷医＆繧後∪縺励◆
from models.core.interfaces import (
    PredictionResult as TickData,
)
from models.monitoring.cache_manager import AdvancedCacheManager, RealTimeCacheManager


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1100, 1200],
        },
    )


@pytest.fixture
def sample_tick_data():
    """Create sample TickData for testing."""
    return TickData(
        symbol="AAPL",
        timestamp=datetime.now(),
        price=150.0,
        volume=1000,
        bid_price=149.9,
        ask_price=150.1,
        trade_type="buy",
    )


@pytest.fixture
def sample_order_book_data():
    """Create sample OrderBookData for testing."""
    return OrderBookData(
        symbol="AAPL",
        timestamp=datetime.now(),
        bids=[(149.9, 100), (149.8, 200)],
        asks=[(150.1, 150), (150.2, 250)],
    )


class TestAdvancedCacheManager:
    """Comprehensive tests for AdvancedCacheManager."""

    def test_initialization_defaults(self):
        """Test AdvancedCacheManager initialization with default values."""
        cache = AdvancedCacheManager()

        # Check default values
        assert cache.max_cache_size == 1000
        assert cache.ttl_hours == 24
        assert cache.cleanup_interval == 1800
        assert isinstance(cache.feature_cache, dict)
        assert isinstance(cache.prediction_cache, dict)
        assert cache.feature_cache == {}
        assert cache.prediction_cache == {}

        # Check stats initialization
        assert cache.cache_stats["hits"] == 0
        assert cache.cache_stats["misses"] == 0
        assert cache.cache_stats["feature_cache_size"] == 0
        assert cache.cache_stats["prediction_cache_size"] == 0

    def test_initialization_custom_values(self):
        """Test AdvancedCacheManager initialization with custom values."""
        cache = AdvancedCacheManager(
            max_cache_size=500,
            ttl_hours=12,
            cleanup_interval=900,
        )

        assert cache.max_cache_size == 500
        assert cache.ttl_hours == 12
        assert cache.cleanup_interval == 900

    def test_cache_get_set_generic(self, sample_dataframe):
        """Test generic get/set operations."""
        cache = AdvancedCacheManager()

        # Test setting a value
        test_value = "test_value"
        cache.set("test_key", test_value, ttl=3600)

        # Test getting the value
        result = cache.get("test_key")
        assert result == test_value

        # Test getting non-existent key
        result = cache.get("non_existent_key")
        assert result is None

    def test_cache_get_set_dataframe(self, sample_dataframe):
        """Test caching DataFrame objects."""
        cache = AdvancedCacheManager()

        # Test setting DataFrame
        cache.set("df_key", sample_dataframe, ttl=3600)

        # Test getting DataFrame
        result = cache.get("df_key")
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_cache_get_set_numeric(self):
        """Test caching numeric values."""
        cache = AdvancedCacheManager()

        # Test setting numeric value
        cache.set("numeric_key", 75.5, ttl=3600)

        # Test getting numeric value
        result = cache.get("numeric_key")
        assert result == 75.5
        assert isinstance(result, float)

    def test_cache_delete(self):
        """Test cache deletion."""
        cache = AdvancedCacheManager()

        # Set some values
        cache.set("key1", "value1", ttl=3600)
        cache.set("key2", "value2", ttl=3600)

        # Verify they exist
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

        # Delete one key
        cache.delete("key1")

        # Check that only the deleted key is gone
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_cache_clear(self):
        """Test clearing all cache."""
        cache = AdvancedCacheManager()

        # Set some values
        cache.set("key1", "value1", ttl=3600)
        cache.set("key2", "value2", ttl=3600)
        cache.set("key3", pd.DataFrame({"a": [1, 2]}), ttl=3600)

        # Verify cache is not empty
        assert len(cache.feature_cache) > 0 or len(cache.prediction_cache) > 0

        # Clear all cache
        cache.clear()

        # Verify cache is empty
        assert cache.feature_cache == {}
        assert cache.prediction_cache == {}
        assert cache.cache_stats["feature_cache_size"] == 0
        assert cache.cache_stats["prediction_cache_size"] == 0

    def test_cache_features_methods(self, sample_dataframe):
        """Test feature caching specific methods."""
        cache = AdvancedCacheManager()

        # Test caching features
        symbol = "AAPL"
        data_hash = "test_hash_123"
        cache.cache_features(symbol, data_hash, sample_dataframe)

        # Test retrieving cached features
        result = cache.get_cached_features(symbol, data_hash)
        assert result is not None
        pd.testing.assert_frame_equal(result, sample_dataframe)

        # Test retrieving non-existent features
        result = cache.get_cached_features("GOOGL", data_hash)
        assert result is None

    def test_cache_prediction_methods(self):
        """Test prediction caching specific methods."""
        cache = AdvancedCacheManager()

        # Test caching prediction
        symbol = "AAPL"
        features_hash = "feature_hash_123"
        prediction = 75.5
        cache.cache_prediction(symbol, features_hash, prediction)

        # Test retrieving cached prediction
        result = cache.get_cached_prediction(symbol, features_hash)
        assert result == prediction

        # Test retrieving non-existent prediction
        result = cache.get_cached_prediction("GOOGL", features_hash)
        assert result is None

    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        cache = AdvancedCacheManager()

        # Set a value with short TTL
        cache.set("expiring_key", "expiring_value", ttl=1)  # 1 second TTL

        # Value should be available immediately
        result = cache.get("expiring_key")
        assert result == "expiring_value"

        # Wait for expiration
        import time

        time.sleep(2)

        # Value should now be expired
        result = cache.get("expiring_key")
        assert result is None

    def test_cache_size_limit_enforcement(self, sample_dataframe):
        """Test cache size limit enforcement."""
        cache = AdvancedCacheManager(max_cache_size=2)

        # Add more items than cache limit
        for i in range(5):
            cache.cache_features(f"STOCK{i}", f"hash{i}", sample_dataframe)
            cache.cache_prediction(f"STOCK{i}", f"hash{i}", float(i))

        # Cache should be limited to max size
        assert len(cache.feature_cache) <= 2
        assert len(cache.prediction_cache) <= 2

    def test_cache_stats_tracking(self):
        """Test cache statistics tracking."""
        cache = AdvancedCacheManager()

        # Initial stats
        initial_stats = cache.get_cache_stats()
        assert initial_stats["hits"] == 0
        assert initial_stats["misses"] == 0
        assert initial_stats["hit_rate"] == 0

        # Perform some cache operations
        cache.get("non_existent")  # miss
        cache.set("test_key", "test_value")
        cache.get("test_key")  # hit

        # Check updated stats
        stats = cache.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["total_requests"] == 2

    def test_data_hash_generation(self, sample_dataframe):
        """Test data hash generation."""
        cache = AdvancedCacheManager()

        # Test with valid DataFrame
        hash1 = cache.generate_data_hash(sample_dataframe)
        assert isinstance(hash1, str)
        assert len(hash1) > 0

        # Same data should produce same hash
        hash2 = cache.generate_data_hash(sample_dataframe)
        assert hash1 == hash2

        # Different data should produce different hash
        different_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        hash3 = cache.generate_data_hash(different_df)
        assert hash1 != hash3

    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        cache = AdvancedCacheManager()

        # Test with empty cache
        usage = cache._estimate_memory_usage()
        assert usage == 0.0

        # Add some items
        cache.set("test_key1", "test_value1")
        cache.set("test_key2", pd.DataFrame({"a": [1, 2]}))

        # Test with populated cache
        usage = cache._estimate_memory_usage()
        assert isinstance(usage, float)
        assert usage >= 0.0

    def test_cache_serialization(self):
        """Test cache serialization and deserialization."""
        cache = AdvancedCacheManager()

        # Add some data
        cache.set("string_key", "test_string")
        cache.set("numeric_key", 42.5)

        # Test serialization
        serialized = cache._serialize_cache(cache.prediction_cache)
        assert isinstance(serialized, dict)

        # Test deserialization
        deserialized = cache._deserialize_cache(serialized)
        assert isinstance(deserialized, dict)

    def test_save_load_cache_to_disk(self):
        """Test saving and loading cache to/from disk."""
        cache = AdvancedCacheManager()

        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(
            mode="w+",
            delete=False,
            suffix=".json",
        ) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Override cache file path
            cache.cache_file = Path(temp_path)

            # Add some data
            cache.set("test_key", "test_value")
            cache.cache_stats["hits"] = 5

            # Save cache
            cache.save_cache_to_disk()

            # Verify file was created
            assert cache.cache_file.exists()

            # Create new cache instance and load
            new_cache = AdvancedCacheManager()
            new_cache.cache_file = Path(temp_path)
            success = new_cache.load_cache_from_disk()

            assert success is True
            # Note: DataFrame cache is not saved, so only prediction cache is loaded
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_cleanup_old_cache(self, sample_dataframe):
        """Test cleanup of old cache entries."""
        cache = AdvancedCacheManager()

        # Add some items
        cache.cache_features("STOCK1", "hash1", sample_dataframe)
        cache.cache_prediction("STOCK1", "hash1", 75.0)

        # Test cleanup doesn't remove valid entries
        cache.cleanup_old_cache()
        assert len(cache.feature_cache) == 1
        assert len(cache.prediction_cache) == 1

        # Test with size limit enforcement
        cache.max_cache_size = 1
        # Add more items to exceed limit
        cache.cache_features("STOCK2", "hash2", sample_dataframe)
        cache.cleanup_old_cache()
        # Should be limited to max size
        assert len(cache.feature_cache) <= 1


class TestRealTimeCacheManager:
    """Comprehensive tests for RealTimeCacheManager."""

    def test_initialization_defaults(self):
        """Test RealTimeCacheManager initialization with default values."""
        cache = RealTimeCacheManager()

        # Check inherited values
        assert cache.max_cache_size == 5000
        assert cache.ttl_hours == 1
        assert isinstance(cache.tick_cache, dict)
        assert isinstance(cache.order_book_cache, dict)
        assert isinstance(cache.index_cache, dict)
        assert isinstance(cache.news_cache, dict)

        # Check real-time specific values
        assert cache.max_tick_history == 1000
        assert cache.max_order_book_history == 100

        # Check stats initialization
        assert isinstance(cache.real_time_stats, dict)
        assert cache.real_time_stats["total_ticks_cached"] == 0
        assert cache.real_time_stats["total_order_books_cached"] == 0

    def test_cache_tick_data(self, sample_tick_data):
        """Test caching tick data."""
        cache = RealTimeCacheManager()

        # Test caching tick data
        cache.cache_tick_data(sample_tick_data)

        # Verify data was cached
        assert sample_tick_data.symbol in cache.tick_cache
        assert len(cache.tick_cache[sample_tick_data.symbol]) == 1
        assert cache.tick_cache[sample_tick_data.symbol][0]["data"] == sample_tick_data
        assert cache.real_time_stats["total_ticks_cached"] == 1

    def test_cache_order_book_data(self, sample_order_book_data):
        """Test caching order book data."""
        cache = RealTimeCacheManager()

        # Test caching order book data
        cache.cache_order_book_data(sample_order_book_data)

        # Verify data was cached
        assert sample_order_book_data.symbol in cache.order_book_cache
        assert len(cache.order_book_cache[sample_order_book_data.symbol]) == 1
        assert (
            cache.order_book_cache[sample_order_book_data.symbol][0]["data"]
            == sample_order_book_data
        )
        assert cache.real_time_stats["total_order_books_cached"] == 1

    def test_tick_history_retrieval(self, sample_tick_data):
        """Test retrieving tick history."""
        cache = RealTimeCacheManager()

        # Add multiple tick entries
        tick1 = sample_tick_data
        tick2 = TickData(
            symbol="AAPL",
            timestamp=datetime.now() + timedelta(minutes=1),
            price=151.0,
            volume=1100,
        )

        cache.cache_tick_data(tick1)
        cache.cache_tick_data(tick2)

        # Test retrieving all history
        history = cache.get_tick_history("AAPL")
        assert len(history) == 2
        assert history[0] == tick1
        assert history[1] == tick2

        # Test retrieving limited history
        limited_history = cache.get_tick_history("AAPL", limit=1)
        assert len(limited_history) == 1
        assert limited_history[0] == tick2  # Most recent

    def test_order_book_history_retrieval(self, sample_order_book_data):
        """Test retrieving order book history."""
        cache = RealTimeCacheManager()

        # Add multiple order book entries
        ob1 = sample_order_book_data
        ob2 = OrderBookData(
            symbol="AAPL",
            timestamp=datetime.now() + timedelta(minutes=1),
            bids=[(150.0, 150)],
            asks=[(150.2, 300)],
        )

        cache.cache_order_book_data(ob1)
        cache.cache_order_book_data(ob2)

        # Test retrieving all history
        history = cache.get_order_book_history("AAPL")
        assert len(history) == 2
        assert history[0] == ob1
        assert history[1] == ob2

    def test_market_metrics_calculation(self, sample_tick_data):
        """Test market metrics calculation."""
        cache = RealTimeCacheManager()

        # Add multiple tick entries for metrics calculation
        base_time = datetime.now()
        for i in range(10):
            tick = TickData(
                symbol="AAPL",
                timestamp=base_time + timedelta(seconds=i),
                price=150.0 + i * 0.1,  # Gradually increasing prices
                volume=1000 + i * 100,
            )
            cache.cache_tick_data(tick)

        # Calculate metrics
        metrics = cache.calculate_market_metrics("AAPL")

        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert "symbol" in metrics
        assert "current_price" in metrics
        assert "average_price" in metrics
        assert "price_volatility" in metrics
        assert "total_volume" in metrics
        assert "price_change" in metrics
        assert "price_change_percent" in metrics
        assert "tick_count" in metrics
        assert metrics["symbol"] == "AAPL"
        assert metrics["tick_count"] == 10

    def test_real_time_cache_stats(self):
        """Test real-time cache statistics."""
        cache = RealTimeCacheManager()

        # Get initial stats
        stats = cache.get_real_time_cache_stats()

        # Verify stats structure
        assert isinstance(stats, dict)
        assert "real_time_stats" in stats
        assert "cached_symbols" in stats
        assert "cache_sizes" in stats

        # Verify real-time stats
        rt_stats = stats["real_time_stats"]
        assert "total_ticks_cached" in rt_stats
        assert "total_order_books_cached" in rt_stats
        assert "cache_evictions" in rt_stats

    def test_real_time_cache_cleanup(self, sample_tick_data):
        """Test real-time cache cleanup."""
        cache = RealTimeCacheManager()

        # Add some data
        old_tick = TickData(
            symbol="AAPL",
            timestamp=datetime.now() - timedelta(hours=3),  # 3 hours old
            price=150.0,
            volume=1000,
        )
        cache.cache_tick_data(old_tick)

        recent_tick = TickData(
            symbol="AAPL",
            timestamp=datetime.now() - timedelta(minutes=30),  # 30 minutes old
            price=151.0,
            volume=1100,
        )
        cache.cache_tick_data(recent_tick)

        # Cleanup data older than 2 hours
        removed_count = cache.cleanup_real_time_cache(older_than_hours=2)

        # Should remove the old tick but keep the recent one
        assert removed_count == 1
        history = cache.get_tick_history("AAPL")
        assert len(history) == 1
        assert history[0].price == 151.0  # Recent tick should remain

    def test_cache_size_limits(self, sample_tick_data, sample_order_book_data):
        """Test cache size limits enforcement."""
        cache = RealTimeCacheManager()
        cache.max_tick_history = 3
        cache.max_order_book_history = 2

        # Add more ticks than limit
        for i in range(5):
            tick = TickData(
                symbol="AAPL",
                timestamp=datetime.now() + timedelta(seconds=i),
                price=150.0 + i,
                volume=1000 + i * 100,
            )
            cache.cache_tick_data(tick)

        # Add more order books than limit
        for i in range(4):
            ob = OrderBookData(
                symbol="AAPL",
                timestamp=datetime.now() + timedelta(seconds=i),
                bids=[(149.0 + i, 100)],
                asks=[(151.0 + i, 150)],
            )
            cache.cache_order_book_data(ob)

        # Check that limits are enforced
        tick_history = cache.get_tick_history("AAPL")
        ob_history = cache.get_order_book_history("AAPL")

        assert len(tick_history) == 3  # Limited to max_tick_history
        assert len(ob_history) == 2  # Limited to max_order_book_history

        # Check that most recent entries are kept
        assert tick_history[-1].price == 154.0  # Most recent
        assert ob_history[-1].bids[0][0] == 152.0  # Most recent

    def test_export_real_time_data(self, sample_tick_data):
        """Test exporting real-time data."""
        cache = RealTimeCacheManager()

        # Add some tick data
        cache.cache_tick_data(sample_tick_data)

        # Test exporting as list (default)
        exported_list = cache.export_real_time_data("AAPL", format="list")
        assert isinstance(exported_list, list)
        assert len(exported_list) == 1
        assert exported_list[0] == sample_tick_data

        # Test exporting as DataFrame
        exported_df = cache.export_real_time_data("AAPL", format="dataframe")
        assert isinstance(exported_df, pd.DataFrame)
        assert len(exported_df) == 1
        assert exported_df.iloc[0]["symbol"] == "AAPL"

        # Test exporting as JSON
        exported_json = cache.export_real_time_data("AAPL", format="json")
        assert isinstance(exported_json, list)
        assert len(exported_json) == 1
        assert "symbol" in exported_json[0]

    def test_shutdown_procedure(self):
        """Test cache manager shutdown procedure."""
        cache = RealTimeCacheManager()

        # Mock the shutdown event
        with patch.object(cache, "_shutdown_event") as mock_event:
            mock_event.is_set.return_value = False
            mock_event.wait.return_value = True

            # Test shutdown
            cache.shutdown()

            # Verify shutdown event was set
            mock_event.set.assert_called_once()
