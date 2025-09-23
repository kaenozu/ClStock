"""Comprehensive tests for the real-time provider components."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
import json
import asyncio

from data.real_time_provider import (
    ReconnectionManager,
    DataNormalizer,
    RealTimeDataQualityMonitor,
    WebSocketRealTimeProvider,
)
from models_refactored.core.interfaces import (
    PredictionResult as TickData,
    DataProvider as OrderBookData,
    ModelConfiguration as IndexData,
    PerformanceMetrics as NewsData,
)
from models_refactored.monitoring.cache_manager import (
    RealTimeCacheManager as AdvancedCacheManager,
)


class TestDataNormalizer:
    """Tests for the DataNormalizer class."""

    def test_normalize_tick_data_success(self):
        """Test successful tick data normalization."""
        raw_data = {
            "symbol": "AAPL",
            "price": "150.50",
            "volume": "1000",
            "timestamp": "2023-01-01T10:00:00+00:00",
            "bid": "150.49",
            "ask": "150.51",
            "side": "buy",
        }

        normalized = DataNormalizer.normalize_tick_data(raw_data)

        assert isinstance(normalized, TickData)
        assert normalized.symbol == "AAPL"
        assert normalized.price == 150.50
        assert normalized.volume == 1000
        assert normalized.bid_price == 150.49
        assert normalized.ask_price == 150.51
        assert normalized.trade_type == "buy"

    def test_normalize_tick_data_minimal_fields(self):
        """Test tick data normalization with minimal fields."""
        raw_data = {"symbol": "AAPL", "price": "150.50", "volume": "1000"}

        normalized = DataNormalizer.normalize_tick_data(raw_data)

        assert isinstance(normalized, TickData)
        assert normalized.symbol == "AAPL"
        assert normalized.price == 150.50
        assert normalized.volume == 1000
        # Optional fields should be None
        assert normalized.bid_price is None
        assert normalized.ask_price is None
        assert normalized.trade_type == "unknown"

    def test_normalize_tick_data_invalid_data(self):
        """Test tick data normalization with invalid data."""
        raw_data = {
            "symbol": "AAPL",
            "price": "invalid_price",  # Invalid price
            "volume": "1000",
        }

        normalized = DataNormalizer.normalize_tick_data(raw_data)

        # Should return None for invalid data
        assert normalized is None

    def test_normalize_order_book_data_success(self):
        """Test successful order book data normalization."""
        raw_data = {
            "symbol": "AAPL",
            "timestamp": "2023-01-01T10:00:00+00:00",
            "bids": [[150.49, 100], [150.48, 200]],
            "asks": [[150.51, 150], [150.52, 250]],
        }

        normalized = DataNormalizer.normalize_order_book_data(raw_data)

        assert isinstance(normalized, OrderBookData)
        assert normalized.symbol == "AAPL"
        assert len(normalized.bids) == 2
        assert len(normalized.asks) == 2
        assert normalized.bids[0] == (150.49, 100)
        assert normalized.asks[0] == (150.51, 150)

    def test_normalize_order_book_data_alternative_fields(self):
        """Test order book data normalization with alternative field names."""
        raw_data = {
            "code": "AAPL",  # Alternative symbol field
            "time": "2023-01-01T10:00:00+00:00",  # Alternative timestamp field
            "buy": [[150.49, 100], [150.48, 200]],  # Alternative bids field
            "sell": [[150.51, 150], [150.52, 250]],  # Alternative asks field
        }

        normalized = DataNormalizer.normalize_order_book_data(raw_data)

        assert isinstance(normalized, OrderBookData)
        assert normalized.symbol == "AAPL"

    def test_normalize_index_data_success(self):
        """Test successful index data normalization."""
        raw_data = {
            "symbol": "NIKKEI",
            "value": "30000.50",
            "change": "150.25",
            "change_percent": "0.50",
            "timestamp": "2023-01-01T10:00:00+00:00",
        }

        normalized = DataNormalizer.normalize_index_data(raw_data)

        assert isinstance(normalized, IndexData)
        assert normalized.symbol == "NIKKEI"
        assert normalized.value == 30000.50
        assert normalized.change == 150.25
        assert normalized.change_percent == 0.50

    def test_normalize_news_data_success(self):
        """Test successful news data normalization."""
        raw_data = {
            "id": "news_123",
            "title": "Test News",
            "content": "Test news content",
            "symbols": ["AAPL", "GOOGL"],
            "timestamp": "2023-01-01T10:00:00+00:00",
            "sentiment": "positive",
            "impact_score": "0.85",
        }

        normalized = DataNormalizer.normalize_news_data(raw_data)

        assert isinstance(normalized, NewsData)
        assert normalized.id == "news_123"
        assert normalized.title == "Test News"
        assert normalized.content == "Test news content"
        assert normalized.symbols == ["AAPL", "GOOGL"]
        assert normalized.sentiment == "positive"
        assert normalized.impact_score == 0.85


class TestReconnectionManager:
    """Tests for the ReconnectionManager class."""

    def test_initialization(self):
        """Test ReconnectionManager initialization."""
        manager = ReconnectionManager(max_retries=3, base_delay=2.0)

        assert manager.max_retries == 3
        assert manager.base_delay == 2.0
        assert manager.retry_count == 0
        assert manager.last_attempt is None

    def test_should_retry_initial(self):
        """Test should_retry when no attempts have been made."""
        manager = ReconnectionManager(max_retries=3)

        assert manager.should_retry() is True

    def test_should_retry_exceeded(self):
        """Test should_retry when max retries are exceeded."""
        manager = ReconnectionManager(max_retries=2)
        manager.retry_count = 2

        assert manager.should_retry() is False

    def test_get_delay_exponential_backoff(self):
        """Test delay calculation with exponential backoff."""
        manager = ReconnectionManager(max_retries=5, base_delay=1.0)

        # First attempt
        delay = manager.get_delay()
        assert delay == 1.0  # 1 * (2^0) = 1

        # Record attempt and check next delay
        manager.record_attempt()
        delay = manager.get_delay()
        assert delay == 2.0  # 1 * (2^1) = 2

    def test_record_attempt(self):
        """Test recording a reconnection attempt."""
        manager = ReconnectionManager()
        initial_time = datetime.now()

        manager.record_attempt()

        assert manager.retry_count == 1
        assert manager.last_attempt is not None
        assert manager.last_attempt >= initial_time

    def test_reset(self):
        """Test resetting the reconnection manager."""
        manager = ReconnectionManager()
        manager.retry_count = 3
        manager.last_attempt = datetime.now()

        manager.reset()

        assert manager.retry_count == 0
        assert manager.last_attempt is None


class TestRealTimeDataQualityMonitor:
    """Tests for the RealTimeDataQualityMonitor class."""

    def test_initialization(self):
        """Test quality monitor initialization."""
        monitor = RealTimeDataQualityMonitor()

        # Check that metrics are initialized
        assert "total_ticks_received" in monitor.metrics
        assert "invalid_ticks" in monitor.metrics
        assert "total_order_books_received" in monitor.metrics
        assert "invalid_order_books" in monitor.metrics
        assert monitor.metrics["total_ticks_received"] == 0
        assert monitor.metrics["invalid_ticks"] == 0

    def test_validate_tick_data_valid(self):
        """Test validation of valid tick data."""
        monitor = RealTimeDataQualityMonitor()
        tick = TickData(
            symbol="AAPL", timestamp=datetime.now(), price=150.0, volume=1000
        )

        result = monitor.validate_tick_data(tick)

        assert result is True
        assert monitor.metrics["total_ticks_received"] == 1
        assert monitor.metrics["invalid_ticks"] == 0

    def test_validate_tick_data_invalid(self):
        """Test validation of invalid tick data."""
        monitor = RealTimeDataQualityMonitor()
        tick = TickData(
            symbol="",  # Empty symbol is invalid
            timestamp=datetime.now(),
            price=150.0,
            volume=1000,
        )

        result = monitor.validate_tick_data(tick)

        assert result is False
        assert monitor.metrics["total_ticks_received"] == 1
        assert monitor.metrics["invalid_ticks"] == 1

    def test_validate_order_book_data_valid(self):
        """Test validation of valid order book data."""
        monitor = RealTimeDataQualityMonitor()
        order_book = OrderBookData(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[(149.9, 100), (149.8, 200)],
            asks=[(150.1, 150), (150.2, 250)],
        )

        result = monitor.validate_order_book(order_book)

        assert result is True
        assert monitor.metrics["total_order_books_received"] == 1
        assert monitor.metrics["invalid_order_books"] == 0

    def test_validate_order_book_data_invalid(self):
        """Test validation of invalid order book data."""
        monitor = RealTimeDataQualityMonitor()
        order_book = OrderBookData(
            symbol="",  # Empty symbol is invalid
            timestamp=datetime.now(),
            bids=[],
            asks=[],
        )

        result = monitor.validate_order_book(order_book)

        assert result is False
        assert monitor.metrics["total_order_books_received"] == 1
        assert monitor.metrics["invalid_order_books"] == 1

    def test_get_quality_metrics(self):
        """Test getting quality metrics."""
        monitor = RealTimeDataQualityMonitor()

        # Add some data
        valid_tick = TickData("AAPL", datetime.now(), 150.0, 1000)
        invalid_tick = TickData("", datetime.now(), 150.0, 1000)
        valid_order_book = OrderBookData(
            "AAPL", datetime.now(), [(149.9, 100)], [(150.1, 150)]
        )
        invalid_order_book = OrderBookData("", datetime.now(), [], [])

        monitor.validate_tick_data(valid_tick)
        monitor.validate_tick_data(invalid_tick)
        monitor.validate_order_book(valid_order_book)
        monitor.validate_order_book(invalid_order_book)

        metrics = monitor.get_quality_metrics()

        # Check that all expected metrics are present
        expected_metrics = [
            "tick_quality_rate",
            "order_book_quality_rate",
            "total_ticks_received",
            "total_order_books_received",
            "data_gaps",
        ]

        for metric in expected_metrics:
            assert metric in metrics

        # Check calculated rates
        assert metrics["tick_quality_rate"] == 0.5  # 1 valid / 2 total
        assert metrics["order_book_quality_rate"] == 0.5  # 1 valid / 2 total


@pytest.mark.asyncio
class TestWebSocketRealTimeProvider:
    """Tests for the WebSocketRealTimeProvider class."""

    def test_initialization_default(self):
        """Test provider initialization with default dependencies."""
        with patch("data.real_time_provider.get_settings") as mock_settings:
            mock_settings.return_value = Mock()

            provider = WebSocketRealTimeProvider()

            # Check that default dependencies were created
            assert isinstance(provider.cache_manager, AdvancedCacheManager)
            assert isinstance(provider.quality_monitor, RealTimeDataQualityMonitor)
            assert isinstance(provider.data_normalizer, DataNormalizer)
            assert isinstance(provider.reconnection_manager, ReconnectionManager)

            # Check that collections are initialized
            assert isinstance(provider.tick_callbacks, list)
            assert isinstance(provider.order_book_callbacks, list)
            assert isinstance(provider.index_callbacks, list)
            assert isinstance(provider.news_callbacks, list)

    def test_initialization_with_dependencies(self):
        """Test provider initialization with injected dependencies."""
        mock_cache = Mock()
        mock_quality = Mock()
        mock_normalizer = Mock()
        mock_reconnection = Mock()

        provider = WebSocketRealTimeProvider(
            cache_manager=mock_cache,
            quality_monitor=mock_quality,
            data_normalizer=mock_normalizer,
            reconnection_manager=mock_reconnection,
        )

        # Check that injected dependencies were used
        assert provider.cache_manager is mock_cache
        assert provider.quality_monitor is mock_quality
        assert provider.data_normalizer is mock_normalizer
        assert provider.reconnection_manager is mock_reconnection

    async def test_connect_success(self):
        """Test successful connection to WebSocket."""
        with patch("data.real_time_provider.get_settings") as mock_settings, patch(
            "data.real_time_provider.websockets.connect"
        ) as mock_connect:

            mock_settings.return_value = Mock()
            mock_websocket = Mock()
            mock_connect.return_value = mock_websocket

            provider = WebSocketRealTimeProvider()
            result = await provider.connect()

            assert result is True
            assert provider.websocket is mock_websocket
            assert provider.is_running is True
            mock_connect.assert_called_once()

    async def test_connect_failure(self):
        """Test failed connection to WebSocket."""
        with patch("data.real_time_provider.get_settings") as mock_settings, patch(
            "data.real_time_provider.websockets.connect"
        ) as mock_connect:

            mock_settings.return_value = Mock()
            mock_connect.side_effect = Exception("Connection failed")

            provider = WebSocketRealTimeProvider()
            result = await provider.connect()

            assert result is False
            assert provider.websocket is None
            assert provider.is_running is False

    async def test_disconnect(self):
        """Test disconnecting from WebSocket."""
        with patch("data.real_time_provider.get_settings") as mock_settings:
            mock_settings.return_value = Mock()

            provider = WebSocketRealTimeProvider()
            mock_websocket = Mock()
            mock_websocket.close = AsyncMock()
            provider.websocket = mock_websocket
            provider.is_running = True

            await provider.disconnect()

            assert provider.is_running is False
            mock_websocket.close.assert_called_once()

    async def test_is_connected(self):
        """Test checking connection status."""
        with patch("data.real_time_provider.get_settings") as mock_settings:
            mock_settings.return_value = Mock()

            provider = WebSocketRealTimeProvider()

            # Test when not connected
            assert await provider.is_connected() is False

            # Test when connected
            mock_websocket = Mock()
            mock_websocket.closed = False
            provider.websocket = mock_websocket

            assert await provider.is_connected() is True

    async def test_subscribe_ticks(self):
        """Test subscribing to tick data."""
        with patch("data.real_time_provider.get_settings") as mock_settings:
            mock_settings.return_value = Mock()

            provider = WebSocketRealTimeProvider()
            mock_websocket = Mock()
            mock_websocket.send = AsyncMock()
            provider.websocket = mock_websocket

            symbols = ["AAPL", "GOOGL"]
            await provider.subscribe_ticks(symbols)

            # Should send subscription messages for each symbol
            assert mock_websocket.send.call_count == 2
            assert "AAPL" in provider.subscribed_symbols
            assert "GOOGL" in provider.subscribed_symbols

    async def test_subscribe_ticks_not_connected(self):
        """Test subscribing to tick data when not connected."""
        with patch("data.real_time_provider.get_settings") as mock_settings:
            mock_settings.return_value = Mock()

            provider = WebSocketRealTimeProvider()

            # Should raise an exception when not connected
            with pytest.raises(RuntimeError):
                await provider.subscribe_ticks(["AAPL"])

    async def test_get_latest_tick_with_cache(self):
        """Test getting latest tick with cache hit."""
        with patch("data.real_time_provider.get_settings") as mock_settings:
            mock_settings.return_value = Mock()

            provider = WebSocketRealTimeProvider()
            mock_cache = Mock()
            cached_tick = TickData("AAPL", datetime.now(), 150.0, 1000)
            mock_cache.get.return_value = cached_tick
            provider.cache_manager = mock_cache

            result = await provider.get_latest_tick("AAPL")

            assert result is cached_tick
            mock_cache.get.assert_called_once_with("latest_tick_AAPL")

    async def test_get_latest_tick_without_cache(self):
        """Test getting latest tick without cache hit."""
        with patch("data.real_time_provider.get_settings") as mock_settings:
            mock_settings.return_value = Mock()

            provider = WebSocketRealTimeProvider()
            mock_cache = Mock()
            mock_cache.get.return_value = None  # No cache hit
            provider.cache_manager = mock_cache

            # Set a tick in memory
            tick = TickData("AAPL", datetime.now(), 150.0, 1000)
            provider.latest_ticks["AAPL"] = tick

            result = await provider.get_latest_tick("AAPL")

            assert result is tick

    async def test_get_market_status(self):
        """Test getting market status."""
        with patch("data.real_time_provider.get_settings") as mock_settings:
            mock_settings.return_value = Mock()

            provider = WebSocketRealTimeProvider()
            mock_websocket = Mock()
            mock_websocket.closed = False
            provider.websocket = mock_websocket

            # Add some subscriptions
            provider.subscribed_symbols.add("AAPL")
            provider.subscribed_indices.add("NIKKEI")

            # Mock quality metrics
            mock_quality = Mock()
            mock_quality.get_quality_metrics.return_value = {"test": "metrics"}
            provider.quality_monitor = mock_quality

            status = await provider.get_market_status()

            # Check that all expected fields are present
            expected_fields = [
                "is_connected",
                "subscribed_symbols",
                "subscribed_indices",
                "quality_metrics",
                "last_update",
            ]

            for field in expected_fields:
                assert field in status

            assert status["is_connected"] is True
            assert status["subscribed_symbols"] == ["AAPL"]
            assert status["subscribed_indices"] == ["NIKKEI"]
            assert status["quality_metrics"] == {"test": "metrics"}

    def test_add_callbacks(self):
        """Test adding data callbacks."""
        with patch("data.real_time_provider.get_settings") as mock_settings:
            mock_settings.return_value = Mock()

            provider = WebSocketRealTimeProvider()
            callback = Mock()

            # Test adding different types of callbacks
            provider.add_tick_callback(callback)
            provider.add_order_book_callback(callback)
            provider.add_index_callback(callback)
            provider.add_news_callback(callback)

            assert callback in provider.tick_callbacks
            assert callback in provider.order_book_callbacks
            assert callback in provider.index_callbacks
            assert callback in provider.news_callbacks
