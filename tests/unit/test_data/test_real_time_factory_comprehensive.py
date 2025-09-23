"""Comprehensive tests for the real-time factory pattern implementation."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from data.real_time_factory import (
    RealTimeProviderFactory, DefaultRealTimeFactory, MockRealTimeFactory,
    RealTimeSystemManager, get_real_time_system_manager, reset_real_time_system_manager
)
from models_refactored.core.interfaces import DataProvider as RealTimeDataProvider
from models_refactored.monitoring.cache_manager import RealTimeCacheManager
from data.real_time_provider import RealTimeDataQualityMonitor
from config.settings import RealTimeConfig


class TestRealTimeProviderFactory:
    """Tests for the abstract factory base class."""

    def test_abstract_methods(self):
        """Test that the abstract factory methods are properly defined."""
        # This test ensures that the abstract methods exist
        # We can't instantiate the abstract class, so we just verify method signatures
        assert hasattr(RealTimeProviderFactory, 'create_provider')
        assert hasattr(RealTimeProviderFactory, 'create_cache_manager')
        assert hasattr(RealTimeProviderFactory, 'create_quality_monitor')


class TestDefaultRealTimeFactory:
    """Tests for the default real-time factory implementation."""

    def test_initialization(self):
        """Test factory initialization."""
        factory = DefaultRealTimeFactory()
        assert isinstance(factory, DefaultRealTimeFactory)
        assert isinstance(factory, RealTimeProviderFactory)

    def test_create_provider(self):
        """Test creation of real-time provider."""
        factory = DefaultRealTimeFactory()
        config = RealTimeConfig()
        
        with patch('data.real_time_factory.WebSocketRealTimeProvider') as mock_provider_class:
            mock_provider_instance = Mock()
            mock_provider_class.return_value = mock_provider_instance
            
            provider = factory.create_provider(config)
            
            # Should return a RealTimeDataProvider instance
            assert provider is not None
            # Should call the WebSocketRealTimeProvider constructor
            mock_provider_class.assert_called_once()

    def test_create_cache_manager(self):
        """Test creation of cache manager."""
        factory = DefaultRealTimeFactory()
        config = RealTimeConfig()
        
        cache_manager = factory.create_cache_manager(config)
        
        assert isinstance(cache_manager, RealTimeCacheManager)
        # Check that cache manager is configured with values from config
        assert cache_manager.max_cache_size == config.max_tick_history_per_symbol * 10

    def test_create_quality_monitor(self):
        """Test creation of quality monitor."""
        factory = DefaultRealTimeFactory()
        config = RealTimeConfig()
        
        with patch('data.real_time_factory.RealTimeDataQualityMonitor') as mock_monitor_class:
            mock_monitor_instance = Mock()
            mock_monitor_class.return_value = mock_monitor_instance
            
            monitor = factory.create_quality_monitor(config)
            
            assert monitor is not None
            mock_monitor_class.assert_called_once()


class TestMockRealTimeFactory:
    """Tests for the mock real-time factory implementation."""

    def test_initialization(self):
        """Test factory initialization."""
        factory = MockRealTimeFactory()
        assert isinstance(factory, MockRealTimeFactory)
        assert isinstance(factory, RealTimeProviderFactory)

    def test_create_provider(self):
        """Test creation of mock provider."""
        factory = MockRealTimeFactory()
        config = RealTimeConfig()
        
        with patch('data.mock_real_time_provider.MockRealTimeProvider') as mock_provider_class:
            mock_provider_instance = Mock()
            mock_provider_class.return_value = mock_provider_instance
            
            provider = factory.create_provider(config)
            
            assert provider is not None
            mock_provider_class.assert_called_once()

    def test_create_cache_manager(self):
        """Test creation of mock cache manager."""
        factory = MockRealTimeFactory()
        config = RealTimeConfig()
        
        cache_manager = factory.create_cache_manager(config)
        
        assert isinstance(cache_manager, RealTimeCacheManager)
        # Should use test-specific values
        assert cache_manager.max_cache_size == 100
        assert cache_manager.ttl_hours == 1

    def test_create_quality_monitor(self):
        """Test creation of mock quality monitor."""
        factory = MockRealTimeFactory()
        config = RealTimeConfig()
        
        with patch('data.real_time_factory.RealTimeDataQualityMonitor') as mock_monitor_class:
            mock_monitor_instance = Mock()
            mock_monitor_class.return_value = mock_monitor_instance
            
            monitor = factory.create_quality_monitor(config)
            
            assert monitor is not None
            mock_monitor_class.assert_called_once()


class TestRealTimeSystemManager:
    """Tests for the real-time system manager."""

    def setup_method(self):
        """Reset the singleton before each test."""
        reset_real_time_system_manager()

    def test_initialization_with_default_factory(self):
        """Test system manager initialization with default factory."""
        with patch('data.real_time_factory.DefaultRealTimeFactory') as mock_factory_class:
            mock_factory = Mock()
            mock_provider = Mock(spec=RealTimeDataProvider)
            mock_cache_manager = Mock(spec=RealTimeCacheManager)
            mock_quality_monitor = Mock(spec=RealTimeDataQualityMonitor)
            
            mock_factory.create_provider.return_value = mock_provider
            mock_factory.create_cache_manager.return_value = mock_cache_manager
            mock_factory.create_quality_monitor.return_value = mock_quality_monitor
            
            mock_factory_class.return_value = mock_factory
            
            manager = RealTimeSystemManager()
            
            # Verify factory was used
            mock_factory_class.assert_called_once()
            mock_factory.create_provider.assert_called_once()
            mock_factory.create_cache_manager.assert_called_once()
            mock_factory.create_quality_monitor.assert_called_once()
            
            # Verify components were set
            assert manager.provider == mock_provider
            assert manager.cache_manager == mock_cache_manager
            assert manager.quality_monitor == mock_quality_monitor

    def test_initialization_with_custom_factory(self):
        """Test system manager initialization with custom factory."""
        mock_factory = Mock(spec=RealTimeProviderFactory)
        mock_provider = Mock(spec=RealTimeDataProvider)
        mock_cache_manager = Mock(spec=RealTimeCacheManager)
        mock_quality_monitor = Mock(spec=RealTimeDataQualityMonitor)
        
        mock_factory.create_provider.return_value = mock_provider
        mock_factory.create_cache_manager.return_value = mock_cache_manager
        mock_factory.create_quality_monitor.return_value = mock_quality_monitor
        
        manager = RealTimeSystemManager(factory=mock_factory)
        
        # Verify custom factory was used
        mock_factory.create_provider.assert_called_once()
        mock_factory.create_cache_manager.assert_called_once()
        mock_factory.create_quality_monitor.assert_called_once()
        
        # Verify components were set
        assert manager.provider == mock_provider
        assert manager.cache_manager == mock_cache_manager
        assert manager.quality_monitor == mock_quality_monitor

    def test_system_stats_initialization(self):
        """Test that system statistics are properly initialized."""
        with patch('data.real_time_factory.DefaultRealTimeFactory'):
            manager = RealTimeSystemManager()
            
            # Check that all expected stats are initialized
            expected_stats = [
                "start_time", "total_data_processed", "connection_attempts",
                "successful_connections", "data_quality_alerts", "last_health_check"
            ]
            
            for stat in expected_stats:
                assert stat in manager.system_stats

    def test_singleton_pattern(self):
        """Test that get_real_time_system_manager implements singleton pattern."""
        with patch('data.real_time_factory.DefaultRealTimeFactory'):
            manager1 = get_real_time_system_manager()
            manager2 = get_real_time_system_manager()
            
            assert manager1 is manager2

    def test_singleton_reset(self):
        """Test that singleton can be reset."""
        with patch('data.real_time_factory.DefaultRealTimeFactory'):
            manager1 = get_real_time_system_manager()
            reset_real_time_system_manager()
            manager2 = get_real_time_system_manager()
            
            # Should be different instances after reset
            assert manager1 is not manager2

    @pytest.mark.asyncio
    async def test_start_system_success(self):
        """Test successful system start."""
        mock_factory = Mock(spec=RealTimeProviderFactory)
        mock_provider = Mock(spec=RealTimeDataProvider)
        mock_provider.connect = AsyncMock(return_value=True)
        mock_factory.create_provider.return_value = mock_provider
        mock_factory.create_cache_manager.return_value = Mock()
        mock_factory.create_quality_monitor.return_value = Mock()
        
        manager = RealTimeSystemManager(factory=mock_factory)
        
        # Mock the default subscriptions method
        with patch.object(manager, '_start_default_subscriptions') as mock_subscriptions:
            mock_subscriptions.return_value = None
            
            result = await manager.start_system()
            
            assert result is True
            mock_provider.connect.assert_called_once()
            mock_subscriptions.assert_called_once()
            assert manager.system_stats["connection_attempts"] == 1
            assert manager.system_stats["successful_connections"] == 1

    @pytest.mark.asyncio
    async def test_start_system_failure(self):
        """Test system start failure."""
        mock_factory = Mock(spec=RealTimeProviderFactory)
        mock_provider = Mock(spec=RealTimeDataProvider)
        mock_provider.connect = AsyncMock(return_value=False)  # Connection fails
        mock_factory.create_provider.return_value = mock_provider
        mock_factory.create_cache_manager.return_value = Mock()
        mock_factory.create_quality_monitor.return_value = Mock()
        
        manager = RealTimeSystemManager(factory=mock_factory)
        
        result = await manager.start_system()
        
        assert result is False
        mock_provider.connect.assert_called_once()
        assert manager.system_stats["connection_attempts"] == 1
        assert manager.system_stats["successful_connections"] == 0

    @pytest.mark.asyncio
    async def test_start_system_exception(self):
        """Test system start with exception."""
        mock_factory = Mock(spec=RealTimeProviderFactory)
        mock_provider = Mock(spec=RealTimeDataProvider)
        mock_provider.connect = AsyncMock(side_effect=Exception("Connection error"))
        mock_factory.create_provider.return_value = mock_provider
        mock_factory.create_cache_manager.return_value = Mock()
        mock_factory.create_quality_monitor.return_value = Mock()
        
        manager = RealTimeSystemManager(factory=mock_factory)
        
        result = await manager.start_system()
        
        assert result is False
        mock_provider.connect.assert_called_once()
        assert manager.system_stats["connection_attempts"] == 1

    @pytest.mark.asyncio
    async def test_stop_system(self):
        """Test system stop."""
        mock_factory = Mock(spec=RealTimeProviderFactory)
        mock_provider = Mock(spec=RealTimeDataProvider)
        mock_provider.disconnect = AsyncMock()
        mock_factory.create_provider.return_value = mock_provider
        mock_cache_manager = Mock()
        mock_cache_manager.save_cache_to_disk = Mock()
        mock_factory.create_cache_manager.return_value = mock_cache_manager
        mock_factory.create_quality_monitor.return_value = Mock()
        
        manager = RealTimeSystemManager(factory=mock_factory)
        
        await manager.stop_system()
        
        mock_provider.disconnect.assert_called_once()
        mock_cache_manager.save_cache_to_disk.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_system_exception(self):
        """Test system stop with exception."""
        mock_factory = Mock(spec=RealTimeProviderFactory)
        mock_provider = Mock(spec=RealTimeDataProvider)
        mock_provider.disconnect = AsyncMock(side_effect=Exception("Disconnect error"))
        mock_factory.create_provider.return_value = mock_provider
        mock_factory.create_cache_manager.return_value = Mock()
        mock_factory.create_quality_monitor.return_value = Mock()
        
        manager = RealTimeSystemManager(factory=mock_factory)
        
        # Should not raise exception even if disconnect fails
        await manager.stop_system()
        
        mock_provider.disconnect.assert_called_once()

    def test_get_market_data(self):
        """Test getting market data."""
        mock_factory = Mock(spec=RealTimeProviderFactory)
        mock_factory.create_provider.return_value = Mock()
        mock_cache_manager = Mock()
        mock_cache_manager.calculate_market_metrics = Mock(return_value={"test": "data"})
        mock_factory.create_cache_manager.return_value = mock_cache_manager
        mock_factory.create_quality_monitor.return_value = Mock()
        
        manager = RealTimeSystemManager(factory=mock_factory)
        
        result = manager.get_market_data("AAPL")
        
        assert result == {"test": "data"}
        mock_cache_manager.calculate_market_metrics.assert_called_once_with("AAPL")

    @pytest.mark.asyncio
    async def test_subscribe_to_symbol(self):
        """Test subscribing to a symbol."""
        mock_factory = Mock(spec=RealTimeProviderFactory)
        mock_provider = Mock(spec=RealTimeDataProvider)
        mock_provider.subscribe_ticks = AsyncMock()
        mock_provider.subscribe_order_book = AsyncMock()
        mock_factory.create_provider.return_value = mock_provider
        mock_factory.create_cache_manager.return_value = Mock()
        mock_factory.create_quality_monitor.return_value = Mock()
        
        manager = RealTimeSystemManager(factory=mock_factory)
        
        result = await manager.subscribe_to_symbol("AAPL")
        
        assert result is True
        mock_provider.subscribe_ticks.assert_called_once_with(["AAPL"])
        mock_provider.subscribe_order_book.assert_called_once_with(["AAPL"])

    @pytest.mark.asyncio
    async def test_subscribe_to_symbol_exception(self):
        """Test subscribing to a symbol with exception."""
        mock_factory = Mock(spec=RealTimeProviderFactory)
        mock_provider = Mock(spec=RealTimeDataProvider)
        mock_provider.subscribe_ticks = AsyncMock(side_effect=Exception("Subscription error"))
        mock_factory.create_provider.return_value = mock_provider
        mock_factory.create_cache_manager.return_value = Mock()
        mock_factory.create_quality_monitor.return_value = Mock()
        
        manager = RealTimeSystemManager(factory=mock_factory)
        
        result = await manager.subscribe_to_symbol("AAPL")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_system_health(self):
        """Test getting system health information."""
        mock_factory = Mock(spec=RealTimeProviderFactory)
        mock_provider = Mock(spec=RealTimeDataProvider)
        mock_provider.is_connected = AsyncMock(return_value=True)
        mock_factory.create_provider.return_value = mock_provider
        mock_cache_manager = Mock()
        mock_cache_manager.get_real_time_cache_stats = Mock(return_value={"cache": "stats"})
        mock_factory.create_cache_manager.return_value = mock_cache_manager
        mock_quality_monitor = Mock()
        mock_quality_monitor.get_quality_metrics = Mock(return_value={"quality": "metrics"})
        mock_factory.create_quality_monitor.return_value = mock_quality_monitor
        
        manager = RealTimeSystemManager(factory=mock_factory)
        
        # Mock performance monitor
        with patch.object(manager, 'performance_monitor') as mock_perf_monitor:
            mock_perf_monitor.get_performance_metrics = Mock(return_value={"perf": "metrics"})
            
            health = await manager.get_system_health()
            
            # Verify all components contribute to health report
            assert "is_connected" in health
            assert "system_stats" in health
            assert "cache_stats" in health
            assert "quality_metrics" in health
            assert "performance_metrics" in health
            assert "config" in health
            
            assert health["is_connected"] is True


# AsyncMock for Python < 3.8 compatibility
class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)