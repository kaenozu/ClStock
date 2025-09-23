"""
リアルタイムデータプロバイダーファクトリーと統合管理

このモジュールは、依存関係注入パターンを使用して、
テスト可能で拡張可能なリアルタイムデータシステムを提供します。
"""

import logging
from typing import Dict, Optional, Protocol, Type, Any
from abc import ABC, abstractmethod

from config.settings import get_settings, RealTimeConfig
from models_refactored.monitoring.cache_manager import RealTimeCacheManager
from models_refactored.monitoring.performance_monitor import ModelPerformanceMonitor as PerformanceMonitor
from models_refactored.core.interfaces import DataProvider as RealTimeDataProvider
from data.real_time_provider import WebSocketRealTimeProvider, RealTimeDataQualityMonitor

logger = logging.getLogger(__name__)


class RealTimeProviderFactory(ABC):
    """リアルタイムデータプロバイダーファクトリーの基底クラス"""

    @abstractmethod
    def create_provider(self, config: RealTimeConfig) -> RealTimeDataProvider:
        """データプロバイダーを作成"""
        pass

    @abstractmethod
    def create_cache_manager(self, config: RealTimeConfig) -> RealTimeCacheManager:
        """キャッシュマネージャーを作成"""
        pass

    @abstractmethod
    def create_quality_monitor(self, config: RealTimeConfig) -> RealTimeDataQualityMonitor:
        """データ品質監視を作成"""
        pass


class DefaultRealTimeFactory(RealTimeProviderFactory):
    """デフォルトのリアルタイムデータファクトリー"""

    def create_provider(self, config: RealTimeConfig) -> RealTimeDataProvider:
        """WebSocketベースのデータプロバイダーを作成"""
        return WebSocketRealTimeProvider()

    def create_cache_manager(self, config: RealTimeConfig) -> RealTimeCacheManager:
        """リアルタイムキャッシュマネージャーを作成"""
        return RealTimeCacheManager(
            max_cache_size=config.max_tick_history_per_symbol * 10,  # 複数銘柄対応
            ttl_hours=config.cache_cleanup_interval_hours
        )

    def create_quality_monitor(self, config: RealTimeConfig) -> RealTimeDataQualityMonitor:
        """データ品質監視を作成"""
        return RealTimeDataQualityMonitor()


class MockRealTimeFactory(RealTimeProviderFactory):
    """テスト用のモックファクトリー"""

    def create_provider(self, config: RealTimeConfig) -> RealTimeDataProvider:
        """モックプロバイダーを作成"""
        from data.mock_real_time_provider import MockRealTimeProvider
        return MockRealTimeProvider()

    def create_cache_manager(self, config: RealTimeConfig) -> RealTimeCacheManager:
        """テスト用キャッシュマネージャーを作成"""
        return RealTimeCacheManager(max_cache_size=100, ttl_hours=1)

    def create_quality_monitor(self, config: RealTimeConfig) -> RealTimeDataQualityMonitor:
        """テスト用品質監視を作成"""
        return RealTimeDataQualityMonitor()


class RealTimeSystemManager:
    """リアルタイムデータシステムの統合管理クラス"""

    def __init__(self, factory: Optional[RealTimeProviderFactory] = None):
        self.settings = get_settings()
        self.config = self.settings.real_time
        self.factory = factory or DefaultRealTimeFactory()

        # コンポーネントの初期化
        self.provider = self.factory.create_provider(self.config)
        self.cache_manager = self.factory.create_cache_manager(self.config)
        self.quality_monitor = self.factory.create_quality_monitor(self.config)

        # パフォーマンス監視
        self.performance_monitor = PerformanceMonitor()

        # システム統計
        self.system_stats = {
            "start_time": None,
            "total_data_processed": 0,
            "connection_attempts": 0,
            "successful_connections": 0,
            "data_quality_alerts": 0,
            "last_health_check": None
        }

        # コールバックの設定
        self._setup_callbacks()

        logger.info("RealTimeSystemManager initialized")

    def _setup_callbacks(self) -> None:
        """データプロバイダーのコールバックを設定"""
        if hasattr(self.provider, 'add_tick_callback'):
            self.provider.add_tick_callback(self._on_tick_received)

        if hasattr(self.provider, 'add_order_book_callback'):
            self.provider.add_order_book_callback(self._on_order_book_received)

        if hasattr(self.provider, 'add_index_callback'):
            self.provider.add_index_callback(self._on_index_received)

        if hasattr(self.provider, 'add_news_callback'):
            self.provider.add_news_callback(self._on_news_received)

    async def start_system(self) -> bool:
        """リアルタイムデータシステムを開始"""
        try:
            logger.info("Starting real-time data system...")

            # データプロバイダーへの接続
            self.system_stats["connection_attempts"] += 1
            connection_success = await self.provider.connect()

            if connection_success:
                self.system_stats["successful_connections"] += 1
                self.system_stats["start_time"] = datetime.now()

                # デフォルトサブスクリプションの開始
                await self._start_default_subscriptions()

                # 監視タスクの開始
                if self.config.enable_performance_monitoring:
                    asyncio.create_task(self._performance_monitoring_loop())

                logger.info("Real-time data system started successfully")
                return True
            else:
                logger.error("Failed to start real-time data system")
                return False

        except Exception as e:
            logger.error(f"Error starting real-time data system: {e}")
            return False

    async def stop_system(self) -> None:
        """リアルタイムデータシステムを停止"""
        try:
            logger.info("Stopping real-time data system...")

            # データプロバイダーの切断
            await self.provider.disconnect()

            # キャッシュの保存
            if hasattr(self.cache_manager, 'save_cache_to_disk'):
                self.cache_manager.save_cache_to_disk()

            logger.info("Real-time data system stopped")

        except Exception as e:
            logger.error(f"Error stopping real-time data system: {e}")

    async def _start_default_subscriptions(self) -> None:
        """デフォルトサブスクリプションを開始"""
        # ティックデータサブスクリプション
        if self.config.default_tick_subscription:
            await self.provider.subscribe_ticks(self.config.default_tick_subscription)

        # 板情報サブスクリプション
        if self.config.default_order_book_subscription:
            await self.provider.subscribe_order_book(self.config.default_order_book_subscription)

        # 指数データサブスクリプション
        if self.config.default_index_subscription:
            await self.provider.subscribe_indices(self.config.default_index_subscription)

        # ニュースサブスクリプション
        if self.config.enable_news_subscription:
            await self.provider.subscribe_news()

    def _on_tick_received(self, tick_data) -> None:
        """ティックデータ受信時のコールバック"""
        try:
            # データ品質チェック
            if self.config.enable_data_quality_monitoring:
                is_valid = self.quality_monitor.validate_tick_data(tick_data)
                if not is_valid:
                    self.system_stats["data_quality_alerts"] += 1
                    return

            # キャッシュに保存
            if self.config.enable_real_time_caching:
                self.cache_manager.cache_tick_data(
                    tick_data,
                    max_history=self.config.max_tick_history_per_symbol
                )

            # 統計更新
            self.system_stats["total_data_processed"] += 1

            # パフォーマンス監視
            if self.config.enable_performance_monitoring:
                self.performance_monitor.record_data_point("tick_processing", 1)

            if self.config.enable_detailed_logging:
                logger.debug(f"Processed tick data for {tick_data.symbol}: {tick_data.price}")

        except Exception as e:
            logger.error(f"Error processing tick data: {e}")

    def _on_order_book_received(self, order_book_data) -> None:
        """板情報受信時のコールバック"""
        try:
            # データ品質チェック
            if self.config.enable_data_quality_monitoring:
                is_valid = self.quality_monitor.validate_order_book(order_book_data)
                if not is_valid:
                    self.system_stats["data_quality_alerts"] += 1
                    return

            # キャッシュに保存
            if self.config.enable_real_time_caching:
                self.cache_manager.cache_order_book_data(
                    order_book_data,
                    max_history=self.config.max_order_book_history_per_symbol
                )

            # 統計更新
            self.system_stats["total_data_processed"] += 1

            if self.config.enable_detailed_logging:
                logger.debug(f"Processed order book for {order_book_data.symbol}")

        except Exception as e:
            logger.error(f"Error processing order book data: {e}")

    def _on_index_received(self, index_data) -> None:
        """指数データ受信時のコールバック"""
        try:
            # キャッシュに保存
            cache_key = f"latest_index_{index_data.symbol}"
            self.cache_manager.set(cache_key, index_data, ttl=self.config.tick_cache_ttl_seconds)

            # 統計更新
            self.system_stats["total_data_processed"] += 1

            if self.config.enable_detailed_logging:
                logger.debug(f"Processed index data for {index_data.symbol}: {index_data.value}")

        except Exception as e:
            logger.error(f"Error processing index data: {e}")

    def _on_news_received(self, news_data) -> None:
        """ニュースデータ受信時のコールバック"""
        try:
            # 関連性フィルタリング
            if (news_data.impact_score and
                news_data.impact_score < self.config.news_relevance_threshold):
                return

            # キャッシュに保存
            cache_key = f"news_{news_data.id}"
            self.cache_manager.set(cache_key, news_data, ttl=3600)  # 1時間

            # 統計更新
            self.system_stats["total_data_processed"] += 1

            if self.config.enable_detailed_logging:
                logger.info(f"Processed news: {news_data.title}")

        except Exception as e:
            logger.error(f"Error processing news data: {e}")

    async def _performance_monitoring_loop(self) -> None:
        """パフォーマンス監視ループ"""
        while True:
            try:
                await asyncio.sleep(self.config.performance_logging_interval)

                # システムヘルスチェック
                health_status = await self.get_system_health()
                self.system_stats["last_health_check"] = datetime.now()

                # データ品質チェック
                quality_metrics = self.quality_monitor.get_quality_metrics()

                # アラートチェック
                if self.config.alert_on_data_quality_degradation:
                    await self._check_data_quality_alerts(quality_metrics)

                # 接続状態チェック
                if self.config.alert_on_connection_loss:
                    is_connected = await self.provider.is_connected()
                    if not is_connected:
                        logger.warning("Real-time data connection lost!")

                # パフォーマンスログ
                if self.config.enable_detailed_logging:
                    logger.info(f"System health: {health_status}")
                    logger.info(f"Data quality: {quality_metrics}")

            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")

    async def _check_data_quality_alerts(self, quality_metrics: Dict[str, float]) -> None:
        """データ品質アラートをチェック"""
        tick_quality = quality_metrics.get("tick_quality_rate", 1.0)
        order_book_quality = quality_metrics.get("order_book_quality_rate", 1.0)

        if (tick_quality < self.config.data_quality_alert_threshold or
            order_book_quality < self.config.data_quality_alert_threshold):

            self.system_stats["data_quality_alerts"] += 1
            logger.warning(f"Data quality degradation detected: "
                         f"tick_quality={tick_quality:.3f}, "
                         f"order_book_quality={order_book_quality:.3f}")

    async def get_system_health(self) -> Dict[str, Any]:
        """システムヘルス状態を取得"""
        return {
            "is_connected": await self.provider.is_connected(),
            "system_stats": self.system_stats,
            "cache_stats": self.cache_manager.get_real_time_cache_stats(),
            "quality_metrics": self.quality_monitor.get_quality_metrics(),
            "performance_metrics": self.performance_monitor.get_performance_metrics(),
            "config": {
                "data_source": self.config.data_source,
                "enable_monitoring": self.config.enable_performance_monitoring,
                "cache_enabled": self.config.enable_real_time_caching
            }
        }

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """指定銘柄の市場データを取得"""
        return self.cache_manager.calculate_market_metrics(symbol)

    async def subscribe_to_symbol(self, symbol: str,
                                 include_ticks: bool = True,
                                 include_order_book: bool = True) -> bool:
        """新しい銘柄をサブスクリプション"""
        try:
            if include_ticks:
                await self.provider.subscribe_ticks([symbol])

            if include_order_book:
                await self.provider.subscribe_order_book([symbol])

            logger.info(f"Successfully subscribed to {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False


# グローバルなシステムマネージャーインスタンス
_real_time_system_manager: Optional[RealTimeSystemManager] = None


def get_real_time_system_manager(factory: Optional[RealTimeProviderFactory] = None) -> RealTimeSystemManager:
    """リアルタイムシステムマネージャーのシングルトンインスタンスを取得"""
    global _real_time_system_manager

    if _real_time_system_manager is None:
        _real_time_system_manager = RealTimeSystemManager(factory)

    return _real_time_system_manager


def reset_real_time_system_manager() -> None:
    """テスト用：システムマネージャーをリセット"""
    global _real_time_system_manager
    _real_time_system_manager = None