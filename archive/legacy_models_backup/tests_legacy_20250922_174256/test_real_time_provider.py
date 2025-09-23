"""
リアルタイムデータプロバイダーのテストスイート

このモジュールは、リアルタイムデータプロバイダーの
ユニットテストとインテグレーションテストを提供します。
"""

import asyncio
import pytest
import unittest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import List

from data.real_time_provider import (
    WebSocketRealTimeProvider,
    DataNormalizer,
    RealTimeDataQualityMonitor,
    ReconnectionManager
)
from data.mock_real_time_provider import MockRealTimeProvider
from data.real_time_factory import (
    RealTimeSystemManager,
    DefaultRealTimeFactory,
    MockRealTimeFactory,
    get_real_time_system_manager,
    reset_real_time_system_manager
)
from models_new.base.interfaces import TickData, OrderBookData, IndexData, NewsData
from models_new.monitoring.cache_manager import RealTimeCacheManager
from config.settings import RealTimeConfig


class TestDataNormalizer(unittest.TestCase):
    """データ正規化のテスト"""

    def setUp(self):
        self.normalizer = DataNormalizer()

    def test_normalize_tick_data_valid(self):
        """有効なティックデータの正規化テスト"""
        raw_data = {
            "symbol": "7203",
            "price": 2500.0,
            "volume": 1000,
            "timestamp": "2025-01-20T10:30:00Z",
            "bid": 2499.0,
            "ask": 2501.0,
            "side": "buy"
        }

        tick = self.normalizer.normalize_tick_data(raw_data)

        self.assertIsNotNone(tick)
        self.assertEqual(tick.symbol, "7203")
        self.assertEqual(tick.price, 2500.0)
        self.assertEqual(tick.volume, 1000)
        self.assertEqual(tick.bid_price, 2499.0)
        self.assertEqual(tick.ask_price, 2501.0)
        self.assertEqual(tick.trade_type, "buy")

    def test_normalize_tick_data_minimal(self):
        """最小限のティックデータの正規化テスト"""
        raw_data = {
            "symbol": "7203",
            "price": 2500.0,
            "volume": 1000
        }

        tick = self.normalizer.normalize_tick_data(raw_data)

        self.assertIsNotNone(tick)
        self.assertEqual(tick.symbol, "7203")
        self.assertEqual(tick.price, 2500.0)
        self.assertEqual(tick.volume, 1000)
        self.assertIsNone(tick.bid_price)
        self.assertIsNone(tick.ask_price)
        self.assertEqual(tick.trade_type, "unknown")

    def test_normalize_tick_data_invalid(self):
        """無効なティックデータの正規化テスト"""
        raw_data = {
            "symbol": "",  # 空のシンボル
            "price": "invalid",  # 無効な価格
            "volume": -1000  # 負の出来高
        }

        tick = self.normalizer.normalize_tick_data(raw_data)
        self.assertIsNone(tick)

    def test_normalize_order_book_valid(self):
        """有効な板情報の正規化テスト"""
        raw_data = {
            "symbol": "7203",
            "timestamp": "2025-01-20T10:30:00Z",
            "bids": [[2499.0, 1000], [2498.0, 2000]],
            "asks": [[2501.0, 1500], [2502.0, 2500]]
        }

        order_book = self.normalizer.normalize_order_book(raw_data)

        self.assertIsNotNone(order_book)
        self.assertEqual(order_book.symbol, "7203")
        self.assertEqual(len(order_book.bids), 2)
        self.assertEqual(len(order_book.asks), 2)
        self.assertEqual(order_book.bids[0], (2499.0, 1000))
        self.assertEqual(order_book.asks[0], (2501.0, 1500))

    def test_normalize_order_book_invalid(self):
        """無効な板情報の正規化テスト"""
        raw_data = {
            "symbol": "",
            "bids": "invalid",
            "asks": []
        }

        order_book = self.normalizer.normalize_order_book(raw_data)
        self.assertIsNone(order_book)

    def test_normalize_index_data_valid(self):
        """有効な指数データの正規化テスト"""
        raw_data = {
            "symbol": "NIKKEI",
            "value": 28000.0,
            "change": 100.0,
            "change_percent": 0.36,
            "timestamp": "2025-01-20T10:30:00Z"
        }

        index_data = self.normalizer.normalize_index_data(raw_data)

        self.assertIsNotNone(index_data)
        self.assertEqual(index_data.symbol, "NIKKEI")
        self.assertEqual(index_data.value, 28000.0)
        self.assertEqual(index_data.change, 100.0)
        self.assertEqual(index_data.change_percent, 0.36)

    def test_normalize_news_data_valid(self):
        """有効なニュースデータの正規化テスト"""
        raw_data = {
            "id": "news_001",
            "title": "トヨタが業績予想上方修正",
            "content": "トヨタ自動車が2025年3月期の業績予想を上方修正した。",
            "symbols": ["7203"],
            "timestamp": "2025-01-20T10:30:00Z",
            "sentiment": "positive",
            "impact_score": 0.8
        }

        news = self.normalizer.normalize_news_data(raw_data)

        self.assertIsNotNone(news)
        self.assertEqual(news.id, "news_001")
        self.assertEqual(news.title, "トヨタが業績予想上方修正")
        self.assertEqual(news.symbols, ["7203"])
        self.assertEqual(news.sentiment, "positive")
        self.assertEqual(news.impact_score, 0.8)


class TestRealTimeDataQualityMonitor(unittest.TestCase):
    """データ品質監視のテスト"""

    def setUp(self):
        self.monitor = RealTimeDataQualityMonitor()

    def test_validate_tick_data_valid(self):
        """有効なティックデータの検証テスト"""
        tick = TickData(
            symbol="7203",
            timestamp=datetime.now(),
            price=2500.0,
            volume=1000,
            bid_price=2499.0,
            ask_price=2501.0,
            trade_type="buy"
        )

        result = self.monitor.validate_tick_data(tick)
        self.assertTrue(result)

    def test_validate_tick_data_invalid_price(self):
        """無効な価格のティックデータ検証テスト"""
        tick = TickData(
            symbol="7203",
            timestamp=datetime.now(),
            price=-100.0,  # 負の価格
            volume=1000,
            bid_price=2499.0,
            ask_price=2501.0,
            trade_type="buy"
        )

        result = self.monitor.validate_tick_data(tick)
        self.assertFalse(result)

    def test_validate_tick_data_invalid_symbol(self):
        """無効なシンボルのティックデータ検証テスト"""
        tick = TickData(
            symbol="",  # 空のシンボル
            timestamp=datetime.now(),
            price=2500.0,
            volume=1000,
            bid_price=2499.0,
            ask_price=2501.0,
            trade_type="buy"
        )

        result = self.monitor.validate_tick_data(tick)
        self.assertFalse(result)

    def test_validate_order_book_valid(self):
        """有効な板情報の検証テスト"""
        order_book = OrderBookData(
            symbol="7203",
            timestamp=datetime.now(),
            bids=[(2499.0, 1000), (2498.0, 2000)],
            asks=[(2501.0, 1500), (2502.0, 2500)]
        )

        result = self.monitor.validate_order_book(order_book)
        self.assertTrue(result)

    def test_validate_order_book_invalid_spread(self):
        """無効なスプレッドの板情報検証テスト"""
        order_book = OrderBookData(
            symbol="7203",
            timestamp=datetime.now(),
            bids=[(2502.0, 1000)],  # 買い価格が売り価格より高い
            asks=[(2501.0, 1500)]
        )

        result = self.monitor.validate_order_book(order_book)
        self.assertFalse(result)

    def test_get_quality_metrics(self):
        """品質メトリクス取得テスト"""
        # 有効なデータを処理
        valid_tick = TickData("7203", datetime.now(), 2500.0, 1000)
        self.monitor.validate_tick_data(valid_tick)

        # 無効なデータを処理
        invalid_tick = TickData("", datetime.now(), -100.0, 1000)
        self.monitor.validate_tick_data(invalid_tick)

        metrics = self.monitor.get_quality_metrics()

        self.assertIn("tick_quality_rate", metrics)
        self.assertIn("total_ticks_received", metrics)
        self.assertEqual(metrics["total_ticks_received"], 2.0)
        self.assertEqual(metrics["tick_quality_rate"], 0.5)  # 1/2 = 50%


class TestReconnectionManager(unittest.TestCase):
    """再接続管理のテスト"""

    def test_initial_state(self):
        """初期状態のテスト"""
        manager = ReconnectionManager(max_retries=3, base_delay=1.0)

        self.assertTrue(manager.should_retry())
        self.assertEqual(manager.get_delay(), 1.0)
        self.assertEqual(manager.retry_count, 0)

    def test_retry_logic(self):
        """再試行ロジックのテスト"""
        manager = ReconnectionManager(max_retries=3, base_delay=1.0)

        # 1回目の試行
        manager.record_attempt()
        self.assertTrue(manager.should_retry())
        self.assertEqual(manager.get_delay(), 2.0)  # 指数バックオフ
        self.assertEqual(manager.retry_count, 1)

        # 2回目の試行
        manager.record_attempt()
        self.assertTrue(manager.should_retry())
        self.assertEqual(manager.get_delay(), 4.0)
        self.assertEqual(manager.retry_count, 2)

        # 3回目の試行
        manager.record_attempt()
        self.assertFalse(manager.should_retry())  # 最大試行回数に達した
        self.assertEqual(manager.retry_count, 3)

    def test_reset(self):
        """リセット機能のテスト"""
        manager = ReconnectionManager(max_retries=3, base_delay=1.0)

        # 複数回試行
        manager.record_attempt()
        manager.record_attempt()

        # リセット
        manager.reset()

        self.assertTrue(manager.should_retry())
        self.assertEqual(manager.get_delay(), 1.0)
        self.assertEqual(manager.retry_count, 0)


class TestRealTimeCacheManager(unittest.TestCase):
    """リアルタイムキャッシュマネージャーのテスト"""

    def setUp(self):
        self.cache_manager = RealTimeCacheManager(max_cache_size=100, ttl_hours=1)

    def test_cache_tick_data(self):
        """ティックデータキャッシュのテスト"""
        tick = TickData("7203", datetime.now(), 2500.0, 1000)

        self.cache_manager.cache_tick_data(tick)

        # 履歴取得
        history = self.cache_manager.get_tick_history("7203")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].symbol, "7203")
        self.assertEqual(history[0].price, 2500.0)

    def test_cache_order_book_data(self):
        """板情報キャッシュのテスト"""
        order_book = OrderBookData(
            "7203", datetime.now(),
            [(2499.0, 1000)], [(2501.0, 1500)]
        )

        self.cache_manager.cache_order_book_data(order_book)

        # 履歴取得
        history = self.cache_manager.get_order_book_history("7203")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].symbol, "7203")

    def test_calculate_market_metrics(self):
        """市場メトリクス計算のテスト"""
        # 複数のティックデータを追加
        for i in range(10):
            tick = TickData("7203", datetime.now(), 2500.0 + i, 1000 + i * 100)
            self.cache_manager.cache_tick_data(tick)

        metrics = self.cache_manager.calculate_market_metrics("7203")

        self.assertIn("symbol", metrics)
        self.assertIn("current_price", metrics)
        self.assertIn("average_price", metrics)
        self.assertIn("price_volatility", metrics)
        self.assertIn("total_volume", metrics)
        self.assertEqual(metrics["symbol"], "7203")
        self.assertEqual(metrics["tick_count"], 10)

    def test_cleanup_real_time_cache(self):
        """リアルタイムキャッシュクリーンアップのテスト"""
        # 古いデータを追加
        old_time = datetime.now() - timedelta(hours=25)
        old_tick = TickData("7203", old_time, 2500.0, 1000)
        self.cache_manager.tick_cache["7203"] = [{
            "data": old_tick,
            "timestamp": old_time,
            "cached_at": old_time
        }]

        # 新しいデータを追加
        new_tick = TickData("7203", datetime.now(), 2600.0, 1200)
        self.cache_manager.cache_tick_data(new_tick)

        # クリーンアップ実行
        self.cache_manager.cleanup_real_time_cache(older_than_hours=24)

        # 古いデータが削除され、新しいデータのみ残ることを確認
        history = self.cache_manager.get_tick_history("7203")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].price, 2600.0)


class TestMockRealTimeProvider(unittest.IsolatedAsyncioTestCase):
    """モックリアルタイムプロバイダーのテスト"""

    async def asyncSetUp(self):
        self.provider = MockRealTimeProvider()

    async def asyncTearDown(self):
        await self.provider.disconnect()

    async def test_connection(self):
        """接続テスト"""
        # 初期状態では未接続
        self.assertFalse(await self.provider.is_connected())

        # 接続
        result = await self.provider.connect()
        self.assertTrue(result)
        self.assertTrue(await self.provider.is_connected())

        # 切断
        await self.provider.disconnect()
        self.assertFalse(await self.provider.is_connected())

    async def test_subscription(self):
        """サブスクリプションテスト"""
        await self.provider.connect()

        # ティックデータサブスクリプション
        await self.provider.subscribe_ticks(["7203", "6758"])
        self.assertIn("7203", self.provider.subscribed_symbols)
        self.assertIn("6758", self.provider.subscribed_symbols)

        # 指数データサブスクリプション
        await self.provider.subscribe_indices(["NIKKEI", "TOPIX"])
        self.assertIn("NIKKEI", self.provider.subscribed_indices)
        self.assertIn("TOPIX", self.provider.subscribed_indices)

        # ニュースサブスクリプション
        await self.provider.subscribe_news(["7203"])
        self.assertTrue(self.provider.subscribed_news)

    async def test_latest_data_retrieval(self):
        """最新データ取得テスト"""
        await self.provider.connect()

        # ティックデータ取得
        tick = await self.provider.get_latest_tick("7203")
        self.assertIsNotNone(tick)
        self.assertEqual(tick.symbol, "7203")
        self.assertGreater(tick.price, 0)

        # 板情報取得
        order_book = await self.provider.get_latest_order_book("7203")
        self.assertIsNotNone(order_book)
        self.assertEqual(order_book.symbol, "7203")
        self.assertGreater(len(order_book.bids), 0)
        self.assertGreater(len(order_book.asks), 0)

    async def test_market_status(self):
        """市場状況取得テスト"""
        await self.provider.connect()
        await self.provider.subscribe_ticks(["7203"])

        status = await self.provider.get_market_status()

        self.assertIn("is_connected", status)
        self.assertIn("subscribed_symbols", status)
        self.assertIn("last_update", status)
        self.assertEqual(status["data_source"], "mock")
        self.assertTrue(status["is_connected"])

    async def test_callbacks(self):
        """コールバック機能テスト"""
        await self.provider.connect()

        # コールバック用のモック
        tick_callback = Mock()
        order_book_callback = Mock()

        self.provider.add_tick_callback(tick_callback)
        self.provider.add_order_book_callback(order_book_callback)

        # サブスクリプション開始
        await self.provider.subscribe_ticks(["7203"])
        await self.provider.subscribe_order_book(["7203"])

        # 少し待ってコールバックが呼ばれることを確認
        await asyncio.sleep(2)

        # コールバックが呼ばれたことを確認
        # 注意: モックプロバイダーはランダムなタイミングでデータを生成するため、
        # 正確なコール回数は予測できないが、少なくとも1回は呼ばれるはず
        self.assertTrue(tick_callback.called or order_book_callback.called)


class TestRealTimeSystemManager(unittest.IsolatedAsyncioTestCase):
    """リアルタイムシステムマネージャーのテスト"""

    async def asyncSetUp(self):
        reset_real_time_system_manager()
        self.factory = MockRealTimeFactory()
        self.manager = RealTimeSystemManager(self.factory)

    async def asyncTearDown(self):
        await self.manager.stop_system()
        reset_real_time_system_manager()

    async def test_system_initialization(self):
        """システム初期化テスト"""
        self.assertIsNotNone(self.manager.provider)
        self.assertIsNotNone(self.manager.cache_manager)
        self.assertIsNotNone(self.manager.quality_monitor)
        self.assertIsNotNone(self.manager.performance_monitor)

    async def test_system_start_stop(self):
        """システム開始・停止テスト"""
        # システム開始
        result = await self.manager.start_system()
        self.assertTrue(result)

        # 接続状態確認
        self.assertTrue(await self.manager.provider.is_connected())

        # システム停止
        await self.manager.stop_system()
        self.assertFalse(await self.manager.provider.is_connected())

    async def test_subscription_management(self):
        """サブスクリプション管理テスト"""
        await self.manager.start_system()

        # 新しい銘柄をサブスクリプション
        result = await self.manager.subscribe_to_symbol("7203")
        self.assertTrue(result)

    async def test_system_health(self):
        """システムヘルス取得テスト"""
        await self.manager.start_system()

        health = await self.manager.get_system_health()

        self.assertIn("is_connected", health)
        self.assertIn("system_stats", health)
        self.assertIn("cache_stats", health)
        self.assertIn("quality_metrics", health)
        self.assertIn("performance_metrics", health)
        self.assertIn("config", health)

    async def test_market_data_retrieval(self):
        """市場データ取得テスト"""
        await self.manager.start_system()

        # ティックデータを生成して処理
        tick = TickData("7203", datetime.now(), 2500.0, 1000)
        self.manager._on_tick_received(tick)

        # 市場データ取得
        market_data = self.manager.get_market_data("7203")

        self.assertIn("symbol", market_data)
        self.assertEqual(market_data["symbol"], "7203")

    def test_singleton_pattern(self):
        """シングルトンパターンテスト"""
        manager1 = get_real_time_system_manager()
        manager2 = get_real_time_system_manager()

        self.assertIs(manager1, manager2)


class TestIntegration(unittest.IsolatedAsyncioTestCase):
    """統合テストクラス"""

    async def asyncSetUp(self):
        reset_real_time_system_manager()

    async def asyncTearDown(self):
        reset_real_time_system_manager()

    async def test_end_to_end_data_flow(self):
        """エンドツーエンドデータフローテスト"""
        # モックファクトリーを使用してシステムを構築
        factory = MockRealTimeFactory()
        manager = RealTimeSystemManager(factory)

        try:
            # システム開始
            await manager.start_system()

            # サブスクリプション設定
            await manager.subscribe_to_symbol("7203",
                                            include_ticks=True,
                                            include_order_book=True)

            # データが流れるまで少し待機
            await asyncio.sleep(3)

            # システムヘルスをチェック
            health = await manager.get_system_health()
            self.assertTrue(health["is_connected"])

            # キャッシュに何らかのデータが入っていることを確認
            cache_stats = health["cache_stats"]
            self.assertIn("real_time_stats", cache_stats)

        finally:
            await manager.stop_system()


if __name__ == "__main__":
    # 非同期テストを実行するためのヘルパー
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "async":
        # 非同期テストのみを実行
        async def run_async_tests():
            import unittest
            loader = unittest.TestLoader()
            suite = unittest.TestSuite()

            # 非同期テストクラスを追加
            suite.addTests(loader.loadTestsFromTestCase(TestMockRealTimeProvider))
            suite.addTests(loader.loadTestsFromTestCase(TestRealTimeSystemManager))
            suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

            runner = unittest.TextTestRunner(verbosity=2)

            # 各テストケースを個別に実行
            for test_class in [TestMockRealTimeProvider, TestRealTimeSystemManager, TestIntegration]:
                print(f"\n=== Running {test_class.__name__} ===")
                test_suite = loader.loadTestsFromTestCase(test_class)
                for test in test_suite:
                    try:
                        await test.debug()
                        print(f"✓ {test._testMethodName}")
                    except Exception as e:
                        print(f"✗ {test._testMethodName}: {e}")

        asyncio.run(run_async_tests())
    else:
        # 同期テストを実行
        unittest.main(verbosity=2)