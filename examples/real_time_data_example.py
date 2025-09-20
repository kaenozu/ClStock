"""
リアルタイムデータフィードシステム使用例

このサンプルコードは、リアルタイムデータプロバイダーの
基本的な使用方法を示します。
"""

import asyncio
import logging
from utils.logger_config import setup_logger
from datetime import datetime
from typing import List

from data.real_time_factory import (
    get_real_time_system_manager,
    MockRealTimeFactory,
    reset_real_time_system_manager
)
from models_new.base.interfaces import TickData, OrderBookData, IndexData, NewsData
from config.settings import get_settings

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = setup_logger(__name__)


class RealTimeDataMonitor:
    """リアルタイムデータモニタリングクラス"""

    def __init__(self):
        self.received_ticks = []
        self.received_order_books = []
        self.received_indices = []
        self.received_news = []

    def on_tick_received(self, tick: TickData) -> None:
        """ティックデータ受信時の処理"""
        self.received_ticks.append(tick)
        logger.info(f"📈 ティック: {tick.symbol} - 価格: ¥{tick.price:,.0f}, "
                   f"出来高: {tick.volume:,}, タイプ: {tick.trade_type}")

        # 価格アラート例
        if tick.symbol == "7203" and tick.price > 2600:
            logger.warning(f"🚨 {tick.symbol}の価格が基準値を超過: ¥{tick.price:,.0f}")

    def on_order_book_received(self, order_book: OrderBookData) -> None:
        """板情報受信時の処理"""
        self.received_order_books.append(order_book)

        best_bid = order_book.bids[0][0] if order_book.bids else 0
        best_ask = order_book.asks[0][0] if order_book.asks else 0
        spread = best_ask - best_bid if best_bid and best_ask else 0

        logger.info(f"📊 板情報: {order_book.symbol} - "
                   f"最良買い: ¥{best_bid:,.0f}, 最良売り: ¥{best_ask:,.0f}, "
                   f"スプレッド: ¥{spread:.0f}")

    def on_index_received(self, index: IndexData) -> None:
        """指数データ受信時の処理"""
        self.received_indices.append(index)

        change_sign = "📈" if index.change >= 0 else "📉"
        logger.info(f"{change_sign} 指数: {index.symbol} - "
                   f"値: {index.value:,.2f}, 変化: {index.change:+.2f} "
                   f"({index.change_percent:+.2f}%)")

    def on_news_received(self, news: NewsData) -> None:
        """ニュースデータ受信時の処理"""
        self.received_news.append(news)

        sentiment_emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}.get(news.sentiment, "❓")
        impact_level = "高" if news.impact_score and news.impact_score > 0.7 else "中" if news.impact_score and news.impact_score > 0.4 else "低"

        logger.info(f"📰 ニュース {sentiment_emoji}: {news.title}")
        logger.info(f"   関連銘柄: {', '.join(news.symbols)}, インパクト: {impact_level}")

    def get_statistics(self) -> dict:
        """受信統計を取得"""
        return {
            "ticks_received": len(self.received_ticks),
            "order_books_received": len(self.received_order_books),
            "indices_received": len(self.received_indices),
            "news_received": len(self.received_news),
            "unique_symbols": len(set(tick.symbol for tick in self.received_ticks))
        }


async def basic_usage_example():
    """基本的な使用例"""
    logger.info("=== 基本的な使用例 ===")

    # システムマネージャーを取得（モックファクトリーを使用）
    reset_real_time_system_manager()
    factory = MockRealTimeFactory()
    manager = get_real_time_system_manager(factory)

    try:
        # システム開始
        logger.info("リアルタイムデータシステムを開始中...")
        success = await manager.start_system()
        if not success:
            logger.error("システム開始に失敗しました")
            return

        # 監視対象銘柄をサブスクリプション
        target_symbols = ["7203", "6758", "8058"]  # トヨタ、ソニー、三菱商事

        for symbol in target_symbols:
            await manager.subscribe_to_symbol(
                symbol,
                include_ticks=True,
                include_order_book=True
            )
            logger.info(f"銘柄 {symbol} をサブスクリプションしました")

        # 指数データもサブスクリプション
        await manager.provider.subscribe_indices(["NIKKEI", "TOPIX"])
        await manager.provider.subscribe_news(target_symbols)

        # データ監視クラスを設定
        monitor = RealTimeDataMonitor()
        manager.provider.add_tick_callback(monitor.on_tick_received)
        manager.provider.add_order_book_callback(monitor.on_order_book_received)
        manager.provider.add_index_callback(monitor.on_index_received)
        manager.provider.add_news_callback(monitor.on_news_received)

        # 30秒間データを監視
        logger.info("30秒間データを監視します...")
        await asyncio.sleep(30)

        # 統計表示
        stats = monitor.get_statistics()
        logger.info("=== 受信統計 ===")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")

        # システムヘルス確認
        health = await manager.get_system_health()
        logger.info("=== システムヘルス ===")
        logger.info(f"接続状態: {health['is_connected']}")
        logger.info(f"処理データ総数: {health['system_stats']['total_data_processed']}")

    finally:
        # システム停止
        logger.info("システムを停止中...")
        await manager.stop_system()


async def market_analysis_example():
    """市場分析の使用例"""
    logger.info("=== 市場分析の使用例 ===")

    reset_real_time_system_manager()
    factory = MockRealTimeFactory()
    manager = get_real_time_system_manager(factory)

    try:
        await manager.start_system()

        # 複数銘柄を監視
        symbols = ["7203", "6758", "8058", "4519", "6861"]

        for symbol in symbols:
            await manager.subscribe_to_symbol(symbol, include_ticks=True)

        # データ収集期間
        logger.info("市場データを収集中...")
        await asyncio.sleep(20)

        # 各銘柄の市場メトリクスを分析
        logger.info("=== 市場分析結果 ===")
        for symbol in symbols:
            metrics = manager.get_market_data(symbol)

            if metrics:
                logger.info(f"\n📊 {symbol} 分析結果:")
                logger.info(f"  現在価格: ¥{metrics.get('current_price', 0):,.0f}")
                logger.info(f"  平均価格: ¥{metrics.get('average_price', 0):,.0f}")
                logger.info(f"  ボラティリティ: {metrics.get('price_volatility', 0):.2f}%")
                logger.info(f"  価格変化: {metrics.get('price_change_percent', 0):+.2f}%")
                logger.info(f"  総出来高: {metrics.get('total_volume', 0):,}")
                logger.info(f"  ティック数: {metrics.get('tick_count', 0)}")

    finally:
        await manager.stop_system()


async def alert_system_example():
    """アラートシステムの使用例"""
    logger.info("=== アラートシステムの使用例 ===")

    class AlertManager:
        def __init__(self):
            self.price_alerts = {
                "7203": {"upper": 2600, "lower": 2400},  # トヨタ
                "6758": {"upper": 12500, "lower": 11500}  # ソニー
            }
            self.volume_threshold = 5000

        def check_alerts(self, tick: TickData) -> None:
            symbol = tick.symbol

            # 価格アラート
            if symbol in self.price_alerts:
                thresholds = self.price_alerts[symbol]
                if tick.price > thresholds["upper"]:
                    logger.warning(f"🚨 価格上限アラート: {symbol} が ¥{tick.price:,.0f} "
                                 f"(上限: ¥{thresholds['upper']:,.0f})")
                elif tick.price < thresholds["lower"]:
                    logger.warning(f"🚨 価格下限アラート: {symbol} が ¥{tick.price:,.0f} "
                                 f"(下限: ¥{thresholds['lower']:,.0f})")

            # 出来高アラート
            if tick.volume > self.volume_threshold:
                logger.warning(f"📊 大口取引アラート: {symbol} 出来高 {tick.volume:,} "
                             f"(閾値: {self.volume_threshold:,})")

    reset_real_time_system_manager()
    factory = MockRealTimeFactory()
    manager = get_real_time_system_manager(factory)

    try:
        await manager.start_system()

        # アラートマネージャーを設定
        alert_manager = AlertManager()
        manager.provider.add_tick_callback(alert_manager.check_alerts)

        # 監視銘柄をサブスクリプション
        await manager.subscribe_to_symbol("7203", include_ticks=True)
        await manager.subscribe_to_symbol("6758", include_ticks=True)

        logger.info("アラートシステムを30秒間動作させます...")
        await asyncio.sleep(30)

    finally:
        await manager.stop_system()


async def main():
    """メイン実行関数"""
    logger.info("🚀 ClStock リアルタイムデータフィードシステム サンプル実行")

    try:
        # 基本的な使用例
        await basic_usage_example()
        await asyncio.sleep(2)

        # 市場分析例
        await market_analysis_example()
        await asyncio.sleep(2)

        # アラートシステム例
        await alert_system_example()

        logger.info("✅ 全ての例の実行が完了しました")

    except Exception as e:
        logger.error(f"❌ エラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    # 設定確認
    settings = get_settings()
    logger.info(f"設定確認 - データソース: {settings.real_time.data_source}")
    logger.info(f"設定確認 - 監視有効: {settings.real_time.enable_performance_monitoring}")

    # サンプル実行
    asyncio.run(main())