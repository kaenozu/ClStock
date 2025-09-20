"""
テスト用のモックリアルタイムデータプロバイダー

このモジュールは、テストや開発環境でリアルタイムデータの
動作を模擬するためのモックプロバイダーを提供します。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import random
import pandas as pd

from models_new.base.interfaces import (
    RealTimeDataProvider,
    TickData,
    OrderBookData,
    IndexData,
    NewsData
)

logger = logging.getLogger(__name__)


class MockRealTimeProvider(RealTimeDataProvider):
    """テスト用のモックリアルタイムデータプロバイダー"""

    def __init__(self):
        self.is_connected_flag = False
        self.subscribed_symbols = set()
        self.subscribed_indices = set()
        self.subscribed_news = False

        # シミュレーション用データ
        self.symbol_prices = {}  # 各銘柄の現在価格
        self.symbol_base_prices = {
            "7203": 2500.0,  # トヨタ
            "6758": 12000.0,  # ソニー
            "8058": 3200.0,  # 三菱商事
            "4519": 4800.0,  # 中外製薬
            "6861": 8500.0   # キーエンス
        }

        # 指数データ
        self.index_values = {
            "NIKKEI": 28000.0,
            "TOPIX": 2000.0
        }

        # データ生成タスク
        self.data_generation_tasks = []
        self.tick_callbacks = []
        self.order_book_callbacks = []
        self.index_callbacks = []
        self.news_callbacks = []

        logger.info("MockRealTimeProvider initialized")

    async def connect(self) -> bool:
        """モック接続処理"""
        try:
            # 接続遅延をシミュレート
            await asyncio.sleep(0.1)

            self.is_connected_flag = True

            # 基本価格を初期化
            for symbol, base_price in self.symbol_base_prices.items():
                self.symbol_prices[symbol] = base_price

            logger.info("Mock connection established")
            return True

        except Exception as e:
            logger.error(f"Mock connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """モック切断処理"""
        self.is_connected_flag = False

        # データ生成タスクを停止
        for task in self.data_generation_tasks:
            task.cancel()

        self.data_generation_tasks.clear()
        logger.info("Mock connection closed")

    async def subscribe_ticks(self, symbols: List[str]) -> None:
        """ティックデータ購読のモック"""
        if not self.is_connected_flag:
            raise RuntimeError("Not connected")

        for symbol in symbols:
            self.subscribed_symbols.add(symbol)

            # 基本価格が設定されていない場合はランダム生成
            if symbol not in self.symbol_prices:
                self.symbol_prices[symbol] = random.uniform(1000, 10000)

            # ティックデータ生成タスクを開始
            task = asyncio.create_task(self._generate_tick_data(symbol))
            self.data_generation_tasks.append(task)

            logger.info(f"Mock subscribed to tick data for {symbol}")

    async def subscribe_order_book(self, symbols: List[str]) -> None:
        """板情報購読のモック"""
        if not self.is_connected_flag:
            raise RuntimeError("Not connected")

        for symbol in symbols:
            # 板情報生成タスクを開始
            task = asyncio.create_task(self._generate_order_book_data(symbol))
            self.data_generation_tasks.append(task)

            logger.info(f"Mock subscribed to order book for {symbol}")

    async def subscribe_indices(self, indices: List[str]) -> None:
        """指数データ購読のモック"""
        if not self.is_connected_flag:
            raise RuntimeError("Not connected")

        for index in indices:
            self.subscribed_indices.add(index)

            # 指数データ生成タスクを開始
            task = asyncio.create_task(self._generate_index_data(index))
            self.data_generation_tasks.append(task)

            logger.info(f"Mock subscribed to index data for {index}")

    async def subscribe_news(self, symbols: Optional[List[str]] = None) -> None:
        """ニュースデータ購読のモック"""
        if not self.is_connected_flag:
            raise RuntimeError("Not connected")

        self.subscribed_news = True

        # ニュースデータ生成タスクを開始
        task = asyncio.create_task(self._generate_news_data(symbols))
        self.data_generation_tasks.append(task)

        logger.info("Mock subscribed to news data")

    async def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """最新ティックデータの取得（モック）"""
        if symbol not in self.symbol_prices:
            return None

        return self._create_mock_tick_data(symbol)

    async def get_latest_order_book(self, symbol: str) -> Optional[OrderBookData]:
        """最新板情報の取得（モック）"""
        if symbol not in self.symbol_prices:
            return None

        return self._create_mock_order_book_data(symbol)

    async def get_market_status(self) -> Dict[str, Any]:
        """市場状況の取得（モック）"""
        return {
            "is_connected": self.is_connected_flag,
            "subscribed_symbols": list(self.subscribed_symbols),
            "subscribed_indices": list(self.subscribed_indices),
            "last_update": datetime.now().isoformat(),
            "data_source": "mock"
        }

    async def is_connected(self) -> bool:
        """接続状態の確認"""
        return self.is_connected_flag

    async def _generate_tick_data(self, symbol: str) -> None:
        """ティックデータ生成ループ"""
        while self.is_connected_flag and symbol in self.subscribed_symbols:
            try:
                # 1-5秒間隔でランダム生成
                await asyncio.sleep(random.uniform(1, 5))

                tick_data = self._create_mock_tick_data(symbol)

                # コールバック実行
                for callback in self.tick_callbacks:
                    try:
                        callback(tick_data)
                    except Exception as e:
                        logger.error(f"Error in tick callback: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generating tick data for {symbol}: {e}")

    async def _generate_order_book_data(self, symbol: str) -> None:
        """板情報データ生成ループ"""
        while self.is_connected_flag:
            try:
                # 2-8秒間隔
                await asyncio.sleep(random.uniform(2, 8))

                order_book_data = self._create_mock_order_book_data(symbol)

                # コールバック実行
                for callback in self.order_book_callbacks:
                    try:
                        callback(order_book_data)
                    except Exception as e:
                        logger.error(f"Error in order book callback: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generating order book data for {symbol}: {e}")

    async def _generate_index_data(self, index: str) -> None:
        """指数データ生成ループ"""
        while self.is_connected_flag and index in self.subscribed_indices:
            try:
                # 10-30秒間隔
                await asyncio.sleep(random.uniform(10, 30))

                index_data = self._create_mock_index_data(index)

                # コールバック実行
                for callback in self.index_callbacks:
                    try:
                        callback(index_data)
                    except Exception as e:
                        logger.error(f"Error in index callback: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generating index data for {index}: {e}")

    async def _generate_news_data(self, symbols: Optional[List[str]]) -> None:
        """ニュースデータ生成ループ"""
        news_templates = [
            "{company}が業績予想を上方修正",
            "{company}の新製品発表が好評",
            "{company}株主総会で新戦略発表",
            "市場全体の動向が{company}に影響",
            "{company}の四半期決算発表"
        ]

        company_names = {
            "7203": "トヨタ自動車",
            "6758": "ソニーグループ",
            "8058": "三菱商事",
            "4519": "中外製薬",
            "6861": "キーエンス"
        }

        while self.is_connected_flag and self.subscribed_news:
            try:
                # 60-300秒間隔
                await asyncio.sleep(random.uniform(60, 300))

                # ランダムな銘柄を選択
                available_symbols = symbols or list(company_names.keys())
                if available_symbols:
                    symbol = random.choice(available_symbols)
                    company_name = company_names.get(symbol, f"企業{symbol}")

                    news_data = NewsData(
                        id=f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        timestamp=datetime.now(),
                        title=random.choice(news_templates).format(company=company_name),
                        content=f"{company_name}に関する重要なニュースが発表されました。",
                        symbols=[symbol],
                        sentiment=random.choice(["positive", "negative", "neutral"]),
                        impact_score=random.uniform(0.3, 1.0)
                    )

                    # コールバック実行
                    for callback in self.news_callbacks:
                        try:
                            callback(news_data)
                        except Exception as e:
                            logger.error(f"Error in news callback: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generating news data: {e}")

    def _create_mock_tick_data(self, symbol: str) -> TickData:
        """モックティックデータを作成"""
        # 価格をランダムウォークで変動
        current_price = self.symbol_prices[symbol]
        price_change = random.uniform(-0.02, 0.02)  # ±2%の変動
        new_price = current_price * (1 + price_change)
        self.symbol_prices[symbol] = new_price

        # ティックデータを作成
        return TickData(
            symbol=symbol,
            timestamp=datetime.now(),
            price=round(new_price, 2),
            volume=random.randint(100, 10000),
            bid_price=round(new_price * 0.999, 2),
            ask_price=round(new_price * 1.001, 2),
            trade_type=random.choice(["buy", "sell", "unknown"])
        )

    def _create_mock_order_book_data(self, symbol: str) -> OrderBookData:
        """モック板情報データを作成"""
        current_price = self.symbol_prices[symbol]

        # 買い注文（現在価格より低い価格で降順）
        bids = []
        for i in range(5):
            price = round(current_price * (1 - (i + 1) * 0.001), 2)
            volume = random.randint(100, 5000)
            bids.append((price, volume))

        # 売り注文（現在価格より高い価格で昇順）
        asks = []
        for i in range(5):
            price = round(current_price * (1 + (i + 1) * 0.001), 2)
            volume = random.randint(100, 5000)
            asks.append((price, volume))

        return OrderBookData(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

    def _create_mock_index_data(self, index: str) -> IndexData:
        """モック指数データを作成"""
        current_value = self.index_values[index]

        # 指数をランダムウォークで変動
        change = random.uniform(-0.01, 0.01)  # ±1%の変動
        new_value = current_value * (1 + change)
        self.index_values[index] = new_value

        change_points = new_value - current_value
        change_percent = (change_points / current_value) * 100

        return IndexData(
            symbol=index,
            timestamp=datetime.now(),
            value=round(new_value, 2),
            change=round(change_points, 2),
            change_percent=round(change_percent, 3)
        )

    # コールバック登録メソッド
    def add_tick_callback(self, callback) -> None:
        """ティックデータコールバックを追加"""
        self.tick_callbacks.append(callback)

    def add_order_book_callback(self, callback) -> None:
        """板情報コールバックを追加"""
        self.order_book_callbacks.append(callback)

    def add_index_callback(self, callback) -> None:
        """指数データコールバックを追加"""
        self.index_callbacks.append(callback)

    def add_news_callback(self, callback) -> None:
        """ニュースデータコールバックを追加"""
        self.news_callbacks.append(callback)

    # DataProviderインターフェースの実装
    def get_stock_data(self, symbol: str, period: str = "1d") -> pd.DataFrame:
        """モック過去データ取得"""
        # シンプルなモックデータを生成
        dates = pd.date_range(start=datetime.now() - timedelta(days=30),
                             end=datetime.now(), freq='D')

        base_price = self.symbol_prices.get(symbol, 1000.0)

        data = []
        for date in dates:
            price = base_price * (1 + random.uniform(-0.05, 0.05))
            data.append({
                'Date': date,
                'Open': price * (1 + random.uniform(-0.02, 0.02)),
                'High': price * (1 + random.uniform(0.0, 0.03)),
                'Low': price * (1 + random.uniform(-0.03, 0.0)),
                'Close': price,
                'Volume': random.randint(100000, 1000000)
            })

        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        return df

    def get_technical_indicators(self, symbol: str) -> Dict[str, float]:
        """モック技術指標取得"""
        current_price = self.symbol_prices.get(symbol, 1000.0)

        return {
            "last_price": current_price,
            "sma_20": current_price * (1 + random.uniform(-0.05, 0.05)),
            "sma_50": current_price * (1 + random.uniform(-0.1, 0.1)),
            "rsi": random.uniform(20, 80),
            "volume": float(random.randint(100000, 1000000))
        }