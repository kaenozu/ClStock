"""
リアルタイムデータプロバイダーの実装

このモジュールは、WebSocket接続によるリアルタイムデータ取得、
データ正規化、品質監視機能を提供します。
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import pandas as pd
import hashlib
from abc import ABC, abstractmethod

from models_new.base.interfaces import (
    RealTimeDataProvider,
    TickData,
    OrderBookData,
    IndexData,
    NewsData,
    MarketData,
    DataQualityMonitor
)
from models_new.monitoring.cache_manager import AdvancedCacheManager
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ReconnectionManager:
    """再接続管理クラス"""

    def __init__(self, max_retries: int = 5, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_count = 0
        self.last_attempt = None

    def should_retry(self) -> bool:
        """再試行すべきかどうかを判定"""
        return self.retry_count < self.max_retries

    def get_delay(self) -> float:
        """次の再試行までの遅延時間を計算（指数バックオフ）"""
        return self.base_delay * (2 ** self.retry_count)

    def record_attempt(self):
        """再試行の記録"""
        self.retry_count += 1
        self.last_attempt = datetime.now()

    def reset(self):
        """再試行カウンターをリセット"""
        self.retry_count = 0
        self.last_attempt = None


class DataNormalizer:
    """データ正規化クラス"""

    @staticmethod
    def normalize_tick_data(raw_data: Dict[str, Any]) -> Optional[TickData]:
        """生のティックデータを正規化"""
        try:
            # 共通的なフィールドマッピング
            symbol = raw_data.get('symbol', raw_data.get('code', ''))
            price = float(raw_data.get('price', raw_data.get('last', 0)))
            volume = int(raw_data.get('volume', raw_data.get('vol', 0)))

            # タイムスタンプの正規化
            timestamp_str = raw_data.get('timestamp', raw_data.get('time', ''))
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()

            # オプショナルフィールド
            bid_price = raw_data.get('bid')
            ask_price = raw_data.get('ask')
            if bid_price:
                bid_price = float(bid_price)
            if ask_price:
                ask_price = float(ask_price)

            trade_type = raw_data.get('side', 'unknown')

            return TickData(
                symbol=symbol,
                timestamp=timestamp,
                price=price,
                volume=volume,
                bid_price=bid_price,
                ask_price=ask_price,
                trade_type=trade_type
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Failed to normalize tick data: {e}")
            return None

    @staticmethod
    def normalize_order_book(raw_data: Dict[str, Any]) -> Optional[OrderBookData]:
        """生の板情報データを正規化"""
        try:
            symbol = raw_data.get('symbol', raw_data.get('code', ''))

            timestamp_str = raw_data.get('timestamp', raw_data.get('time', ''))
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()

            # 買い注文（bids）の正規化
            bids = []
            bid_data = raw_data.get('bids', raw_data.get('buy', []))
            for bid in bid_data:
                if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                    bids.append((float(bid[0]), int(bid[1])))

            # 売り注文（asks）の正規化
            asks = []
            ask_data = raw_data.get('asks', raw_data.get('sell', []))
            for ask in ask_data:
                if isinstance(ask, (list, tuple)) and len(ask) >= 2:
                    asks.append((float(ask[0]), int(ask[1])))

            return OrderBookData(
                symbol=symbol,
                timestamp=timestamp,
                bids=bids,
                asks=asks
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Failed to normalize order book data: {e}")
            return None

    @staticmethod
    def normalize_index_data(raw_data: Dict[str, Any]) -> Optional[IndexData]:
        """生の指数データを正規化"""
        try:
            symbol = raw_data.get('symbol', raw_data.get('index', ''))
            value = float(raw_data.get('value', raw_data.get('price', 0)))
            change = float(raw_data.get('change', 0))
            change_percent = float(raw_data.get('change_percent',
                                               raw_data.get('change_pct', 0)))

            timestamp_str = raw_data.get('timestamp', raw_data.get('time', ''))
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()

            return IndexData(
                symbol=symbol,
                timestamp=timestamp,
                value=value,
                change=change,
                change_percent=change_percent
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Failed to normalize index data: {e}")
            return None

    @staticmethod
    def normalize_news_data(raw_data: Dict[str, Any]) -> Optional[NewsData]:
        """生のニュースデータを正規化"""
        try:
            id_str = raw_data.get('id', raw_data.get('uuid', str(hash(str(raw_data)))))
            title = raw_data.get('title', raw_data.get('headline', ''))
            content = raw_data.get('content', raw_data.get('body', ''))

            # 関連銘柄の抽出
            symbols = raw_data.get('symbols', raw_data.get('codes', []))
            if isinstance(symbols, str):
                symbols = [symbols]

            timestamp_str = raw_data.get('timestamp', raw_data.get('published', ''))
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()

            sentiment = raw_data.get('sentiment')
            impact_score = raw_data.get('impact_score')
            if impact_score:
                impact_score = float(impact_score)

            return NewsData(
                id=id_str,
                timestamp=timestamp,
                title=title,
                content=content,
                symbols=symbols,
                sentiment=sentiment,
                impact_score=impact_score
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Failed to normalize news data: {e}")
            return None


class RealTimeDataQualityMonitor(DataQualityMonitor):
    """リアルタイムデータ品質監視実装"""

    def __init__(self):
        self.metrics = {
            'total_ticks_received': 0,
            'invalid_ticks': 0,
            'total_order_books_received': 0,
            'invalid_order_books': 0,
            'data_gaps': 0,
            'last_data_timestamp': None
        }
        self.price_history = {}  # 価格の履歴を保持してスパイク検出

    def validate_tick_data(self, tick: TickData) -> bool:
        """ティックデータの品質検証"""
        self.metrics['total_ticks_received'] += 1

        # 基本的な妥当性チェック
        if not tick.symbol or tick.price <= 0 or tick.volume < 0:
            self.metrics['invalid_ticks'] += 1
            return False

        # 価格スパイク検出
        if tick.symbol in self.price_history and self.price_history[tick.symbol]:
            last_price = self.price_history[tick.symbol][-1] if self.price_history[tick.symbol] else tick.price
            
            # ゼロ除算を防ぐためのチェック
            if last_price > 0:
                price_change_ratio = abs(tick.price - last_price) / last_price
                
                # 10%以上の急激な価格変動をスパイクとして検出
                if price_change_ratio > 0.1:
                    logger.warning(f"Price spike detected for {tick.symbol}: "
                                 f"{last_price} -> {tick.price} ({price_change_ratio:.2%})")
            else:
                # last_priceが0以下の場合は警告を出力
                logger.warning(f"Invalid last_price ({last_price}) for {tick.symbol}, skipping spike detection")

        # 価格履歴を更新
        if tick.symbol not in self.price_history:
            self.price_history[tick.symbol] = []
        self.price_history[tick.symbol].append(tick.price)
        if len(self.price_history[tick.symbol]) > 100:
            self.price_history[tick.symbol] = self.price_history[tick.symbol][-100:]

        # タイムスタンプチェック
        now = datetime.now()
        if abs((tick.timestamp - now).total_seconds()) > 3600:  # 1時間以上ずれている
            logger.warning(f"Timestamp anomaly detected for {tick.symbol}: {tick.timestamp}")

        self.metrics['last_data_timestamp'] = tick.timestamp
        return True

    def validate_order_book(self, order_book: OrderBookData) -> bool:
        """板情報の品質検証"""
        self.metrics['total_order_books_received'] += 1

        # 基本的な妥当性チェック
        if not order_book.symbol or not order_book.bids or not order_book.asks:
            self.metrics['invalid_order_books'] += 1
            return False

        # 買い注文が価格順に並んでいるかチェック（降順）
        bid_prices = [bid[0] for bid in order_book.bids]
        if bid_prices != sorted(bid_prices, reverse=True):
            logger.warning(f"Bid prices not in correct order for {order_book.symbol}")

        # 売り注文が価格順に並んでいるかチェック（昇順）
        ask_prices = [ask[0] for ask in order_book.asks]
        if ask_prices != sorted(ask_prices):
            logger.warning(f"Ask prices not in correct order for {order_book.symbol}")

        # スプレッドチェック（改善版）
        if order_book.bids and order_book.asks:
            best_bid = order_book.bids[0][0]
            best_ask = order_book.asks[0][0]
            
            # 正常なスプレッドかチェック
            if best_bid >= best_ask:
                logger.warning(f"Invalid spread for {order_book.symbol}: "
                             f"bid={best_bid}, ask={best_ask}")
                return False
        else:
            # 片方が空の場合は警告を出すが無効ではない
            if not order_book.bids:
                logger.warning(f"No bids available for {order_book.symbol}")
            if not order_book.asks:
                logger.warning(f"No asks available for {order_book.symbol}")

        return True

    def get_quality_metrics(self) -> Dict[str, float]:
        """データ品質メトリクスを取得"""
        total_ticks = self.metrics['total_ticks_received']
        total_order_books = self.metrics['total_order_books_received']

        tick_quality_rate = 1.0
        order_book_quality_rate = 1.0

        if total_ticks > 0:
            tick_quality_rate = (total_ticks - self.metrics['invalid_ticks']) / total_ticks

        if total_order_books > 0:
            order_book_quality_rate = (total_order_books - self.metrics['invalid_order_books']) / total_order_books

        return {
            'tick_quality_rate': tick_quality_rate,
            'order_book_quality_rate': order_book_quality_rate,
            'total_ticks_received': float(total_ticks),
            'total_order_books_received': float(total_order_books),
            'data_gaps': float(self.metrics['data_gaps'])
        }


class WebSocketRealTimeProvider(RealTimeDataProvider):
    """WebSocketベースのリアルタイムデータプロバイダー"""

    def __init__(self, cache_manager=None, quality_monitor=None, data_normalizer=None, reconnection_manager=None):
        """
        WebSocketRealTimeProviderの初期化

        Args:
            cache_manager: キャッシュマネージャー（Noneの場合は新規作成）
            quality_monitor: データ品質監視（Noneの場合は新規作成）
            data_normalizer: データ正規化（Noneの場合は新規作成）
            reconnection_manager: 再接続マネージャー（Noneの場合は新規作成）
        """
        self.settings = get_settings()
        
        # 依存性注入または新規作成
        self.cache_manager = cache_manager or AdvancedCacheManager(max_cache_size=5000, ttl_hours=1)
        self.quality_monitor = quality_monitor or RealTimeDataQualityMonitor()
        self.data_normalizer = data_normalizer or DataNormalizer()
        self.reconnection_manager = reconnection_manager or ReconnectionManager()

        # WebSocket設定
        self.websocket = None
        self.is_running = False
        self.subscribed_symbols = set()
        self.subscribed_indices = set()

        # データコールバック
        self.tick_callbacks: List[Callable[[TickData], None]] = []
        self.order_book_callbacks: List[Callable[[OrderBookData], None]] = []
        self.index_callbacks: List[Callable[[IndexData], None]] = []
        self.news_callbacks: List[Callable[[NewsData], None]] = []

        # データストレージ
        self.latest_ticks = {}
        self.latest_order_books = {}
        self.latest_indices = {}

        logger.info("WebSocketRealTimeProvider initialized")

    async def connect(self) -> bool:
        """WebSocketデータソースに接続"""
        try:
            # 設定から接続先URLを取得
            ws_url = self._get_websocket_url()

            logger.info(f"Connecting to WebSocket: {ws_url}")

            self.websocket = await websockets.connect(
                ws_url,
                timeout=10,
                ping_interval=30,
                ping_timeout=10
            )

            self.is_running = True
            self.reconnection_manager.reset()

            # メッセージ処理タスクを開始
            asyncio.create_task(self._message_handler())

            logger.info("WebSocket connection established")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False

    async def disconnect(self) -> None:
        """WebSocket接続を切断"""
        self.is_running = False

        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        logger.info("WebSocket connection closed")

    async def subscribe_ticks(self, symbols: List[str]) -> None:
        """ティックデータの購読を開始"""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")

        for symbol in symbols:
            subscribe_message = {
                "action": "subscribe",
                "type": "tick",
                "symbol": symbol
            }

            await self.websocket.send(json.dumps(subscribe_message))
            self.subscribed_symbols.add(symbol)

            logger.info(f"Subscribed to tick data for {symbol}")

    async def subscribe_order_book(self, symbols: List[str]) -> None:
        """板情報の購読を開始"""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")

        for symbol in symbols:
            subscribe_message = {
                "action": "subscribe",
                "type": "orderbook",
                "symbol": symbol
            }

            await self.websocket.send(json.dumps(subscribe_message))

            logger.info(f"Subscribed to order book for {symbol}")

    async def subscribe_indices(self, indices: List[str]) -> None:
        """指数データの購読を開始"""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")

        for index in indices:
            subscribe_message = {
                "action": "subscribe",
                "type": "index",
                "symbol": index
            }

            await self.websocket.send(json.dumps(subscribe_message))
            self.subscribed_indices.add(index)

            logger.info(f"Subscribed to index data for {index}")

    async def subscribe_news(self, symbols: Optional[List[str]] = None) -> None:
        """ニュースデータの購読を開始"""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")

        subscribe_message = {
            "action": "subscribe",
            "type": "news"
        }

        if symbols:
            subscribe_message["symbols"] = symbols

        await self.websocket.send(json.dumps(subscribe_message))

        logger.info(f"Subscribed to news data for symbols: {symbols}")

    async def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """最新のティックデータを取得"""
        # キャッシュから取得を試行
        cache_key = f"latest_tick_{symbol}"
        cached_tick = self.cache_manager.get(cache_key)

        if cached_tick:
            return cached_tick

        # メモリから取得
        return self.latest_ticks.get(symbol)

    async def get_latest_order_book(self, symbol: str) -> Optional[OrderBookData]:
        """最新の板情報を取得"""
        cache_key = f"latest_orderbook_{symbol}"
        cached_order_book = self.cache_manager.get(cache_key)

        if cached_order_book:
            return cached_order_book

        return self.latest_order_books.get(symbol)

    async def get_market_status(self) -> Dict[str, Any]:
        """市場状況を取得"""
        return {
            "is_connected": await self.is_connected(),
            "subscribed_symbols": list(self.subscribed_symbols),
            "subscribed_indices": list(self.subscribed_indices),
            "quality_metrics": self.quality_monitor.get_quality_metrics(),
            "last_update": datetime.now().isoformat()
        }

    async def is_connected(self) -> bool:
        """接続状態を確認"""
        return self.websocket is not None and not self.websocket.closed

    async def _message_handler(self) -> None:
        """WebSocketメッセージハンドラー"""
        while self.is_running and self.websocket:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)

                await self._process_message(data)

            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                await self._handle_reconnection()

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode WebSocket message: {e}")

            except Exception as e:
                logger.error(f"Error in message handler: {e}")

    async def _process_message(self, data: Dict[str, Any]) -> None:
        """受信メッセージを処理"""
        message_type = data.get('type', '')

        if message_type == 'tick':
            await self._process_tick_data(data)
        elif message_type == 'orderbook':
            await self._process_order_book_data(data)
        elif message_type == 'index':
            await self._process_index_data(data)
        elif message_type == 'news':
            await self._process_news_data(data)
        else:
            logger.debug(f"Unknown message type: {message_type}")

    async def _process_tick_data(self, data: Dict[str, Any]) -> None:
        """ティックデータを処理"""
        tick = self.data_normalizer.normalize_tick_data(data)

        if tick and self.quality_monitor.validate_tick_data(tick):
            # メモリに保存
            self.latest_ticks[tick.symbol] = tick

            # キャッシュに保存
            cache_key = f"latest_tick_{tick.symbol}"
            self.cache_manager.set(cache_key, tick, ttl=300)  # 5分間

            # コールバック実行
            for callback in self.tick_callbacks:
                try:
                    callback(tick)
                except Exception as e:
                    logger.error(f"Error in tick callback: {e}")

    async def _process_order_book_data(self, data: Dict[str, Any]) -> None:
        """板情報データを処理"""
        order_book = self.data_normalizer.normalize_order_book(data)

        if order_book and self.quality_monitor.validate_order_book(order_book):
            # メモリに保存
            self.latest_order_books[order_book.symbol] = order_book

            # キャッシュに保存
            cache_key = f"latest_orderbook_{order_book.symbol}"
            self.cache_manager.set(cache_key, order_book, ttl=60)  # 1分間

            # コールバック実行
            for callback in self.order_book_callbacks:
                try:
                    callback(order_book)
                except Exception as e:
                    logger.error(f"Error in order book callback: {e}")

    async def _process_index_data(self, data: Dict[str, Any]) -> None:
        """指数データを処理"""
        index_data = self.data_normalizer.normalize_index_data(data)

        if index_data:
            # メモリに保存
            self.latest_indices[index_data.symbol] = index_data

            # キャッシュに保存
            cache_key = f"latest_index_{index_data.symbol}"
            self.cache_manager.set(cache_key, index_data, ttl=300)  # 5分間

            # コールバック実行
            for callback in self.index_callbacks:
                try:
                    callback(index_data)
                except Exception as e:
                    logger.error(f"Error in index callback: {e}")

    async def _process_news_data(self, data: Dict[str, Any]) -> None:
        """ニュースデータを処理"""
        news = self.data_normalizer.normalize_news_data(data)

        if news:
            # キャッシュに保存
            cache_key = f"news_{news.id}"
            self.cache_manager.set(cache_key, news, ttl=3600)  # 1時間

            # コールバック実行
            for callback in self.news_callbacks:
                try:
                    callback(news)
                except Exception as e:
                    logger.error(f"Error in news callback: {e}")

    async def _handle_reconnection(self) -> None:
        """再接続処理"""
        if not self.reconnection_manager.should_retry():
            logger.error("Max reconnection attempts reached")
            return

        delay = self.reconnection_manager.get_delay()
        self.reconnection_manager.record_attempt()

        logger.info(f"Attempting reconnection in {delay} seconds (attempt {self.reconnection_manager.retry_count})")

        await asyncio.sleep(delay)

        if await self.connect():
            # 購読を再開
            if self.subscribed_symbols:
                await self.subscribe_ticks(list(self.subscribed_symbols))
            if self.subscribed_indices:
                await self.subscribe_indices(list(self.subscribed_indices))

    def _get_websocket_url(self) -> str:
        """設定からWebSocket URLを取得"""
        # デフォルトではYahoo Finance WebSocketを使用
        # 実際の運用では、適切なWebSocket APIのURLに変更する必要があります
        data_source = self.settings.real_time.data_source

        if data_source == "yahoo":
            return "wss://streamer.finance.yahoo.com"
        elif data_source == "alpha_vantage":
            return "wss://ws.finnhub.io"
        else:
            # デモ用のWebSocketサーバー
            return "wss://demo-realtime-data.example.com/ws"

    def add_tick_callback(self, callback: Callable[[TickData], None]) -> None:
        """ティックデータコールバックを追加"""
        self.tick_callbacks.append(callback)

    def add_order_book_callback(self, callback: Callable[[OrderBookData], None]) -> None:
        """板情報コールバックを追加"""
        self.order_book_callbacks.append(callback)

    def add_index_callback(self, callback: Callable[[IndexData], None]) -> None:
        """指数データコールバックを追加"""
        self.index_callbacks.append(callback)

    def add_news_callback(self, callback: Callable[[NewsData], None]) -> None:
        """ニュースデータコールバックを追加"""
        self.news_callbacks.append(callback)

    # DataProviderインターフェースの実装
    def get_stock_data(self, symbol: str, period: str = "1d") -> pd.DataFrame:
        """過去データの取得（リアルタイムデータプロバイダーでは簡易実装）"""
        # リアルタイムデータから過去1日分のデータを構築
        # 実際の実装では、適切な履歴データサービスを使用する
        return pd.DataFrame()

    def get_technical_indicators(self, symbol: str) -> Dict[str, float]:
        """技術指標の取得（リアルタイムデータから算出）"""
        # 簡易実装：最新のティックデータから基本的な指標を算出
        latest_tick = self.latest_ticks.get(symbol)
        if latest_tick:
            return {
                "last_price": latest_tick.price,
                "volume": float(latest_tick.volume),
                "bid_ask_spread": (latest_tick.ask_price - latest_tick.bid_price)
                                 if latest_tick.bid_price and latest_tick.ask_price else 0.0
            }
        return {}