#!/usr/bin/env python3
"""
基底インターフェースと抽象クラス定義
ClStockシステムの共通インターフェース
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class PredictionResult:
    """予測結果の標準データ構造"""
    prediction: float
    confidence: float
    accuracy: float
    timestamp: datetime
    symbol: str
    metadata: Dict[str, Any]


@dataclass
class ModelPerformance:
    """モデルパフォーマンス指標"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mae: float
    rmse: float


class StockPredictor(ABC):
    """株価予測システムの基底インターフェース"""

    @abstractmethod
    def predict(self, symbol: str) -> PredictionResult:
        """単一銘柄の予測を実行"""
        pass

    @abstractmethod
    def predict_batch(self, symbols: List[str]) -> List[PredictionResult]:
        """複数銘柄の一括予測"""
        pass

    @abstractmethod
    def get_confidence(self, symbol: str) -> float:
        """予測信頼度を取得"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        pass


class DataProvider(ABC):
    """データ提供者の基底インターフェース"""

    @abstractmethod
    def get_stock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """株価データ取得"""
        pass

    @abstractmethod
    def get_technical_indicators(self, symbol: str) -> Dict[str, float]:
        """技術指標取得"""
        pass


@dataclass
class TickData:
    """株価ティックデータ"""
    
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    trade_type: str = "unknown"  # buy, sell, unknown


@dataclass  
class OrderBookData:
    """板情報データ"""
    
    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, int]]  # (price, volume) のリスト
    asks: List[Tuple[float, int]]  # (price, volume) のリスト
    

@dataclass
class IndexData:
    """指数データ"""
    
    symbol: str  # NIKKEI, TOPIX, etc.
    timestamp: datetime
    value: float
    change: float
    change_percent: float


@dataclass
class NewsData:
    """ニュース・イベントデータ"""
    
    id: str
    timestamp: datetime
    title: str
    content: str
    symbols: List[str]  # 関連銘柄
    sentiment: Optional[str] = None  # positive, negative, neutral
    impact_score: Optional[float] = None  # 0-1のスコア


@dataclass
class MarketData:
    """市場データの統合形式"""
    
    timestamp: datetime
    tick_data: List[TickData]
    order_books: List[OrderBookData]
    indices: List[IndexData]
    news: List[NewsData]


class RealTimeDataProvider(DataProvider):
    """リアルタイムデータプロバイダーの基底インターフェース"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """データソースに接続"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """データソース接続を切断"""
        pass
    
    @abstractmethod
    async def subscribe_ticks(self, symbols: List[str]) -> None:
        """ティックデータの購読を開始"""
        pass
    
    @abstractmethod
    async def subscribe_order_book(self, symbols: List[str]) -> None:
        """板情報の購読を開始"""
        pass
    
    @abstractmethod
    async def subscribe_indices(self, indices: List[str]) -> None:
        """指数データの購読を開始"""
        pass
    
    @abstractmethod
    async def subscribe_news(self, symbols: Optional[List[str]] = None) -> None:
        """ニュースデータの購読を開始"""
        pass
    
    @abstractmethod
    async def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """最新のティックデータを取得"""
        pass
    
    @abstractmethod
    async def get_latest_order_book(self, symbol: str) -> Optional[OrderBookData]:
        """最新の板情報を取得"""
        pass
    
    @abstractmethod
    async def get_market_status(self) -> Dict[str, Any]:
        """市場状況を取得"""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """接続状態を確認"""
        pass


class DataQualityMonitor(ABC):
    """データ品質監視インターフェース"""
    
    @abstractmethod
    def validate_tick_data(self, tick: TickData) -> bool:
        """ティックデータの品質検証"""
        pass
    
    @abstractmethod
    def validate_order_book(self, order_book: OrderBookData) -> bool:
        """板情報の品質検証"""
        pass
    
    @abstractmethod
    def get_quality_metrics(self) -> Dict[str, float]:
        """データ品質メトリクスを取得"""
        pass


class ModelTrainer(ABC):
    """モデル訓練の基底インターフェース"""

    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        """モデル訓練実行"""
        pass

    @abstractmethod
    def evaluate(self, test_data: pd.DataFrame) -> ModelPerformance:
        """モデル評価実行"""
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """モデル保存"""
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """モデル読み込み"""
        pass


class CacheManager(ABC):
    """キャッシュ管理の基底インターフェース"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """キャッシュ値取得"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """キャッシュ値設定"""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """キャッシュ削除"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """全キャッシュ削除"""
        pass


class PerformanceMonitor(ABC):
    """パフォーマンス監視の基底インターフェース"""

    @abstractmethod
    def record_prediction(self, result: PredictionResult) -> None:
        """予測結果記録"""
        pass

    @abstractmethod
    def get_accuracy_metrics(self, period_days: int = 30) -> Dict[str, float]:
        """精度指標取得"""
        pass

    @abstractmethod
    def get_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポート生成"""
        pass