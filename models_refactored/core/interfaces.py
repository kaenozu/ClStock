"""
統合されたClStock予測システムインターフェース
全ての予測モデルが準拠すべき標準インターフェース定義
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Protocol
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class PredictionMode(Enum):
    """予測モード"""

    CONSERVATIVE = "conservative"  # 保守的予測
    BALANCED = "balanced"  # バランス予測
    AGGRESSIVE = "aggressive"  # 積極的予測
    ULTRA_FAST = "ultra_fast"  # 超高速予測
    HIGH_PRECISION = "high_precision"  # 高精度予測


class ModelType(Enum):
    """モデルタイプ"""

    ENSEMBLE = "ensemble"
    DEEP_LEARNING = "deep_learning"
    HYBRID = "hybrid"
    PRECISION_87 = "precision_87"
    PARALLEL = "parallel"


@dataclass
class PredictionResult:
    """統一された予測結果クラス"""

    prediction: float  # 予測値 (0-100)
    confidence: float  # 信頼度 (0-1)
    accuracy: float  # 推定精度 (0-100)
    timestamp: datetime  # 予測実行時刻
    symbol: str  # 銘柄コード
    model_type: ModelType = ModelType.ENSEMBLE  # 使用モデルタイプ（デフォルト値追加）
    execution_time: float = 0.0  # 実行時間(秒)（デフォルト値追加）
    metadata: Dict[str, Any] = None  # 追加メタデータ（デフォルト値追加）

    def __post_init__(self):
        """データクラス初期化後の処理"""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式での出力"""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "accuracy": self.accuracy,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "model_type": self.model_type.value,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


@dataclass
class BatchPredictionResult:
    """バッチ予測結果クラス"""

    predictions: Dict[str, float]  # 銘柄ごとの予測値
    errors: Dict[str, str]  # エラーが発生した銘柄とエラーメッセージ
    metadata: Dict[str, Any] = None  # 追加メタデータ

    def __post_init__(self):
        """データクラス初期化後の処理"""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelConfiguration:
    """モデル設定クラス"""

    model_type: ModelType = ModelType.ENSEMBLE
    prediction_mode: PredictionMode = PredictionMode.BALANCED
    cache_enabled: bool = True
    parallel_enabled: bool = True
    max_workers: int = 4
    cache_size: int = 1000
    timeout_seconds: int = 300
    custom_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class PerformanceMetrics:
    """性能指標クラス"""

    accuracy: float  # 精度
    precision: float  # 適合率
    recall: float  # 再現率
    f1_score: float  # F1スコア
    execution_time: float  # 実行時間
    memory_usage: float  # メモリ使用量
    cache_hit_rate: float  # キャッシュヒット率
    total_predictions: int  # 総予測数
    successful_predictions: int  # 成功予測数


class DataProvider(Protocol):
    """データプロバイダーインターフェース"""

    def get_stock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """株価データ取得"""
        ...

    def get_market_data(self) -> Dict[str, Any]:
        """市場データ取得"""
        ...

    def is_market_open(self) -> bool:
        """市場開場状態確認"""
        ...


class StockPredictor(ABC):
    """株価予測システムの統合ベースクラス"""

    def __init__(self, config: ModelConfiguration):
        self.config = config
        self.is_trained = False
        self.performance_metrics: Optional[PerformanceMetrics] = None
        self._model_version = "2.0.0"

    @abstractmethod
    def predict(self, symbol: str) -> PredictionResult:
        """単一銘柄の予測実行"""
        pass

    @abstractmethod
    def predict_batch(self, symbols: List[str]) -> List[PredictionResult]:
        """バッチ予測実行"""
        pass

    @abstractmethod
    def train(self, data: pd.DataFrame) -> bool:
        """モデル訓練"""
        pass

    @abstractmethod
    def get_confidence(self, symbol: str) -> float:
        """信頼度取得"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得"""
        pass

    # オプショナルメソッド（デフォルト実装）

    def validate_symbol(self, symbol: str) -> bool:
        """銘柄コード検証"""
        if not symbol or not isinstance(symbol, str):
            return False
        return len(symbol) == 4 and symbol.isdigit()

    def get_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """性能指標取得"""
        return self.performance_metrics

    def set_prediction_mode(self, mode: PredictionMode):
        """予測モード設定"""
        self.config.prediction_mode = mode

    def is_ready(self) -> bool:
        """予測実行可能状態確認"""
        return self.is_trained

    def get_version(self) -> str:
        """モデルバージョン取得"""
        return self._model_version


class CacheProvider(Protocol):
    """キャッシュプロバイダーインターフェース"""

    def get(self, key: str) -> Optional[Any]:
        """キャッシュから値取得"""
        ...

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """キャッシュに値設定"""
        ...

    def clear(self) -> bool:
        """キャッシュクリア"""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計取得"""
        ...


class ModelRegistry(Protocol):
    """モデルレジストリインターフェース"""

    def register_model(self, model: StockPredictor, name: str) -> bool:
        """モデル登録"""
        ...

    def get_model(self, name: str) -> Optional[StockPredictor]:
        """モデル取得"""
        ...

    def list_models(self) -> List[str]:
        """登録モデル一覧"""
        ...

    def remove_model(self, name: str) -> bool:
        """モデル削除"""
        ...


# 型エイリアス
PredictionTarget = Union[str, List[str]]
ModelParams = Dict[str, Any]
TrainingData = Union[pd.DataFrame, Dict[str, pd.DataFrame]]
