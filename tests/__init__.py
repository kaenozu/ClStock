import sys
import types
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


if "data" not in sys.modules:
    data_module = types.ModuleType("data")
    sys.modules["data"] = data_module
else:
    data_module = sys.modules["data"]

stock_data_stub = types.ModuleType("data.stock_data")


class _StubStockDataProvider:
    def get_stock_data(self, symbol: str, data_range: str):
        raise NotImplementedError

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return data


stock_data_stub.StockDataProvider = _StubStockDataProvider
sys.modules["data.stock_data"] = stock_data_stub
setattr(data_module, "stock_data", stock_data_stub)


if "models" not in sys.modules:
    models_module = types.ModuleType("models")
    sys.modules["models"] = models_module
else:
    models_module = sys.modules["models"]

recommendation_stub = types.ModuleType("models.recommendation")


@dataclass
class _StubStockRecommendation:
    rank: int = 0
    symbol: str = ""
    company_name: str = ""
    buy_timing: str = ""
    target_price: float = 0.0
    stop_loss: float = 0.0
    profit_target_1: float = 0.0
    profit_target_2: float = 0.0
    holding_period: str = ""
    score: float = 0.0
    current_price: float = 0.0
    recommendation_reason: str = ""
    recommendation_level: str = "neutral"
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self):  # pragma: no cover - compatibility helper
        return {
            "rank": self.rank,
            "symbol": self.symbol,
            "company_name": self.company_name,
            "buy_timing": self.buy_timing,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "profit_target_1": self.profit_target_1,
            "profit_target_2": self.profit_target_2,
            "holding_period": self.holding_period,
            "score": self.score,
            "current_price": self.current_price,
            "recommendation_reason": self.recommendation_reason,
            "recommendation_level": self.recommendation_level,
            "generated_at": self.generated_at.isoformat(),
        }


recommendation_stub.StockRecommendation = _StubStockRecommendation
sys.modules["models.recommendation"] = recommendation_stub
setattr(models_module, "recommendation", recommendation_stub)
