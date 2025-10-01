import sys
import types
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.__dict__.update(
        {
            "__version__": "0",
            "array": lambda *args, **kwargs: args[0] if args else None,
            "ndarray": object,
        }
    )
    sys.modules["numpy"] = numpy_stub


if "scipy" not in sys.modules:
    scipy_stub = types.ModuleType("scipy")
    scipy_stats_stub = types.ModuleType("scipy.stats")
    scipy_stub.stats = scipy_stats_stub
    sys.modules["scipy"] = scipy_stub
    sys.modules["scipy.stats"] = scipy_stats_stub


if "matplotlib" not in sys.modules:
    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = types.ModuleType("matplotlib.pyplot")

    def _no_op(*_, **__):  # pragma: no cover - chart stubs
        return None

    pyplot_stub.figure = _no_op
    pyplot_stub.plot = _no_op
    pyplot_stub.title = _no_op
    pyplot_stub.xlabel = _no_op
    pyplot_stub.ylabel = _no_op
    pyplot_stub.legend = _no_op
    pyplot_stub.tight_layout = _no_op
    pyplot_stub.savefig = _no_op
    pyplot_stub.close = _no_op
    pyplot_stub.style = types.SimpleNamespace(use=_no_op)

    matplotlib_stub.pyplot = pyplot_stub

    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub


if "seaborn" not in sys.modules:
    seaborn_stub = types.ModuleType("seaborn")

    def _no_op(*_, **__):  # pragma: no cover - seaborn stubs
        return None

    seaborn_stub.set_style = _no_op
    seaborn_stub.lineplot = _no_op
    seaborn_stub.barplot = _no_op
    seaborn_stub.histplot = _no_op
    seaborn_stub.set_palette = _no_op

    sys.modules["seaborn"] = seaborn_stub


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
