import sys
import types

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
