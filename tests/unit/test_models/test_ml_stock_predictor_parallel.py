from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

import pandas as pd
import pandas.testing as pdt

# lightgbmが利用できない場合はテストをスキップ
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

if not LIGHTGBM_AVAILABLE:
    pytest.skip("lightgbm not available", allow_module_level=True)

from models.ml_stock_predictor import MLStockPredictor


@pytest.mark.parametrize("symbols", [["AAA", "BBB", "CCC"]])
def test_prepare_dataset_parallel_matches_sequential_and_submits_tasks(
    monkeypatch,
    symbols,
):
    predictor = MLStockPredictor()

    # Ensure deterministic data and feature creation
    def fake_get_stock_data(symbol, _range):
        return pd.DataFrame({"dummy": range(120), "symbol": [symbol] * 120})

    monkeypatch.setattr(predictor.data_provider, "get_stock_data", fake_get_stock_data)

    def fake_prepare_features(data):
        symbol = data["symbol"].iloc[0]
        symbol_value = sum(ord(ch) for ch in symbol)
        return pd.DataFrame({"feature": [symbol_value]}, index=[0])

    def fake_create_targets(data):
        idx = len(data)
        return (
            pd.DataFrame({"reg": [idx]}, index=[idx]),
            pd.DataFrame({"cls": [idx]}, index=[idx]),
        )

    monkeypatch.setattr(predictor, "prepare_features", fake_prepare_features)
    monkeypatch.setattr(predictor, "create_targets", fake_create_targets)

    sequential = predictor.prepare_dataset(symbols, parallel=False)

    submitted_symbols = []

    def submit_side_effect(func, symbol):
        submitted_symbols.append(symbol)
        future = Future()
        future.set_result(func(symbol))
        return future

    mock_executor = MagicMock()
    mock_executor.__enter__.return_value.submit.side_effect = submit_side_effect

    monkeypatch.setattr(
        "models.ml_stock_predictor.ThreadPoolExecutor",
        lambda max_workers=None: mock_executor,
    )

    parallel = predictor.prepare_dataset(symbols, parallel=True)

    assert submitted_symbols == symbols

    for seq, par in zip(sequential, parallel):
        if isinstance(seq, pd.DataFrame):
            pdt.assert_frame_equal(seq.sort_index(axis=1), par.sort_index(axis=1))
        else:
            assert seq == par
