import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import numpy as np
import pandas as pd

# data.stock_data は構文エラーを含むためテストでは軽量スタブを使用する
if "data.stock_data" not in sys.modules:
    import data

    dummy_stock_data = ModuleType("data.stock_data")

    class _DummyStockDataProvider:  # pragma: no cover - シンプルスタブ
        def __init__(self, *_, **__):
            pass

    dummy_stock_data.StockDataProvider = _DummyStockDataProvider
    sys.modules["data.stock_data"] = dummy_stock_data
    data.stock_data = dummy_stock_data

from models.ensemble.ensemble_predictor import EnsembleStockPredictor


class IdentityScaler:
    def fit_transform(self, X):
        return np.asarray(X)

    def transform(self, X):
        return np.asarray(X)


class DeterministicModel:
    def __init__(self, predictions_by_length):
        self.predictions_by_length = {
            length: [np.asarray(pred) for pred in preds]
            for length, preds in predictions_by_length.items()
        }
        self.fit_calls = []

    def fit(self, X, y):
        self.fit_calls.append((np.asarray(X), np.asarray(y)))
        return self

    def predict(self, X):
        length = len(X)
        if length not in self.predictions_by_length:
            raise AssertionError(f"No predictions configured for length {length}")
        if not self.predictions_by_length[length]:
            raise AssertionError(f"No predictions left for length {length}")
        return self.predictions_by_length[length].pop(0)


class DummyTimeSeriesSplit:
    def __init__(self, splits):
        self._splits = splits

    def split(self, X):
        for train_idx, test_idx in self._splits:
            yield np.asarray(train_idx), np.asarray(test_idx)


@pytest.mark.unit
def test_train_ensemble_uses_time_series_split_scores_for_weights():
    predictor = EnsembleStockPredictor(data_provider=Mock())

    model_a = DeterministicModel(
        {
            4: [[0.0, 0.0, 0.0, 0.0]],
            2: [[2.0, 2.0]],
            1: [[4.0]],
        },
    )
    model_b = DeterministicModel(
        {
            4: [[1.0, 1.0, 1.0, 1.0]],
            2: [[3.0, 3.0]],
            1: [[2.0]],
        },
    )

    predictor.models = {"model_a": model_a, "model_b": model_b}
    predictor.weights = {"model_a": 0.5, "model_b": 0.5}

    features = pd.DataFrame(
        {
            "f1": [0, 1, 2, 3, 4, 5],
            "f2": [5, 4, 3, 2, 1, 0],
        },
    )
    targets = pd.DataFrame({"recommendation_score": [0, 0, 1, 2, 2, 2]})

    splits = [
        (np.array([0, 1, 2]), np.array([3, 4])),
        (np.array([0, 1, 2, 3, 4]), np.array([5])),
    ]

    mock_settings = SimpleNamespace(
        model=SimpleNamespace(min_training_data=1, train_test_split=0.8),
    )

    original_ensemble_predict = (
        EnsembleStockPredictor._ensemble_predict_from_predictions
    )

    with patch.object(predictor, "prepare_ensemble_models"), patch.object(
        predictor, "save_ensemble",
    ), patch("config.settings.get_settings", return_value=mock_settings), patch(
        "models.ml_stock_predictor.MLStockPredictor",
    ) as mock_ml_predictor_cls, patch(
        "models.ensemble.ensemble_predictor.StandardScaler",
    ) as mock_scaler_cls, patch(
        "models.ensemble.ensemble_predictor.TimeSeriesSplit",
        return_value=DummyTimeSeriesSplit(splits),
        create=True,
    ), patch.object(
        EnsembleStockPredictor,
        "_ensemble_predict_from_predictions",
        autospec=True,
    ) as mock_ensemble_predict:
        mock_ml_predictor = mock_ml_predictor_cls.return_value
        mock_ml_predictor.prepare_dataset.return_value = (
            features,
            targets,
            pd.DataFrame(),
        )
        mock_scaler_cls.side_effect = lambda: IdentityScaler()
        mock_ensemble_predict.side_effect = (
            lambda self, preds: original_ensemble_predict(self, preds)
        )

        predictor.train_ensemble(["TEST"], target_column="recommendation_score")

    # 平均MSEに基づいた動的重み調整の検証
    assert predictor.weights["model_b"] == pytest.approx(0.8, rel=1e-2)
    assert predictor.weights["model_a"] == pytest.approx(0.2, rel=1e-2)

    # アンサンブル評価に渡された予測が全fold分含まれていることを検証
    passed_predictions = mock_ensemble_predict.call_args[0][1]
    np.testing.assert_allclose(passed_predictions["model_a"], np.array([2.0, 2.0, 4.0]))
    np.testing.assert_allclose(passed_predictions["model_b"], np.array([3.0, 3.0, 2.0]))

    # foldごとのターゲット数と最終学習データのfit呼び出しを確認
    assert any(len(call[0]) == len(features) for call in model_a.fit_calls)
    assert any(len(call[0]) == len(features) for call in model_b.fit_calls)
