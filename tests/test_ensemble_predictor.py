import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from models.ensemble.ensemble_predictor import EnsembleStockPredictor
from models.ml_stock_predictor import MLStockPredictor
from models.recommendation import StockRecommendation
from data.stock_data import StockDataProvider


@pytest.fixture
def mock_stock_data_provider():
    mock = MagicMock()
    mock.get_stock_data.return_value = pd.DataFrame(
        {
            "Close": np.random.rand(100) * 100,
            "Volume": np.random.randint(1000, 10000, 100),
        },
        index=pd.to_datetime(pd.date_range(start="2023-01-01", periods=100)),
    )
    return mock


class TestEnsembleStockPredictor:
    """EnsembleStockPredictorクラスのテスト"""

    def test_add_model(self):
        """モデル追加のテスト"""
        ensemble = EnsembleStockPredictor()
        mock_model = MagicMock()
        ensemble.add_model("test_model", mock_model, 0.5)
        assert "test_model" in ensemble.models
        assert ensemble.weights["test_model"] == 0.5

    def test_train_ensemble(self, mock_stock_data_provider):
        """アンサンブルモデル訓練のテスト"""
        # Mock MLStockPredictor to avoid actual data fetching and complex training
        with patch("models.ml_stock_predictor.MLStockPredictor") as MockMLPredictor:
            mock_ml_predictor_instance = MockMLPredictor.return_value
            mock_ml_predictor_instance.prepare_dataset.return_value = (
                pd.DataFrame(np.random.rand(100, 10)),  # features
                pd.DataFrame(
                    {"recommendation_score": np.random.rand(100)}
                ),  # targets_reg
                pd.DataFrame(
                    {"class_target": np.random.randint(0, 2, 100)}
                ),  # targets_cls
            )
            mock_ml_predictor_instance.prepare_features.return_value = pd.DataFrame(
                np.random.rand(1, 10)
            )

            ensemble = EnsembleStockPredictor()
            ensemble.data_provider = mock_stock_data_provider  # Inject mock
            ensemble.prepare_ensemble_models()
            ensemble.train_ensemble(
                symbols=["7203"], target_column="recommendation_score"
            )

            assert ensemble.is_trained
            assert len(ensemble.models) > 0
            assert sum(ensemble.weights.values()) == pytest.approx(1.0)

    def test_predict_score(self, mock_stock_data_provider):
        """予測スコアのテスト"""
        ensemble = EnsembleStockPredictor()
        ensemble.data_provider = mock_stock_data_provider
        ensemble.is_trained = True  # Assume trained for prediction test
        ensemble.scaler = MagicMock()
        ensemble.scaler.transform.return_value = np.random.rand(1, 10)

        # Add a mock model to the ensemble
        mock_sub_model = MagicMock()
        mock_sub_model.predict.return_value = np.array([70.0])
        ensemble.add_model("mock_model", mock_sub_model, 1.0)

        score = ensemble.predict_score("7203")
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_save_load_ensemble(self):
        """アンサンブルモデルの保存と読み込みのテスト"""
        ensemble = EnsembleStockPredictor()
        ensemble.prepare_ensemble_models()
        ensemble.is_trained = True
        ensemble.feature_names = ["feat1", "feat2"]
        ensemble.scaler = MagicMock()

        ensemble.save_ensemble()

        loaded_ensemble = EnsembleStockPredictor()
        assert loaded_ensemble.load_ensemble()
        assert loaded_ensemble.is_trained
        assert loaded_ensemble.feature_names == ["feat1", "feat2"]
        assert isinstance(loaded_ensemble.scaler, MagicMock)
        assert len(loaded_ensemble.models) > 0
