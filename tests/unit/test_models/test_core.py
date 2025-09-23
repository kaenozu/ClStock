"""Test core ML models."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 蜿､縺・odels繝・ぅ繝ｬ繧ｯ繝医Μ縺ｯ蟒・ｭ｢縺輔ｌ縺ｾ縺励◆縲Ｎodels_refactored縺ｫ遘ｻ陦梧ｸ医∩
# from models.core import MLStockPredictor, EnsembleStockPredictor
# from models.base import PredictionResult

# 譁ｰ縺励＞import繝代せ
from models.core import (
    EnsembleStockPredictor,
    MLStockPredictor,
    PredictionResult,
)


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)

    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    return pd.DataFrame(
        {
            "Open": close_prices + np.random.randn(100) * 0.1,
            "High": close_prices + np.abs(np.random.randn(100) * 0.3),
            "Low": close_prices - np.abs(np.random.randn(100) * 0.3),
            "Close": close_prices,
            "Volume": np.random.randint(1000, 10000, 100),
            "SMA_20": close_prices,  # Simplified for testing
            "SMA_50": close_prices,
            "RSI": np.random.uniform(30, 70, 100),
            "MACD": np.random.randn(100) * 0.1,
            "MACD_Signal": np.random.randn(100) * 0.1,
            "ATR": np.random.uniform(0.5, 2.0, 100),
        },
        index=dates,
    )


class TestMLStockPredictor:
    """Test MLStockPredictor class."""

    def test_initialization(self):
        """Test MLStockPredictor initialization."""
        predictor = MLStockPredictor("xgboost")

        assert predictor.model_type == "xgboost"
        assert not predictor.is_trained()
        assert predictor.model is None
        assert predictor.scaler is not None

    def test_initialization_default(self):
        """Test MLStockPredictor default initialization."""
        predictor = MLStockPredictor()
        assert predictor.model_type == "xgboost"

    @patch("models.core.MLStockPredictor.data_provider")
    def test_prepare_features_empty_data(self, mock_data_provider):
        """Test prepare_features with empty data."""
        predictor = MLStockPredictor()

        result = predictor.prepare_features(pd.DataFrame())
        assert result.empty

    @patch("models.core.MLStockPredictor.data_provider")
    def test_prepare_features_valid_data(self, mock_data_provider, sample_stock_data):
        """Test prepare_features with valid data."""
        predictor = MLStockPredictor()
        mock_data_provider.calculate_technical_indicators.return_value = (
            sample_stock_data
        )

        features = predictor.prepare_features(sample_stock_data)

        assert not features.empty
        assert "price_change" in features.columns
        assert "volume_change" in features.columns
        assert "rsi" in features.columns

    def test_create_targets(self, sample_stock_data):
        """Test create_targets method."""
        predictor = MLStockPredictor()

        targets_reg, targets_cls = predictor.create_targets(sample_stock_data)

        assert not targets_reg.empty
        assert not targets_cls.empty
        assert "return_1d" in targets_reg.columns
        assert "direction_1d" in targets_cls.columns
        assert "recommendation_score" in targets_reg.columns

    def test_calculate_future_performance_score(self, sample_stock_data):
        """Test _calculate_future_performance_score method."""
        predictor = MLStockPredictor()

        scores = predictor._calculate_future_performance_score(sample_stock_data)

        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_stock_data)
        # Check scores are in valid range
        valid_scores = scores.dropna()
        assert all(0 <= score <= 100 for score in valid_scores)

    @patch("models.core.MLStockPredictor.data_provider")
    def test_prepare_dataset_empty_symbols(self, mock_data_provider):
        """Test prepare_dataset with empty symbols list."""
        predictor = MLStockPredictor()

        with pytest.raises(ValueError, match="No valid data available"):
            predictor.prepare_dataset([])

    @patch("models.core.MLStockPredictor.data_provider")
    def test_prepare_dataset_insufficient_data(self, mock_data_provider):
        """Test prepare_dataset with insufficient data."""
        predictor = MLStockPredictor()
        mock_data_provider.get_stock_data.return_value = pd.DataFrame()  # Empty data

        with pytest.raises(ValueError, match="No valid data available"):
            predictor.prepare_dataset(["AAPL"])

    @patch("joblib.dump")
    def test_save_model(self, mock_dump):
        """Test save_model method."""
        predictor = MLStockPredictor()
        predictor.model = Mock()
        predictor.feature_names = ["feature1", "feature2"]

        predictor.save_model()

        assert mock_dump.call_count == 3  # model, scaler, features

    @patch("joblib.load")
    @patch("pathlib.Path.exists")
    def test_load_model_success(self, mock_exists, mock_load):
        """Test successful model loading."""
        predictor = MLStockPredictor()
        mock_exists.return_value = True
        mock_load.side_effect = [Mock(), Mock(), ["feature1", "feature2"]]

        result = predictor.load_model()

        assert result is True
        assert predictor.is_trained() is True
        assert mock_load.call_count == 3

    @patch("pathlib.Path.exists")
    def test_load_model_files_not_exist(self, mock_exists):
        """Test model loading when files don't exist."""
        predictor = MLStockPredictor()
        mock_exists.return_value = False

        result = predictor.load_model()

        assert result is False
        assert predictor.is_trained() is False

    @patch("models.core.MLStockPredictor.data_provider")
    def test_predict_not_trained(self, mock_data_provider):
        """Test predict when model is not trained."""
        predictor = MLStockPredictor()

        with pytest.raises(ValueError, match="Model must be trained"):
            predictor.predict("AAPL")

    @patch("models.core.MLStockPredictor.load_model")
    @patch("models.core.MLStockPredictor.data_provider")
    def test_predict_with_training_data(
        self, mock_data_provider, mock_load_model, sample_stock_data
    ):
        """Test predict with valid training and data."""
        predictor = MLStockPredictor()

        # Mock training
        predictor._is_trained = True
        predictor.model = Mock()
        predictor.model.predict.return_value = [75.0]
        predictor.scaler = Mock()
        predictor.scaler.transform.return_value = np.array([[1, 2, 3]])
        predictor.feature_names = ["feature1", "feature2", "feature3"]

        mock_data_provider.get_stock_data.return_value = sample_stock_data

        with patch.object(predictor, "prepare_features") as mock_prepare_features:
            mock_features = pd.DataFrame(
                {"feature1": [1], "feature2": [2], "feature3": [3]}
            )
            mock_prepare_features.return_value = mock_features

            result = predictor.predict("AAPL")

            assert isinstance(result, PredictionResult)
            assert result.prediction == 75.0
            assert result.metadata["symbol"] == "AAPL"

    def test_predict_score_integration(self):
        """Test predict_score method integration."""
        predictor = MLStockPredictor()

        with patch.object(predictor, "predict") as mock_predict:
            mock_predict.return_value = PredictionResult(
                prediction=80.0, confidence=0.9, timestamp=datetime.now(), metadata={}
            )

            score = predictor.predict_score("AAPL")
            assert score == 80.0

    def test_predict_return_rate_integration(self):
        """Test predict_return_rate method integration."""
        predictor = MLStockPredictor()

        with patch.object(predictor, "predict") as mock_predict:
            mock_predict.return_value = PredictionResult(
                prediction=0.05, confidence=0.9, timestamp=datetime.now(), metadata={}
            )

            return_rate = predictor.predict_return_rate("AAPL", days=5)
            expected_max = 0.006 * 5  # max return per the algorithm
            assert abs(return_rate) <= expected_max

    def test_get_feature_importance_no_model(self):
        """Test get_feature_importance when no model is trained."""
        predictor = MLStockPredictor()

        importance = predictor.get_feature_importance()
        assert importance == {}

    def test_get_feature_importance_with_model(self):
        """Test get_feature_importance with trained model."""
        predictor = MLStockPredictor()
        predictor._is_trained = True
        predictor.model = Mock()
        predictor.model.feature_importances_ = [0.1, 0.2, 0.3]
        predictor.feature_names = ["feature1", "feature2", "feature3"]

        importance = predictor.get_feature_importance()

        expected = {"feature1": 0.1, "feature2": 0.2, "feature3": 0.3}
        assert importance == expected


class TestEnsembleStockPredictor:
    """Test EnsembleStockPredictor class."""

    def test_initialization(self):
        """Test EnsembleStockPredictor initialization."""
        ensemble = EnsembleStockPredictor()

        assert ensemble.model_type == "ensemble"
        assert ensemble.data_provider is not None

    def test_train_empty_ensemble(self):
        """Test training empty ensemble."""
        ensemble = EnsembleStockPredictor()

        # Should complete without error even with no models
        ensemble.train(pd.DataFrame(), pd.Series())
        assert ensemble.is_trained() is True

    def test_train_with_models(self, sample_stock_data):
        """Test training ensemble with models."""
        ensemble = EnsembleStockPredictor()

        model1 = Mock()
        model2 = Mock()

        ensemble.add_model(model1, 0.6)
        ensemble.add_model(model2, 0.4)

        target = pd.Series([1, 2, 3])
        ensemble.train(sample_stock_data, target)

        model1.train.assert_called_once_with(sample_stock_data, target)
        model2.train.assert_called_once_with(sample_stock_data, target)
        assert ensemble.is_trained() is True

    @patch("models.core.EnsembleStockPredictor.data_provider")
    def test_predict_not_trained(self, mock_data_provider):
        """Test predict when ensemble is not trained."""
        ensemble = EnsembleStockPredictor()

        with pytest.raises(ValueError, match="Ensemble must be trained"):
            ensemble.predict("AAPL")

    @patch("models.core.EnsembleStockPredictor.data_provider")
    def test_predict_with_models(self, mock_data_provider, sample_stock_data):
        """Test predict with trained models."""
        ensemble = EnsembleStockPredictor()
        ensemble._is_trained = True

        model1 = Mock()
        model1.predict.return_value = PredictionResult(
            prediction=70.0, confidence=0.8, timestamp=datetime.now(), metadata={}
        )

        model2 = Mock()
        model2.predict.return_value = PredictionResult(
            prediction=80.0, confidence=0.9, timestamp=datetime.now(), metadata={}
        )

        ensemble.add_model(model1, 0.6)
        ensemble.add_model(model2, 0.4)

        mock_data_provider.get_stock_data.return_value = sample_stock_data

        result = ensemble.predict("AAPL")

        assert isinstance(result, PredictionResult)
        # Weighted average: (70*0.6 + 80*0.4) / (0.6 + 0.4) = 74
        assert result.prediction == 74.0
        assert result.metadata["model_type"] == "ensemble"

    @patch("models.core.EnsembleStockPredictor.data_provider")
    def test_predict_model_failure(self, mock_data_provider, sample_stock_data):
        """Test predict when some models fail."""
        ensemble = EnsembleStockPredictor()
        ensemble._is_trained = True

        model1 = Mock()
        model1.predict.side_effect = Exception("Model failed")

        model2 = Mock()
        model2.predict.return_value = PredictionResult(
            prediction=80.0, confidence=0.9, timestamp=datetime.now(), metadata={}
        )

        ensemble.add_model(model1, 0.6)
        ensemble.add_model(model2, 0.4)

        mock_data_provider.get_stock_data.return_value = sample_stock_data

        result = ensemble.predict("AAPL")

        assert isinstance(result, PredictionResult)
        # Only model2 should contribute
        assert result.prediction == 80.0

    @patch("models.core.EnsembleStockPredictor.data_provider")
    def test_predict_all_models_fail(self, mock_data_provider, sample_stock_data):
        """Test predict when all models fail."""
        ensemble = EnsembleStockPredictor()
        ensemble._is_trained = True

        model1 = Mock()
        model1.predict.side_effect = Exception("Model failed")

        ensemble.add_model(model1, 1.0)

        mock_data_provider.get_stock_data.return_value = sample_stock_data

        with pytest.raises(ValueError, match="All models failed"):
            ensemble.predict("AAPL")
