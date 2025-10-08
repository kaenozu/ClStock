"""Lightweight behavioural checks for RefactoredEnsemblePredictor.

These tests ensure the modern ensemble predictor is importable and exposes
basic introspection helpers required by higher level integration tests.
"""

from models.core.interfaces import ModelConfiguration, ModelType
from models.ensemble.ensemble_predictor import RefactoredEnsemblePredictor


def test_predictor_is_instantiable_without_arguments():
    predictor = RefactoredEnsemblePredictor()

    assert predictor.config.model_type == ModelType.ENSEMBLE
    assert predictor.get_prediction_period() == "1y"


def test_predictor_respects_explicit_configuration():
    config = ModelConfiguration(model_type=ModelType.ENSEMBLE, cache_enabled=False)

    predictor = RefactoredEnsemblePredictor(config=config)

    assert predictor.config is config
    assert predictor.config.cache_enabled is False


def test_predictor_predict_method():
    """Test RefactoredEnsemblePredictor's predict method."""
    from unittest.mock import patch, MagicMock
    import pandas as pd

    predictor = RefactoredEnsemblePredictor()

    # Mock StockDataProvider
    with patch('models.ensemble.ensemble_predictor.StockDataProvider') as mock_provider:
        mock_data = pd.DataFrame({
            'Close': [100, 102, 101, 103, 104],
            'Open': [99, 101, 100, 102, 103],
            'High': [101, 103, 102, 104, 105],
            'Low': [98, 100, 99, 101, 102],
            'Volume': [1000, 1100, 1050, 1150, 1200]
        })
        mock_provider.return_value.get_stock_data.return_value = mock_data

        # Mock internal methods if necessary
        with patch.object(predictor, '_get_data_provider', return_value=mock_provider.return_value):
            result = predictor.predict("TEST.T")

            # Check if result is an instance of PredictionResult (or similar expected type)
            # Assuming PredictionResult has attributes like prediction, confidence, accuracy
            assert hasattr(result, 'prediction')
            assert hasattr(result, 'confidence')
            assert hasattr(result, 'accuracy')


def test_predictor_get_confidence_method():
    """Test RefactoredEnsemblePredictor's get_confidence method."""
    from unittest.mock import patch, MagicMock
    import pandas as pd

    predictor = RefactoredEnsemblePredictor()

    # Mock the internal methods that affect confidence calculation
    # For example, if models are added, mock their get_confidence methods
    mock_model = MagicMock()
    mock_model.get_confidence.return_value = 0.7
    predictor.add_model('mock_model', mock_model, 1.0)

    result = predictor.get_confidence("TEST.T")

    # Confidence might be an aggregate, so just check if it's a float between 0 and 1
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
