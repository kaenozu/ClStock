import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from models.precision.precision_87_system import Precision87BreakthroughSystem


def test_predict_with_87_precision():
    """Test predict_with_87_precision method."""
    predictor = Precision87BreakthroughSystem()

    # Mock StockDataProvider
    with patch('models.precision.precision_87_system.StockDataProvider') as mock_provider:
        mock_data = pd.DataFrame({
            'Close': [100, 102, 101, 103, 104],
            'Open': [99, 101, 100, 102, 103],
            'High': [101, 103, 102, 104, 105],
            'Low': [98, 100, 99, 101, 102],
            'Volume': [1000, 1100, 1050, 1150, 1200]
        })
        mock_provider.return_value.get_stock_data.return_value = mock_data
        mock_provider.return_value.calculate_technical_indicators.return_value = mock_data

        # Mock meta_learner
        with patch.object(predictor, 'meta_learner', MagicMock()) as mock_meta:
            mock_meta.create_symbol_profile.return_value = {'current_price': 104, 'trend_persistence': 0.5}
            mock_meta.adapt_model_parameters.return_value = {'adapted_prediction': 52, 'adapted_confidence': 0.8}

            # Mock dqn_agent
            with patch.object(predictor, 'dqn_agent', MagicMock()) as mock_dqn:
                mock_dqn.get_trading_signal.return_value = {'signal_strength': 0.1, 'confidence': 0.7}

                result = predictor.predict_with_87_precision("TEST.T")

                assert 'final_prediction' in result
                assert 'final_confidence' in result
                assert 'final_accuracy' in result

def test_predict():
    """Test predict method."""
    predictor = Precision87BreakthroughSystem()

    # Mock predict_with_87_precision
    with patch.object(predictor, 'predict_with_87_precision') as mock_predict_87:
        mock_predict_87.return_value = {
            'final_prediction': 105.0,
            'final_confidence': 0.8,
            'final_accuracy': 87.0,
            'precision_87_achieved': True
        }

        result = predictor.predict("TEST.T")

        assert result.prediction == 105.0
        assert result.confidence == 0.8
        assert result.accuracy == 87.0

def test_get_confidence():
    """Test get_confidence method."""
    predictor = Precision87BreakthroughSystem()

    # Mock predict_with_87_precision
    with patch.object(predictor, 'predict_with_87_precision') as mock_predict_87:
        mock_predict_87.return_value = {
            'final_prediction': 105.0,
            'final_confidence': 0.8,
            'final_accuracy': 87.0,
            'precision_87_achieved': True
        }

        result = predictor.get_confidence("TEST.T")

        assert result == 0.8

if __name__ == "__main__":
    pytest.main()