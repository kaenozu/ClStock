"""
EnsembleStockPredictor のテスト
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from models_refactored.ensemble.ensemble_predictor import EnsemblePredictor
from models_refactored.core.interfaces import (
    ModelConfiguration,
    ModelType,
    PredictionMode,
)


class TestEnsembleStockPredictor:
    """EnsembleStockPredictor のテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        self.config = ModelConfiguration(
            model_type=ModelType.ENSEMBLE,
            prediction_mode=PredictionMode.BALANCED,
            cache_enabled=False,  # テスト用にキャッシュ無効
        )
        self.predictor = EnsemblePredictor(self.config)

    def test_predictor_initialization(self):
        """予測器初期化のテスト"""
        assert self.predictor.config.model_type == ModelType.ENSEMBLE
        assert self.predictor.config.prediction_mode == PredictionMode.BALANCED
        assert self.predictor.config.cache_enabled is False

    @patch("data.stock_data.StockDataProvider")
    def test_predict_with_valid_symbol(self, mock_data_provider):
        """有効な銘柄での予測テスト"""
        # モックデータの設定
        mock_data = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104],
                "Volume": [1000, 1100, 1200, 1300, 1400],
                "Open": [99, 100, 101, 102, 103],
                "High": [101, 102, 103, 104, 105],
                "Low": [98, 99, 100, 101, 102],
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

        mock_provider = Mock()
        mock_provider.get_stock_data.return_value = mock_data
        mock_data_provider.return_value = mock_provider

        # 予測実行
        result = self.predictor.predict("7203", period="1mo")

        # 検証
        assert result is not None
        assert result.symbol == "7203"
        assert isinstance(result.prediction, (int, float))
        assert 0 <= result.confidence <= 1
        assert result.model_type == ModelType.ENSEMBLE

    @patch("data.stock_data.StockDataProvider")
    def test_predict_with_no_data(self, mock_data_provider):
        """データなしでの予測テスト"""
        # 空のデータフレームを返すモック
        mock_provider = Mock()
        mock_provider.get_stock_data.return_value = pd.DataFrame()
        mock_data_provider.return_value = mock_provider

        # 例外発生を確認
        with pytest.raises(ValueError, match="No data available"):
            self.predictor.predict("INVALID", period="1mo")

    @patch("data.stock_data.StockDataProvider")
    def test_predict_batch(self, mock_data_provider):
        """バッチ予測のテスト"""
        # モックデータの設定
        mock_data = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104],
                "Volume": [1000, 1100, 1200, 1300, 1400],
                "Open": [99, 100, 101, 102, 103],
                "High": [101, 102, 103, 104, 105],
                "Low": [98, 99, 100, 101, 102],
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

        mock_provider = Mock()
        mock_provider.get_stock_data.return_value = mock_data
        mock_data_provider.return_value = mock_provider

        # バッチ予測実行
        symbols = ["7203", "6758", "9984"]
        result = self.predictor.predict_batch(symbols, period="1mo")

        # 検証
        assert result is not None
        assert len(result.predictions) <= len(symbols)
        assert isinstance(result.predictions, dict)
        assert isinstance(result.errors, dict)

    def test_feature_calculation(self):
        """特徴量計算のテスト"""
        # テストデータ作成
        data = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104, 105],
                "Volume": [1000, 1100, 1200, 1300, 1400, 1500],
                "Open": [99, 100, 101, 102, 103, 104],
                "High": [101, 102, 103, 104, 105, 106],
                "Low": [98, 99, 100, 101, 102, 103],
            },
            index=pd.date_range("2023-01-01", periods=6),
        )

        # 特徴量計算
        features = self.predictor._calculate_features(data)

        # 検証
        assert features is not None
        assert len(features) > 0
        assert all(not np.isnan(f) for f in features if isinstance(f, float))

    def test_conservative_mode(self):
        """保守的モードのテスト"""
        config = ModelConfiguration(
            model_type=ModelType.ENSEMBLE, prediction_mode=PredictionMode.CONSERVATIVE
        )
        predictor = EnsemblePredictor(config)

        assert predictor.config.prediction_mode == PredictionMode.CONSERVATIVE

    def test_aggressive_mode(self):
        """積極的モードのテスト"""
        config = ModelConfiguration(
            model_type=ModelType.ENSEMBLE, prediction_mode=PredictionMode.AGGRESSIVE
        )
        predictor = EnsemblePredictor(config)

        assert predictor.config.prediction_mode == PredictionMode.AGGRESSIVE

    @patch("data.stock_data.StockDataProvider")
    def test_error_handling(self, mock_data_provider):
        """エラーハンドリングのテスト"""
        # データ取得でエラーが発生するモック
        mock_provider = Mock()
        mock_provider.get_stock_data.side_effect = Exception("Network error")
        mock_data_provider.return_value = mock_provider

        # エラーハンドリングの確認
        with pytest.raises(Exception):
            self.predictor.predict("7203", period="1mo")

    def test_cache_disabled(self):
        """キャッシュ無効化のテスト"""
        assert self.predictor.config.cache_enabled is False

    def test_parallel_processing_config(self):
        """並列処理設定のテスト"""
        predictor = EnsemblePredictor(config)

        assert predictor.config.parallel_enabled is True
        assert predictor.config.max_workers == 8

    @patch("data.stock_data.StockDataProvider")
    def test_prediction_confidence_range(self, mock_data_provider):
        """予測信頼度範囲のテスト"""
        # 十分なデータを持つモック
        mock_data = pd.DataFrame(
            {
                "Close": np.random.uniform(100, 110, 100),
                "Volume": np.random.uniform(1000, 2000, 100),
                "Open": np.random.uniform(99, 109, 100),
                "High": np.random.uniform(101, 111, 100),
                "Low": np.random.uniform(98, 108, 100),
            },
            index=pd.date_range("2023-01-01", periods=100),
        )

        mock_provider = Mock()
        mock_provider.get_stock_data.return_value = mock_data
        mock_data_provider.return_value = mock_provider

        # 予測実行
        result = self.predictor.predict("7203", period="6mo")

        # 信頼度範囲の確認
        assert 0.0 <= result.confidence <= 1.0
        assert result.prediction > 0  # 株価は正の値
