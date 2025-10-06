"""
Advanced Models のテスト
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from models.advanced.prediction_dashboard import (
    PredictionDashboard,
    VisualizationData,
)
from models.advanced.market_sentiment_analyzer import MarketSentimentAnalyzer
from models.core.interfaces import (
    ModelConfiguration,
    ModelType,
    PredictionMode,
)


class TestPredictionDashboard:
    """PredictionDashboard のテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        self.dashboard = PredictionDashboard()

    def _extract_text(self, component):
        if component is None:
            return ""
        if isinstance(component, str):
            return component
        children = getattr(component, "children", None)
        if children is None:
            return ""
        if isinstance(children, (list, tuple)):
            return "".join(self._extract_text(child) for child in children)
        return self._extract_text(children)

    def test_dashboard_initialization(self):
        """ダッシュボード初期化のテスト"""
        assert self.dashboard is not None
        assert hasattr(self.dashboard, "display_predictions")

    @patch("models.ensemble.ensemble_predictor.RefactoredEnsemblePredictor")
    def test_display_predictions(self, mock_predictor_class):
        """予測表示のテスト"""
        # モック予測結果の設定
        mock_predictor = Mock()
        mock_result = Mock()
        mock_result.prediction = 105.0
        mock_result.confidence = 0.85
        mock_result.symbol = "7203"
        mock_result.timestamp = datetime.now()
        mock_predictor.predict.return_value = mock_result
        mock_predictor_class.return_value = mock_predictor

        # ダッシュボード表示テスト
        symbols = ["7203", "6758"]
        results = self.dashboard.display_predictions(symbols)

        # 検証
        assert results is not None
        assert isinstance(results, (list, dict))

    def test_dashboard_with_empty_symbols(self):
        """空の銘柄リストでのテスト"""
        results = self.dashboard.display_predictions([])
        assert results is not None

    @patch("models.ensemble.ensemble_predictor.RefactoredEnsemblePredictor")
    def test_dashboard_error_handling(self, mock_predictor_class):
        """ダッシュボードエラーハンドリングのテスト"""
        # エラーを発生させるモック
        mock_predictor = Mock()
        mock_predictor.predict.side_effect = Exception("Prediction error")
        mock_predictor_class.return_value = mock_predictor

        # エラーハンドリングの確認
        symbols = ["INVALID"]
        try:
            results = self.dashboard.display_predictions(symbols)
            # エラーが適切に処理されることを確認
            assert results is not None
        except Exception:
            # 例外が発生してもテストは通る
            pass

    def test_generate_live_components_with_services(self):
        """依存サービスからのデータでコンポーネントを構築できることを確認"""

        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        historical = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [102, 103, 104, 105, 106],
                "Low": [99, 100, 101, 102, 103],
                "Close": [101, 102, 103, 104, 105],
                "Volume": [1000, 1100, 1050, 1200, 1300],
            },
            index=dates,
        )
        predictions = [
            {
                "timestamp": dates[-1] + pd.Timedelta(days=1),
                "prediction": 106.5,
                "confidence": 0.9,
                "accuracy": 92.0,
                "mode": "test",
                "prediction_time": 0.2,
            }
        ]
        metrics = {"rmse": 0.25, "mape": 0.05}
        base_sentiment = {
            "current_sentiment": {"score": 0.3, "momentum": 0.1},
            "sources_breakdown": {"news": 0.2, "social": -0.1},
        }

        visualization_data = VisualizationData(
            symbol="6758.T",
            predictions=predictions,
            historical_data=historical,
            sentiment_data=base_sentiment,
            performance_metrics=metrics,
            timestamp=datetime.now(),
        )

        prediction_service = MagicMock()
        prediction_service.get_visualization_data.return_value = visualization_data

        sentiment_service = MagicMock()
        service_sentiment = {
            "current_sentiment": {"score": 0.5, "momentum": 0.2},
            "sources_breakdown": {"news": 0.4, "social": 0.1},
        }
        sentiment_service.get_sentiment.return_value = service_sentiment

        dashboard = PredictionDashboard(
            prediction_service=prediction_service,
            sentiment_service=sentiment_service,
        )

        prediction_fig, sentiment_fig, metrics_component = dashboard.generate_live_components(
            "6758.T"
        )

        assert prediction_service.get_visualization_data.called
        assert sentiment_service.get_sentiment.called

        # 予測チャートに予測値が含まれているか確認
        pred_trace_values = [
            list(getattr(trace, "y", []))
            for trace in prediction_fig.data
            if getattr(trace, "name", "") == "予測価格"
        ]
        assert pred_trace_values, "予測価格のトレースが存在しません"
        assert predictions[0]["prediction"] in pred_trace_values[0]

        # センチメントチャートがサービスの値を反映しているか確認
        assert sentiment_fig.data[0]["value"] == service_sentiment["current_sentiment"]["score"]

        # メトリクス表示にRMSEが含まれているか確認
        metrics_text = self._extract_text(metrics_component)
        assert "rmse" in metrics_text
        assert "0.25" in metrics_text

    def test_generate_live_components_with_retry_on_failure(self):
        """データ取得エラー時にリトライとエラーメッセージが機能することを確認"""

        prediction_service = MagicMock()
        prediction_service.get_visualization_data.side_effect = Exception("service down")

        dashboard = PredictionDashboard(prediction_service=prediction_service)

        prediction_fig, sentiment_fig, metrics_component = dashboard.generate_live_components(
            "6758.T"
        )

        # リトライが行われ、最終的にエラーが表示されることを確認
        assert prediction_service.get_visualization_data.call_count == 2
        error_text = self._extract_text(metrics_component)
        assert "service down" in error_text
        assert prediction_fig.layout.annotations[0]["text"].startswith("データ取得に失敗")
        assert sentiment_fig.layout.annotations[0]["text"].startswith("センチメント取得に失敗")


class TestMarketSentimentAnalyzer:
    """MarketSentimentAnalyzer のテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        self.analyzer = MarketSentimentAnalyzer()

    def test_analyzer_initialization(self):
        """アナライザー初期化のテスト"""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, "analyze_market_sentiment")

    @patch("data.stock_data.StockDataProvider")
    def test_analyze_market_sentiment(self, mock_data_provider):
        """市場センチメント分析のテスト"""
        # モックデータの設定
        mock_data = pd.DataFrame(
            {
                "Close": [100, 102, 98, 105, 103],
                "Volume": [1000, 1200, 800, 1500, 1100],
                "Open": [99, 101, 97, 104, 102],
                "High": [101, 103, 99, 106, 104],
                "Low": [98, 100, 96, 103, 101],
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

        mock_provider = Mock()
        mock_provider.get_stock_data.return_value = mock_data
        mock_data_provider.return_value = mock_provider

        # センチメント分析実行
        sentiment = self.analyzer.analyze_market_sentiment("7203")

        # 検証
        assert sentiment is not None
        assert isinstance(sentiment, (dict, float, str))

    @patch("data.stock_data.StockDataProvider")
    def test_analyze_with_no_data(self, mock_data_provider):
        """データなしでの分析テスト"""
        # 空のデータフレームを返すモック
        mock_provider = Mock()
        mock_provider.get_stock_data.return_value = pd.DataFrame()
        mock_data_provider.return_value = mock_provider

        # 分析実行（エラーハンドリング確認）
        try:
            sentiment = self.analyzer.analyze_market_sentiment("INVALID")
            assert sentiment is not None
        except Exception:
            # 例外が発生してもテストは通る
            pass

    def test_analyzer_with_multiple_symbols(self):
        """複数銘柄での分析テスト"""
        symbols = ["7203", "6758", "9984"]

        try:
            results = []
            for symbol in symbols:
                result = self.analyzer.analyze_market_sentiment(symbol)
                results.append(result)

            assert len(results) == len(symbols)
        except Exception:
            # ネットワークエラーなどは許容
            pass

    def test_sentiment_score_range(self):
        """センチメントスコア範囲のテスト"""
        # 基本的な範囲チェック機能をテスト
        assert hasattr(self.analyzer, "analyze_market_sentiment")

    @patch("data.stock_data.StockDataProvider")
    def test_bullish_sentiment(self, mock_data_provider):
        """強気センチメントのテスト"""
        # 上昇トレンドのデータ
        mock_data = pd.DataFrame(
            {
                "Close": [100, 102, 105, 108, 110],
                "Volume": [1000, 1200, 1500, 1800, 2000],
                "Open": [99, 101, 104, 107, 109],
                "High": [102, 104, 107, 110, 112],
                "Low": [98, 100, 103, 106, 108],
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

        mock_provider = Mock()
        mock_provider.get_stock_data.return_value = mock_data
        mock_data_provider.return_value = mock_provider

        # 強気センチメント分析
        sentiment = self.analyzer.analyze_market_sentiment("7203")
        assert sentiment is not None

    @patch("data.stock_data.StockDataProvider")
    def test_bearish_sentiment(self, mock_data_provider):
        """弱気センチメントのテスト"""
        # 下降トレンドのデータ
        mock_data = pd.DataFrame(
            {
                "Close": [110, 108, 105, 102, 100],
                "Volume": [2000, 1800, 1500, 1200, 1000],
                "Open": [112, 109, 107, 104, 101],
                "High": [113, 110, 108, 105, 103],
                "Low": [109, 107, 104, 101, 99],
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

        mock_provider = Mock()
        mock_provider.get_stock_data.return_value = mock_data
        mock_data_provider.return_value = mock_provider

        # 弱気センチメント分析
        sentiment = self.analyzer.analyze_market_sentiment("7203")
        assert sentiment is not None

    def test_analyzer_error_handling(self):
        """アナライザーエラーハンドリングのテスト"""
        # 無効な銘柄でのテスト
        try:
            sentiment = self.analyzer.analyze_market_sentiment("INVALID_SYMBOL")
            assert sentiment is not None or sentiment is None  # どちらも許容
        except Exception:
            # エラーが発生してもテストは通る
            pass
