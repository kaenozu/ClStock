import pytest
import pandas as pd
from unittest.mock import patch, Mock
from models.predictor import StockPredictor
from models.recommendation import StockRecommendation


class TestStockPredictor:
    """StockPredictorのテストクラス"""

    def test_init(self):
        """初期化のテスト"""
        predictor = StockPredictor()
        assert predictor.data_provider is not None
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.is_trained is False

    @pytest.mark.unit
    def test_prepare_features(self, mock_stock_data):
        """特徴量準備のテスト"""
        predictor = StockPredictor()

        # 技術指標を含むテストデータを作成
        test_data = mock_stock_data.copy()
        test_data["SMA_20"] = test_data["Close"].rolling(20).mean()
        test_data["SMA_50"] = test_data["Close"].rolling(50).mean()
        test_data["RSI"] = 50.0  # 固定値
        test_data["MACD"] = 1.0  # 固定値
        test_data["ATR"] = 2.0  # 固定値

        features = predictor.prepare_features(test_data)

        expected_columns = [
            "price_change",
            "volume_change",
            "high_low_ratio",
            "close_open_ratio",
            "sma_20_ratio",
            "sma_50_ratio",
            "rsi",
            "macd",
            "atr_ratio",
            "price_lag_1",
            "price_lag_5",
            "volume_lag_1",
        ]

        for col in expected_columns:
            assert col in features.columns

    @pytest.mark.unit
    def test_prepare_features_empty_data(self):
        """空データでの特徴量準備テスト"""
        predictor = StockPredictor()
        empty_data = pd.DataFrame()

        features = predictor.prepare_features(empty_data)
        assert features.empty

    @pytest.mark.unit
    def test_calculate_score_with_mock(self, mock_stock_data):
        """スコア計算のテスト（モック使用）"""
        predictor = StockPredictor()

        # データプロバイダーをモック
        with patch.object(
            predictor.data_provider, "get_stock_data", return_value=mock_stock_data
        ), patch.object(
            predictor.data_provider, "calculate_technical_indicators"
        ) as mock_calc:

            # 技術指標付きデータを返すようにモック設定
            enriched_data = mock_stock_data.copy()
            enriched_data["SMA_20"] = enriched_data["Close"] * 0.98
            enriched_data["SMA_50"] = enriched_data["Close"] * 0.95
            enriched_data["RSI"] = 55.0
            mock_calc.return_value = enriched_data

            score = predictor.calculate_score("7203")

            assert 0 <= score <= 100
            assert isinstance(score, float)

    @pytest.mark.unit
    def test_calculate_score_empty_data(self):
        """空データでのスコア計算テスト"""
        predictor = StockPredictor()

        with patch.object(
            predictor.data_provider, "get_stock_data", return_value=pd.DataFrame()
        ):
            score = predictor.calculate_score("INVALID")
            assert score == 0

    @pytest.mark.unit
    def test_generate_recommendation_with_mock(self, mock_stock_data):
        """推奨生成のテスト（モック使用）"""
        predictor = StockPredictor()

        # 技術指標付きデータを準備
        enriched_data = mock_stock_data.copy()
        enriched_data["SMA_20"] = enriched_data["Close"] * 0.98
        enriched_data["SMA_50"] = enriched_data["Close"] * 0.95
        enriched_data["RSI"] = 55.0

        with patch.object(
            predictor.data_provider, "get_stock_data", return_value=mock_stock_data
        ), patch.object(
            predictor.data_provider,
            "calculate_technical_indicators",
            return_value=enriched_data,
        ), patch.object(
            predictor.data_provider, "get_financial_metrics", return_value={}
        ), patch.object(
            predictor, "calculate_score", return_value=85.0
        ):

            recommendation = predictor.generate_recommendation("7203")

            assert isinstance(recommendation, StockRecommendation)
            assert recommendation.symbol == "7203"
            assert recommendation.company_name == "トヨタ自動車"
            assert recommendation.score == 85.0
            assert recommendation.target_price > 0
            assert recommendation.stop_loss > 0
            assert recommendation.profit_target_1 > 0
            assert recommendation.profit_target_2 > 0
            assert recommendation.current_price > 0

    @pytest.mark.unit
    def test_generate_recommendation_different_scores(self, mock_stock_data):
        """異なるスコアでの推奨生成テスト"""
        predictor = StockPredictor()

        enriched_data = mock_stock_data.copy()
        enriched_data["SMA_20"] = enriched_data["Close"]
        enriched_data["RSI"] = 50.0

        test_scores = [90, 70, 40]

        for score in test_scores:
            with patch.object(
                predictor.data_provider, "get_stock_data", return_value=mock_stock_data
            ), patch.object(
                predictor.data_provider,
                "calculate_technical_indicators",
                return_value=enriched_data,
            ), patch.object(
                predictor.data_provider, "get_financial_metrics", return_value={}
            ), patch.object(
                predictor, "calculate_score", return_value=score
            ):

                recommendation = predictor.generate_recommendation("7203")

                if score >= 80:
                    assert "1～2か月" in recommendation.holding_period
                elif score >= 60:
                    assert "2～3か月" in recommendation.holding_period
                else:
                    assert "3～4か月" in recommendation.holding_period

    @pytest.mark.unit
    def test_get_top_recommendations_with_mock(self, mock_stock_data):
        """上位推奨取得のテスト（モック使用）"""
        predictor = StockPredictor()

        enriched_data = mock_stock_data.copy()
        enriched_data["SMA_20"] = enriched_data["Close"]
        enriched_data["RSI"] = 50.0

        with patch.object(
            predictor.data_provider,
            "get_all_stock_symbols",
            return_value=["7203", "6758", "9984"],
        ), patch.object(predictor, "generate_recommendation") as mock_generate:

            # 異なるスコアの推奨を返すようにモック設定
            mock_recommendations = []
            for i, symbol in enumerate(["7203", "6758", "9984"]):
                rec = StockRecommendation(
                    rank=0,
                    symbol=symbol,
                    company_name=f"Company {symbol}",
                    buy_timing="Test timing",
                    target_price=100.0,
                    stop_loss=90.0,
                    profit_target_1=105.0,
                    profit_target_2=110.0,
                    holding_period="1～2か月",
                    score=90 - i * 10,  # 90, 80, 70
                    current_price=95.0,
                    recommendation_reason="Test reason",
                )
                mock_recommendations.append(rec)

            mock_generate.side_effect = mock_recommendations

            top_recommendations = predictor.get_top_recommendations(2)

            assert len(top_recommendations) == 2
            assert top_recommendations[0].rank == 1
            assert top_recommendations[1].rank == 2
            # スコアが高い順にソートされているか確認
            assert top_recommendations[0].score >= top_recommendations[1].score

    @pytest.mark.unit
    def test_get_top_recommendations_with_errors(self):
        """エラーハンドリングを含む上位推奨取得テスト"""
        predictor = StockPredictor()

        with patch.object(
            predictor.data_provider,
            "get_all_stock_symbols",
            return_value=["7203", "INVALID", "6758"],
        ), patch.object(predictor, "generate_recommendation") as mock_generate:

            # 一部の銘柄でエラーが発生する設定
            def side_effect(symbol):
                if symbol == "INVALID":
                    raise Exception("Invalid symbol")
                return StockRecommendation(
                    rank=0,
                    symbol=symbol,
                    company_name=f"Company {symbol}",
                    buy_timing="Test timing",
                    target_price=100.0,
                    stop_loss=90.0,
                    profit_target_1=105.0,
                    profit_target_2=110.0,
                    holding_period="1～2か月",
                    score=85.0,
                    current_price=95.0,
                    recommendation_reason="Test reason",
                )

            mock_generate.side_effect = side_effect

            top_recommendations = predictor.get_top_recommendations(3)

            # エラーが発生した銘柄は除外されるため、2件のみ返される
            assert len(top_recommendations) == 2

    @pytest.mark.unit
    def test_score_calculation_logic(self, mock_stock_data):
        """スコア計算ロジックの詳細テスト"""
        predictor = StockPredictor()

        # 特定の条件でのスコア計算をテスト
        test_data = mock_stock_data.copy()
        current_price = 100.0
        test_data.loc[test_data.index[-1], "Close"] = current_price

        # 良好な条件でのテスト
        test_data["SMA_20"] = current_price * 0.98  # 現在価格 > SMA20
        test_data["SMA_50"] = current_price * 0.95  # SMA20 > SMA50
        test_data["RSI"] = 50.0  # 適正範囲
        test_data["Volume"] = [1000000] * len(test_data)

        with patch.object(
            predictor.data_provider, "get_stock_data", return_value=test_data
        ), patch.object(
            predictor.data_provider,
            "calculate_technical_indicators",
            return_value=test_data,
        ):

            score = predictor.calculate_score("7203")

            # 良好な条件なので、ベーススコア50より高くなるはず
            assert score > 50
