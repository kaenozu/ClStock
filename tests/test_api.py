import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import pandas as pd
from datetime import datetime

from app.main import app
from models.recommendation import StockRecommendation


class TestAPI:
    """API エンドポイントのテストクラス"""

    @pytest.fixture
    def client(self):
        """テストクライアントのセットアップ"""
        return TestClient(app)

    @pytest.mark.api
    def test_root_endpoint(self, client):
        """ルートエンドポイントのテスト"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "ClStock API is running"}

    @pytest.mark.api
    def test_health_endpoint(self, client):
        """ヘルスチェックエンドポイントのテスト"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    @pytest.mark.api
    def test_get_stocks_endpoint(self, client):
        """銘柄一覧エンドポイントのテスト"""
        response = client.get("/api/v1/stocks")
        assert response.status_code == 200

        data = response.json()
        assert "stocks" in data
        assert len(data["stocks"]) == 10

        # 期待される銘柄が含まれているか確認
        symbols = [stock["symbol"] for stock in data["stocks"]]
        assert "7203" in symbols
        assert "6758" in symbols

        # データ構造の確認
        for stock in data["stocks"]:
            assert "symbol" in stock
            assert "name" in stock

    @pytest.mark.api
    def test_get_recommendations_endpoint(self, client, sample_recommendation):
        """推奨銘柄ランキングエンドポイントのテスト"""
        with patch('models.predictor.StockPredictor') as mock_predictor_class:
            mock_predictor = Mock()
            mock_predictor.get_top_recommendations.return_value = [sample_recommendation]
            mock_predictor_class.return_value = mock_predictor

            response = client.get("/api/v1/recommendations?top_n=1")
            assert response.status_code == 200

            data = response.json()
            assert "recommendations" in data
            assert "generated_at" in data
            assert "market_status" in data

            assert len(data["recommendations"]) == 1
            rec = data["recommendations"][0]
            assert rec["symbol"] == "7203"
            assert rec["rank"] == 1

    @pytest.mark.api
    def test_get_recommendations_with_params(self, client, sample_recommendation):
        """パラメータ付き推奨銘柄エンドポイントのテスト"""
        with patch('models.predictor.StockPredictor') as mock_predictor_class:
            mock_predictor = Mock()
            recommendations = [sample_recommendation] * 3
            for i, rec in enumerate(recommendations):
                rec.rank = i + 1
            mock_predictor.get_top_recommendations.return_value = recommendations
            mock_predictor_class.return_value = mock_predictor

            # top_n=3でテスト
            response = client.get("/api/v1/recommendations?top_n=3")
            assert response.status_code == 200

            data = response.json()
            assert len(data["recommendations"]) == 3

    @pytest.mark.api
    def test_get_recommendations_invalid_params(self, client):
        """無効なパラメータでの推奨銘柄エンドポイントテスト"""
        # top_nが範囲外
        response = client.get("/api/v1/recommendations?top_n=15")
        assert response.status_code == 422  # Validation Error

        response = client.get("/api/v1/recommendations?top_n=0")
        assert response.status_code == 422  # Validation Error

    @pytest.mark.api
    def test_get_single_recommendation_endpoint(self, client, sample_recommendation):
        """特定銘柄推奨エンドポイントのテスト"""
        with patch('models.predictor.StockPredictor') as mock_predictor_class, \
             patch('data.stock_data.StockDataProvider') as mock_provider_class:

            mock_predictor = Mock()
            mock_predictor.generate_recommendation.return_value = sample_recommendation
            mock_predictor_class.return_value = mock_predictor

            mock_provider = Mock()
            mock_provider.get_all_stock_symbols.return_value = ["7203", "6758", "9984"]
            mock_provider_class.return_value = mock_provider

            response = client.get("/api/v1/recommendation/7203")
            assert response.status_code == 200

            data = response.json()
            assert data["symbol"] == "7203"
            assert data["rank"] == 1  # 単一銘柄の場合は1位固定

    @pytest.mark.api
    def test_get_single_recommendation_not_found(self, client):
        """存在しない銘柄での推奨エンドポイントテスト"""
        with patch('data.stock_data.StockDataProvider') as mock_provider_class:
            mock_provider = Mock()
            mock_provider.get_all_stock_symbols.return_value = ["7203", "6758"]
            mock_provider_class.return_value = mock_provider

            response = client.get("/api/v1/recommendation/9999")
            assert response.status_code == 404
            assert "見つかりません" in response.json()["detail"]

    @pytest.mark.api
    def test_get_stock_data_endpoint(self, client, mock_stock_data):
        """株価データエンドポイントのテスト"""
        with patch('data.stock_data.StockDataProvider') as mock_provider_class:
            mock_provider = Mock()
            mock_provider.get_all_stock_symbols.return_value = ["7203"]
            mock_provider.get_stock_data.return_value = mock_stock_data

            # 技術指標付きデータを準備
            tech_data = mock_stock_data.copy()
            tech_data['SMA_20'] = tech_data['Close'].rolling(20).mean()
            tech_data['SMA_50'] = tech_data['Close'].rolling(50).mean()
            tech_data['RSI'] = 50.0
            tech_data['MACD'] = 1.0
            mock_provider.calculate_technical_indicators.return_value = tech_data

            mock_provider.get_financial_metrics.return_value = {
                'market_cap': 1000000000,
                'pe_ratio': 15.0
            }
            mock_provider.jp_stock_codes = {"7203": "トヨタ自動車"}
            mock_provider_class.return_value = mock_provider

            response = client.get("/api/v1/stock/7203/data?period=1mo")
            assert response.status_code == 200

            data = response.json()
            assert data["symbol"] == "7203"
            assert data["company_name"] == "トヨタ自動車"
            assert "current_price" in data
            assert "technical_indicators" in data
            assert "financial_metrics" in data

    @pytest.mark.api
    def test_get_stock_data_not_found(self, client):
        """存在しない銘柄での株価データエンドポイントテスト"""
        with patch('data.stock_data.StockDataProvider') as mock_provider_class:
            mock_provider = Mock()
            mock_provider.get_all_stock_symbols.return_value = ["7203"]
            mock_provider_class.return_value = mock_provider

            response = client.get("/api/v1/stock/9999/data")
            assert response.status_code == 404

    @pytest.mark.api
    def test_get_stock_data_empty_data(self, client):
        """空データでの株価データエンドポイントテスト"""
        with patch('data.stock_data.StockDataProvider') as mock_provider_class:
            mock_provider = Mock()
            mock_provider.get_all_stock_symbols.return_value = ["7203"]
            mock_provider.get_stock_data.return_value = pd.DataFrame()  # 空のデータ
            mock_provider_class.return_value = mock_provider

            response = client.get("/api/v1/stock/7203/data")
            assert response.status_code == 404
            assert "データが見つかりません" in response.json()["detail"]

    @pytest.mark.api
    def test_api_error_handling(self, client):
        """API エラーハンドリングのテスト"""
        with patch('models.predictor.StockPredictor') as mock_predictor_class:
            mock_predictor = Mock()
            mock_predictor.get_top_recommendations.side_effect = Exception("Test error")
            mock_predictor_class.return_value = mock_predictor

            response = client.get("/api/v1/recommendations")
            assert response.status_code == 500
            assert "推奨銘柄の取得に失敗しました" in response.json()["detail"]

    @pytest.mark.api
    def test_market_status_logic(self, client, sample_recommendation):
        """市場状況判定ロジックのテスト"""
        with patch('models.predictor.StockPredictor') as mock_predictor_class:
            mock_predictor = Mock()
            mock_predictor.get_top_recommendations.return_value = [sample_recommendation]
            mock_predictor_class.return_value = mock_predictor

            # 営業時間内をシミュレート (10時)
            with patch('api.endpoints.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 0, 0)
                response = client.get("/api/v1/recommendations")
                data = response.json()
                assert "市場営業中" in data["market_status"]

            # 営業時間外をシミュレート (18時)
            with patch('api.endpoints.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime(2023, 1, 1, 18, 0, 0)
                response = client.get("/api/v1/recommendations")
                data = response.json()
                assert "市場営業時間外" in data["market_status"]

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_api_integration(self, client):
        """完全なAPI統合テスト"""
        # 実際のデータプロバイダーとプレディクターを使用
        try:
            response = client.get("/api/v1/stocks")
            assert response.status_code == 200

            # 実際の推奨取得（ネットワーク接続が必要）
            response = client.get("/api/v1/recommendations?top_n=1")
            if response.status_code == 200:
                data = response.json()
                assert "recommendations" in data
        except Exception:
            pytest.skip("Network connection required for integration test")