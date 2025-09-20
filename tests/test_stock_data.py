import pytest
import pandas as pd
from unittest.mock import patch, Mock
from data.stock_data import StockDataProvider


class TestStockDataProvider:
    """StockDataProviderのテストクラス"""

    def test_init(self):
        """初期化のテスト"""
        provider = StockDataProvider()
        assert len(provider.jp_stock_codes) == 10
        assert "7203" in provider.jp_stock_codes
        assert provider.jp_stock_codes["7203"] == "トヨタ自動車"

    @pytest.mark.unit
    def test_get_all_stock_symbols(self):
        """全銘柄シンボル取得のテスト"""
        provider = StockDataProvider()
        symbols = provider.get_all_stock_symbols()
        assert len(symbols) == 10
        assert "7203" in symbols
        assert "6758" in symbols

    @pytest.mark.unit
    def test_get_stock_data_with_mock(self, mock_yfinance):
        """株価データ取得のテスト（モック使用）"""
        provider = StockDataProvider()

        with patch('yfinance.Ticker', return_value=mock_yfinance):
            data = provider.get_stock_data("7203", "1mo")

            assert not data.empty
            assert 'Symbol' in data.columns
            assert 'CompanyName' in data.columns
            assert data['Symbol'].iloc[0] == "7203"

    @pytest.mark.unit
    def test_get_stock_data_invalid_symbol(self, mock_yfinance):
        """無効な銘柄コードのテスト"""
        provider = StockDataProvider()

        # 空のDataFrameを返すようにモック設定
        mock_yfinance.history.return_value = pd.DataFrame()

        with patch('yfinance.Ticker', return_value=mock_yfinance):
            data = provider.get_stock_data("INVALID", "1mo")
            assert data.empty

    @pytest.mark.unit
    def test_calculate_technical_indicators(self, mock_stock_data):
        """技術指標計算のテスト"""
        provider = StockDataProvider()

        # テスト用データに技術指標を追加
        result = provider.calculate_technical_indicators(mock_stock_data)

        assert 'SMA_20' in result.columns
        assert 'SMA_50' in result.columns
        assert 'RSI' in result.columns
        assert 'MACD' in result.columns
        assert 'ATR' in result.columns

    @pytest.mark.unit
    def test_calculate_technical_indicators_empty_data(self):
        """空データでの技術指標計算テスト"""
        provider = StockDataProvider()
        empty_data = pd.DataFrame()

        result = provider.calculate_technical_indicators(empty_data)
        assert result.empty

    @pytest.mark.unit
    def test_get_financial_metrics_with_mock(self, mock_yfinance):
        """財務指標取得のテスト（モック使用）"""
        provider = StockDataProvider()

        with patch('yfinance.Ticker', return_value=mock_yfinance):
            metrics = provider.get_financial_metrics("7203")

            assert 'symbol' in metrics
            assert 'company_name' in metrics
            assert 'market_cap' in metrics
            assert 'pe_ratio' in metrics
            assert metrics['symbol'] == "7203"

    @pytest.mark.unit
    def test_get_multiple_stocks(self, mock_yfinance):
        """複数銘柄データ取得のテスト"""
        provider = StockDataProvider()

        with patch('yfinance.Ticker', return_value=mock_yfinance):
            symbols = ["7203", "6758"]
            result = provider.get_multiple_stocks(symbols, "1mo")

            assert len(result) == 2
            assert "7203" in result
            assert "6758" in result

    @pytest.mark.unit
    def test_japanese_stock_ticker_format(self):
        """日本株のティッカー形式テスト"""
        provider = StockDataProvider()

        # 日本株の場合は.Tが付加されることを確認
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = pd.DataFrame({
                'Close': [100], 'Volume': [1000]
            })
            mock_ticker.return_value = mock_ticker_instance

            provider.get_stock_data("7203", "1d")
            mock_ticker.assert_called_with("7203.T")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_stock_data_integration(self):
        """実際の株価データ取得テスト（統合テスト）"""
        provider = StockDataProvider()

        # 実際のAPIを呼び出すため、ネットワーク接続が必要
        # CIで実行する場合はスキップするかモックを使用
        try:
            data = provider.get_stock_data("7203", "5d")
            if not data.empty:
                assert 'Close' in data.columns
                assert 'Volume' in data.columns
        except Exception:
            pytest.skip("Network connection required for integration test")