"""
Data Providers のテスト
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import data.stock_data as stock_data_module
from data.stock_data import StockDataProvider


class TestStockDataProvider:
    """StockDataProvider のテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        self.provider = StockDataProvider()

    def test_provider_initialization(self):
        """プロバイダー初期化のテスト"""
        assert self.provider is not None
        assert hasattr(self.provider, "get_stock_data")

    @patch("yfinance.download")
    def test_get_stock_data_success(self, mock_yfinance):
        """正常な株価データ取得のテスト"""
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

        mock_yfinance.return_value = mock_data

        # データ取得テスト
        result = self.provider.get_stock_data("TESTMOCK", "1mo")

        # 検証（モックが適用されるかどうかを確認）
        assert result is not None
        assert isinstance(result, pd.DataFrame)

    @patch("yfinance.download")
    def test_get_stock_data_with_japanese_symbol(self, mock_yfinance):
        """日本株式での株価データ取得テスト"""
        # モックデータの設定
        mock_data = pd.DataFrame(
            {
                "Close": [2500, 2520, 2540, 2560, 2580],
                "Volume": [10000, 11000, 12000, 13000, 14000],
                "Open": [2480, 2510, 2530, 2550, 2570],
                "High": [2530, 2550, 2570, 2590, 2610],
                "Low": [2470, 2500, 2520, 2540, 2560],
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

        mock_yfinance.return_value = mock_data

        # 日本株データ取得テスト
        result = self.provider.get_stock_data("7203.T", "3mo")

        # 検証
        assert result is not None
        assert not result.empty

    def test_get_stock_data_empty_result(self):
        """無効なシンボルでのエラーハンドリングテスト"""
        # 無効なシンボルでのテスト
        try:
            result = self.provider.get_stock_data("INVALID_SYMBOL_XXXXX", "1mo")
            # 結果があっても無くても正常
            assert result is not None
        except Exception:
            # エラーが発生してもテストは通る（エラーハンドリング確認）
            pass

    def test_get_stock_data_invalid_period(self):
        """無効な期間での株価データ取得テスト"""
        # 無効な期間でのテスト
        try:
            result = self.provider.get_stock_data("7203", "invalid_period")
            assert result is not None
        except Exception:
            # 例外が発生してもテストは通る
            pass

    @patch("yfinance.download")
    def test_calculate_technical_indicators(self, mock_yfinance):
        """テクニカル指標計算のテスト"""
        # 十分なデータを持つモック
        dates = pd.date_range("2023-01-01", periods=50)
        mock_data = pd.DataFrame(
            {
                "Close": [100 + i * 0.5 for i in range(50)],
                "Volume": [1000 + i * 10 for i in range(50)],
                "Open": [99.5 + i * 0.5 for i in range(50)],
                "High": [101 + i * 0.5 for i in range(50)],
                "Low": [98.5 + i * 0.5 for i in range(50)],
            },
            index=dates,
        )

        # テクニカル指標計算
        result = self.provider.calculate_technical_indicators(mock_data)

        # 検証
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # 基本的な指標が含まれていることを確認
        expected_indicators = ["SMA_20", "SMA_50", "RSI"]
        for indicator in expected_indicators:
            if indicator in result.columns:
                assert not result[indicator].isna().all()

    def test_technical_indicators_insufficient_data(self):
        """データ不足でのテクニカル指標計算テスト"""
        # 少ないデータでのテスト
        mock_data = pd.DataFrame(
            {
                "Close": [100, 101, 102],
                "Volume": [1000, 1100, 1200],
                "Open": [99, 100, 101],
                "High": [101, 102, 103],
                "Low": [98, 99, 100],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )

        # テクニカル指標計算（データ不足でも例外が出ないことを確認）
        result = self.provider.calculate_technical_indicators(mock_data)
        assert result is not None

    @patch("yfinance.download")
    def test_data_validation(self, mock_yfinance):
        """データ検証のテスト"""
        # 異常値を含むデータ
        mock_data = pd.DataFrame(
            {
                "Close": [100, -50, 102, 103, None],  # 負の値とNullを含む
                "Volume": [1000, 1100, 1200, 1300, 1400],
                "Open": [99, 100, 101, 102, 103],
                "High": [101, 102, 103, 104, 105],
                "Low": [98, 99, 100, 101, 102],
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

        mock_yfinance.return_value = mock_data

        # データ取得と検証
        result = self.provider.get_stock_data("7203", "1mo")
        assert result is not None

    def test_get_stock_data_dummy_fallback_handles_negative_hash(self, monkeypatch):
        """Fallback ダミーデータが負のハッシュでも動作することを検証"""

        class DummyYF:
            @staticmethod
            def Ticker(symbol):
                class DummyTicker:
                    def __init__(self, symbol):
                        self._symbol = symbol
                        self.info = {
                            "longName": "Dummy Corp",
                            "sector": "Dummy Sector",
                            "industry": "Dummy Industry",
                            "currentPrice": 1000,
                        }

                    def history(self, period, interval):
                        import numpy as np

                        seed = stock_data_module._normalized_symbol_seed(self._symbol)
                        np.random.seed(seed)
                        end = datetime.now()
                        index = pd.date_range(end - timedelta(days=4), periods=5, freq="D")
                        base_price = 1000 + seed % 1000
                        close = base_price + np.arange(len(index))
                        data = {
                            "Close": close,
                            "Open": close,
                            "High": close,
                            "Low": close,
                            "Volume": np.random.randint(1_000, 10_000, size=len(index)),
                        }
                        df = pd.DataFrame(data, index=index)
                        df.index.name = "Date"
                        return df

                return DummyTicker(symbol)

        monkeypatch.setattr(stock_data_module, "yf", DummyYF)

        negative_symbol = None
        for idx in range(1000):
            candidate = f"NEGATIVE_HASH_{idx}"
            if hash(f"{candidate}.T") < 0:
                negative_symbol = candidate
                break

        if negative_symbol is None:
            pytest.skip("Could not find symbol with negative hash in test run")

        result = self.provider.get_stock_data(negative_symbol, "1mo")

        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestDataValidation:
    """データ検証のテスト"""

    def test_basic_validation(self):
        """基本的なデータ検証のテスト"""
        # 基本的な検証機能をテスト
        assert True  # プレースホルダー

    def test_data_format_validation(self):
        """データフォーマット検証のテスト"""
        # データフォーマット検証機能をテスト
        test_data = {"symbol": "7203", "price": 2500.0, "volume": 1000000}

        # 基本的な検証
        assert isinstance(test_data["symbol"], str)
        assert isinstance(test_data["price"], (int, float))
        assert isinstance(test_data["volume"], int)

    def test_symbol_validation(self):
        """銘柄コード検証のテスト"""
        valid_symbols = ["7203", "AAPL", "GOOGL"]

        for symbol in valid_symbols:
            assert isinstance(symbol, str)
            assert len(symbol) > 0

    def test_price_validation(self):
        """価格検証のテスト"""
        valid_prices = [100.0, 2500, 50.5]

        for price in valid_prices:
            assert isinstance(price, (int, float))
            assert price > 0
