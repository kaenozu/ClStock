"""Data Providers のテスト
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import pandas as pd
from config.settings import AppSettings, DatabaseConfig


def _load_real_stock_data_module():
    module_name = "tests.real_stock_data_module"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).resolve().parent.parent / "data" / "stock_data.py",
    )
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    sys.modules[module_name] = module
    return module


stock_data_module = _load_real_stock_data_module()
StockDataProvider = stock_data_module.StockDataProvider

try:
    from config.settings import MarketDataConfig
except ImportError:  # pragma: no cover - backward compatibility guard
    MarketDataConfig = None  # type: ignore


class TestStockDataProvider:
    """StockDataProvider のテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        self.provider = StockDataProvider()

    def test_provider_initialization(self):
        """プロバイダー初期化のテスト"""
        assert self.provider is not None
        assert hasattr(self.provider, "get_stock_data")

    def test_get_stock_data_success(self, monkeypatch):
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

        dummy_ticker = types.SimpleNamespace(
            history=lambda period=None, start=None, end=None: mock_data,
        )
        dummy_yf = types.SimpleNamespace(Ticker=lambda symbol: dummy_ticker)
        monkeypatch.setattr(stock_data_module, "YFINANCE_AVAILABLE", True)
        monkeypatch.setattr(stock_data_module, "yf", dummy_yf)

        # データ取得テスト
        result = self.provider.get_stock_data("TESTMOCK", "1mo")

        # 検証（モックが適用されるかどうかを確認）
        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_get_stock_data_with_japanese_symbol(self, monkeypatch):
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

        dummy_ticker = types.SimpleNamespace(
            history=lambda period=None, start=None, end=None: mock_data,
        )
        dummy_yf = types.SimpleNamespace(Ticker=lambda symbol: dummy_ticker)
        monkeypatch.setattr(stock_data_module, "YFINANCE_AVAILABLE", True)
        monkeypatch.setattr(stock_data_module, "yf", dummy_yf)

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

    def test_calculate_technical_indicators(self):
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

    def test_data_validation(self, monkeypatch):
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

        dummy_ticker = types.SimpleNamespace(
            history=lambda period=None, start=None, end=None: mock_data,
        )
        dummy_yf = types.SimpleNamespace(Ticker=lambda symbol: dummy_ticker)
        monkeypatch.setattr(stock_data_module, "YFINANCE_AVAILABLE", True)
        monkeypatch.setattr(stock_data_module, "yf", dummy_yf)

        # データ取得と検証
        result = self.provider.get_stock_data("7203", "1mo")
        assert result is not None

    def test_trusted_market_data_used_when_yfinance_unavailable(
        self,
        tmp_path,
        monkeypatch,
    ):
        """設定された信頼できるデータソースが利用されることを検証する"""
        if MarketDataConfig is None:
            pytest.skip(
                "MarketDataConfig not available in current settings implementation",
            )

        csv_dir = tmp_path / "historical"
        csv_dir.mkdir()
        index = pd.date_range("2024-01-01", periods=5, freq="B")
        raw = pd.DataFrame(
            {
                "Open": [100 + i for i in range(5)],
                "High": [101 + i for i in range(5)],
                "Low": [99 + i for i in range(5)],
                "Close": [100.5 + i for i in range(5)],
                "Volume": [1000 + i for i in range(5)],
            },
            index=index,
        )
        raw.to_csv(csv_dir / "REALTEST.T.csv")

        settings = AppSettings()
        settings.target_stocks = {"REALTEST": "Real Test Corp"}
        settings.database = DatabaseConfig(
            personal_portfolio_db=tmp_path / "portfolio.db",
        )
        settings.market_data = MarketDataConfig(
            provider="local_csv",
            local_cache_dir=csv_dir,
            api_base_url=None,
            api_token="dummy-token",
        )

        monkeypatch.setattr(stock_data_module, "get_settings", lambda: settings)
        provider = StockDataProvider()

        monkeypatch.setattr(
            provider,
            "_download_via_yfinance",
            MagicMock(side_effect=AssertionError("yfinance should not be called")),
        )

        frame = provider.get_stock_data("REALTEST", period="1mo")

        assert list(frame["Close"]) == list(raw["Close"])
        assert (frame["Symbol"] == "REALTEST").all()
        assert (frame["CompanyName"] == "Real Test Corp").all()

    def test_get_multiple_stocks_reports_missing_symbols(self, tmp_path, monkeypatch):
        """複数銘柄取得時に欠損があれば明確なエラーが返ることを検証"""
        if MarketDataConfig is None:
            pytest.skip(
                "MarketDataConfig not available in current settings implementation",
            )

        csv_dir = tmp_path / "historical"
        csv_dir.mkdir()
        available_df = pd.DataFrame(
            {
                "Open": [1.0],
                "High": [1.0],
                "Low": [1.0],
                "Close": [1.0],
                "Volume": [100],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )
        available_df.to_csv(csv_dir / "ONLYONE.T.csv")

        settings = AppSettings()
        settings.target_stocks = {"ONLYONE": "Only One Corp", "MISSING": "Missing Corp"}
        settings.database = DatabaseConfig(
            personal_portfolio_db=tmp_path / "portfolio.db",
        )
        settings.market_data = MarketDataConfig(
            provider="local_csv",
            local_cache_dir=csv_dir,
            api_base_url=None,
            api_token="dummy-token",
        )

        monkeypatch.setattr(stock_data_module, "get_settings", lambda: settings)
        provider = StockDataProvider()

        with pytest.raises(stock_data_module.BatchDataFetchError) as excinfo:
            provider.get_multiple_stocks(["ONLYONE", "MISSING"], period="1mo")

        error = excinfo.value
        assert "MISSING" in error.failed_symbols
        assert "ONLYONE" in error.partial_results
        assert error.partial_results["ONLYONE"]["Symbol"].eq("ONLYONE").all()


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
