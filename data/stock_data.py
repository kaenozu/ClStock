import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# yfinance を try-except でインポート
try:
    import yfinance as yf
except ModuleNotFoundError:
    # yfinance が見つからない場合、ダミーのクラスを提供
    class yf:
        @staticmethod
        def Ticker(symbol):
            class DummyTicker:
                def __init__(self):
                    self.info = {
                        "longName": "Dummy Corp",
                        "sector": "Dummy Sector",
                        "industry": "Dummy Industry",
                        "marketCap": 0,
                        "trailingPE": 0,
                        "priceToBook": 0,
                        "dividendYield": 0,
                        "returnOnEquity": 0,
                        "currentPrice": 1000, # デフォルトの価格
                        "targetMeanPrice": 1100,
                        "recommendationMean": 2.0, # デフォルトの推奨スコア
                    }

                def history(self, period, interval):
                    # 現在時刻から指定された期間分のダミーデータを生成
                    end_date = datetime.now()
                    if "d" in period:
                        days = int(period.replace("d", ""))
                        start_date = end_date - timedelta(days=days)
                    elif "mo" in period:
                        # 月指定は単純に30日として計算
                        months = int(period.replace("mo", ""))
                        start_date = end_date - timedelta(days=months * 30)
                    elif "y" in period:
                        years = int(period.replace("y", ""))
                        start_date = end_date - timedelta(days=years * 365)
                    else:
                        # デフォルトは1年
                        start_date = end_date - timedelta(days=365)

                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                    # ランダムな価格変動をシミュレート
                    import numpy as np
                    np.random.seed(hash(symbol))  # 銘柄ごとに同じ乱数シード
                    base_price = 1000 + hash(symbol) % 1000  # 銘柄ごとに異なるベース価格
                    price_changes = np.random.normal(0, 10, size=len(date_range))
                    prices = np.concatenate([[base_price], base_price + np.cumsum(price_changes[1:])])
                    df = pd.DataFrame(index=date_range, data={'Close': prices, 'Open': prices, 'High': prices, 'Low': prices, 'Volume': np.random.randint(100000, 1000000, size=len(date_range))})
                    df.index.name = "Date"
                    return df

            return DummyTicker()
from utils.logger import setup_logging

# ロギング設定
setup_logging()
logger = logging.getLogger(__name__)


class StockDataProvider:
    """
    株価データを提供するクラス。
    yfinanceからデータを取得し、キャッシュする機能を持つ。
    """

    def __init__(self):
        self.cache: Dict[str, pd.DataFrame] = {}
        self.logger = logger

    def _fetch_data_from_yfinance(
        self, symbol: str, period: str, interval: str
    ) -> pd.DataFrame:
        """
        yfinanceから株価データを取得する内部メソッド。
        """
        ticker = yf.Ticker(f"{symbol}.T")  # 日本株向けに.Tを付与
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            self.logger.warning(f"No data fetched for {symbol} from yfinance.")
            return pd.DataFrame()
        df.index.name = "Date"
        return df

    def fetch_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        指定された期間の株価データを取得する。
        キャッシュが有効な場合、キャッシュからデータを返す。
        """
        cache_key = f"{symbol}_{period}_{interval}_{start_date}_{end_date}"

        if use_cache and cache_key in self.cache:
            self.logger.info(f"Returning cached data for {symbol}")
            return self.cache[cache_key].copy()

        self.logger.info(f"Fetching stock data for {symbol} from yfinance...")

        df = self._fetch_data_from_yfinance(symbol, period, interval)

        if not df.empty:
            self.cache[cache_key] = df.copy()
            self.logger.info(f"Successfully fetched and cached data for {symbol}.")
        else:
            self.logger.warning(f"Failed to fetch data for {symbol}.")

        return df

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        指定された銘柄の現在の株価を取得する。
        """
        df = self.fetch_stock_data(symbol, period="1d", interval="1m")
        if not df.empty:
            return df["Close"].iloc[-1]
        return None

    def get_historical_data(
        self, symbol: str, days: int = 365
    ) -> Optional[pd.DataFrame]:
        """
        指定された銘柄の過去の株価データを取得する。
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = self.fetch_stock_data(
            symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            period=f"{days}d",
        )
        return df if not df.empty else None

    def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        指定された銘柄の企業情報を取得する。
        """
        ticker = yf.Ticker(f"{symbol}.T")
        info = ticker.info
        if info:
            return {
                "symbol": symbol,
                "longName": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "marketCap": info.get("marketCap"),
                "trailingPE": info.get("trailingPE"),
                "priceToBook": info.get("priceToBook"),
                "dividendYield": info.get("dividendYield"),
                "returnOnEquity": info.get("returnOnEquity"),
                "currentPrice": info.get("currentPrice"),
                "targetMeanPrice": info.get("targetMeanPrice"),
                "recommendationMean": info.get("recommendationMean"),
            }
        return None