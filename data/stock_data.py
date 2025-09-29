import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

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