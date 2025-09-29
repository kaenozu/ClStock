import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple

import pandas as pd

from config.settings import get_settings
from utils.exceptions import DataFetchError

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


def get_cache():
    """Lazy proxy to avoid importing heavy cache infrastructure at module load."""

    from utils.cache import get_cache as _get_cache  # local import to avoid recursion issues

    return _get_cache()

# ロギング設定
setup_logging()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TickerInfo:
    """Simple container for stock symbol metadata."""

    symbol: str
    company_name: str


class StockDataProvider:
    """
    株価データを提供するクラス。
    yfinanceからデータを取得し、キャッシュする機能を持つ。
    """

    def __init__(self):
        settings = get_settings()
        self.jp_stock_codes: Dict[str, str] = dict(settings.target_stocks)
        self.cache: Dict[str, pd.DataFrame] = {}
        self.logger = logger

    def _fetch_data_from_yfinance(
        self, symbol: str, period: str, interval: str
    ) -> pd.DataFrame:
        """
        yfinanceから株価データを取得する内部メソッド。
        """
        ticker_symbol = symbol if symbol.endswith(".T") else f"{symbol}.T"
        ticker = yf.Ticker(ticker_symbol)  # 日本株向けに.Tを付与
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

    def get_financial_metrics(self, symbol: str) -> Dict[str, Any]:
        """Retrieve a simplified set of financial metrics using yfinance."""

        ticker = yf.Ticker(f"{symbol}.T")
        info = getattr(ticker, "info", {}) or {}

        return {
            "symbol": symbol,
            "company_name": info.get("longName", self.jp_stock_codes.get(symbol, symbol)),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "return_on_equity": info.get("returnOnEquity"),
        }

    # ------------------------------------------------------------------
    # 新規追加のユーティリティメソッド
    # ------------------------------------------------------------------

    def get_all_stock_symbols(self) -> List[str]:
        """Return the list of supported Japanese stock symbols."""

        if not self.jp_stock_codes:
            return []
        return sorted(self.jp_stock_codes.keys())

    def get_all_tickers(self) -> List[TickerInfo]:
        """Return metadata objects for all known tickers."""

        return [
            TickerInfo(symbol=symbol, company_name=name)
            for symbol, name in self.jp_stock_codes.items()
        ]

    # ------------------------------------------------------------------
    # データ取得ラッパー
    # ------------------------------------------------------------------

    def _ticker_formats(self, symbol: str) -> List[str]:
        """Generate likely ticker strings for a given symbol."""

        normalized = symbol.upper().strip()
        normalized = normalized.replace(".T", "")
        candidates = [f"{normalized}.T", normalized]
        # 末尾に.TOを付与するケースを追加
        candidates.append(f"{normalized}.TO")
        # 重複排除しつつ順序維持
        seen = set()
        ordered = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)
        return ordered

    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch stock data with optional local caching support."""

        if not symbol:
            raise DataFetchError(symbol or "", "Symbol must be provided")

        normalized = symbol.upper().replace(".T", "")
        cache_key = f"stock_data:{normalized}:{period}"

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        cache_backend = None
        try:
            cache_backend = get_cache()
        except Exception as exc:  # pragma: no cover
            self.logger.debug("Cache backend unavailable: %s", exc)

        backend_data = (
            cache_backend.get(cache_key) if cache_backend is not None else None
        )
        if backend_data is not None:
            self.cache[cache_key] = backend_data.copy()
            return backend_data.copy()

        dataset: Optional[pd.DataFrame] = None
        actual_ticker: Optional[str] = None

        if self._should_use_local_first(period):
            local_result = self._load_first_available_csv(normalized, period)
            if local_result is not None:
                dataset, actual_ticker = local_result

        if dataset is None:
            dataset, actual_ticker = self._download_via_yfinance(normalized, period)

        if dataset is None or dataset.empty:
            raise DataFetchError(normalized, "No historical data available")

        enriched = self._enrich_stock_dataframe(
            dataset, normalized, actual_ticker or normalized
        )

        self.cache[cache_key] = enriched.copy()
        if cache_backend is not None:
            cache_backend.set(cache_key, enriched.copy(), ttl=1800)

        return enriched.copy()

    def get_multiple_stocks(
        self, symbols: List[str], period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols, skipping failures."""

        results: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_stock_data(symbol, period)
            except DataFetchError:
                self.logger.warning("Skipping symbol due to fetch error: %s", symbol)
        return results

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Append a small set of technical indicators for convenience."""

        if data is None or data.empty:
            return pd.DataFrame()

        df = data.copy()
        close = df["Close"]

        df["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
        df["SMA_50"] = close.rolling(window=50, min_periods=1).mean()

        delta = close.diff().fillna(0)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean().replace(0, 1e-9)
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

        high_low = df["High"] - df["Low"] if "High" in df and "Low" in df else close.diff().abs()
        close_prev = (df["Close"] - df["Close"].shift()).abs()
        true_range = pd.concat([high_low, close_prev], axis=1).max(axis=1)
        df["ATR"] = true_range.rolling(window=14, min_periods=1).mean()

        return df

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------

    def _enrich_stock_dataframe(
        self, data: pd.DataFrame, symbol: str, actual_ticker: str
    ) -> pd.DataFrame:
        df = data.copy()
        df["Symbol"] = symbol
        df["CompanyName"] = self.jp_stock_codes.get(symbol, symbol)
        df["ActualTicker"] = actual_ticker

        return df

    def _should_use_local_first(self, period: str) -> bool:
        prefer_local = os.getenv("CLSTOCK_PREFER_LOCAL_DATA")
        if prefer_local is not None:
            return prefer_local.lower() in {"1", "true", "yes"}
        return period not in {"1d", "5d"}

    def _local_data_paths(self, symbol: str, period: str) -> Iterable[Path]:
        candidates = [
            Path("data") / f"{symbol}_{period}.csv",
            Path("data") / "historical" / f"{symbol}_{period}.csv",
        ]
        for candidate in candidates:
            yield candidate

    def _load_first_available_csv(
        self, symbol: str, period: str
    ) -> Optional[Tuple[pd.DataFrame, str]]:
        for path in self._local_data_paths(symbol, period):
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.index.name = "Date"
                self.logger.info("Loaded local data for %s from %s", symbol, path)
                return df, symbol
            except Exception as exc:
                self.logger.warning("Failed to load local data %s: %s", path, exc)
        return None

    def _download_via_yfinance(
        self, symbol: str, period: str
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        last_error: Optional[Exception] = None

        for ticker_symbol in self._ticker_formats(symbol):
            try:
                dataset = self.fetch_stock_data(ticker_symbol, period=period)
                if dataset is not None and not dataset.empty:
                    self.logger.info(
                        "Fetched %d rows for %s via yfinance", len(dataset), ticker_symbol
                    )
                    return dataset, ticker_symbol
            except Exception as exc:
                last_error = exc
                self.logger.warning("Download failed for %s: %s", ticker_symbol, exc)

        if last_error:
            raise DataFetchError(symbol, "Failed to download via yfinance", str(last_error))

        return pd.DataFrame(), None