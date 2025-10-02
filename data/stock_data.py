"""Stock market data utilities used throughout the project.

The original implementation of this module was partially removed which left
the application without a working ``StockDataProvider``.  This file restores a
compact yet fully functional provider along with a deterministic fallback
``yfinance`` stub so the rest of the codebase can continue to work in
environments where the real dependency is unavailable (for example during
tests).
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from config.settings import get_settings
from utils.cache import get_cache
from utils.exceptions import DataFetchError


logger = logging.getLogger(__name__)


def _normalized_symbol_seed(symbol: str) -> int:
    """Return a deterministic, non-negative seed for a ticker symbol."""

    digest = hashlib.sha256(symbol.encode("utf-8", "surrogatepass")).digest()
    return int.from_bytes(digest[:4], "big")


def _period_to_days(period: str) -> int:
    """Convert a Yahoo-finance period string to an approximate number of days."""

    mapping = {
        "1d": 1,
        "5d": 5,
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "10y": 3650,
        "ytd": 365,
        "max": 365 * 30,
    }
    return mapping.get(period, 120)


try:  # pragma: no cover - exercised indirectly through tests
    import yfinance as yf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - hit in constrained envs

    class _FallbackTicker:
        """Minimal stand-in for ``yfinance.Ticker``.

        It produces deterministic pseudo-random data so higher level code can
        continue to operate without network access.
        """

        def __init__(self, symbol: str) -> None:
            self.symbol = symbol
            self._seed = _normalized_symbol_seed(symbol)
            base_price = 80 + (self._seed % 100) / 2
            self._info = {
                "shortName": f"{symbol} Corp",
                "marketCap": 100_000_000 + (self._seed % 10_000_000),
                "forwardPE": 10.0 + (self._seed % 2000) / 200,
                "trailingPE": 12.0 + (self._seed % 3000) / 200,
                "dividendYield": 0.01 + ((self._seed // 5) % 200) / 10_000,
                "beta": 1.0 + (self._seed % 500) / 1000,
            }

            self._fast_info = SimpleNamespace(
                last_price=base_price + 1.5,
                previous_close=base_price,
                ten_day_average_volume=150_000 + (self._seed % 50_000),
            )

        @property
        def info(self) -> Dict[str, float]:
            return self._info

        @property
        def fast_info(self) -> SimpleNamespace:
            return self._fast_info

        def history(self, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
            length = max(5, _period_to_days(period))
            end = datetime.utcnow().date()
            index = pd.date_range(end - timedelta(days=length + 5), periods=length, freq="B")

            base = 90 + (self._seed % 60)
            variation = (self._seed % 13) - 6
            close = pd.Series(
                [base + variation * (i % 5) * 0.5 for i in range(len(index))],
                index=index,
                dtype="float",
            )
            open_ = close.shift(1, fill_value=close.iloc[0] - 0.5)
            high = pd.concat([open_, close], axis=1).max(axis=1) + 0.3
            low = pd.concat([open_, close], axis=1).min(axis=1) - 0.3
            volume = pd.Series(
                [100_000 + ((self._seed + i) % 20_000) for i in range(len(index))],
                index=index,
                dtype="int64",
            )
            return pd.DataFrame(
                {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
                index=index,
            )

    class yf:  # type: ignore
        @staticmethod
        def Ticker(symbol: str) -> _FallbackTicker:
            return _FallbackTicker(symbol)


class StockDataProvider:
    """Fetch and enrich historical stock data.

    The provider favours cached and local data when available but can fall back
    to ``yfinance`` for live downloads.  It also exposes helper utilities for
    technical indicators and basic financial metrics.
    """

    def __init__(self) -> None:
        settings = get_settings()
        # Always keep codes sorted for deterministic behaviour
        self.jp_stock_codes: Dict[str, str] = dict(sorted(settings.target_stocks.items()))
        self.cache_ttl = int(os.getenv("CLSTOCK_STOCK_CACHE_TTL", "1800"))
        default_data_dir = Path(os.getenv("CLSTOCK_DATA_DIR", Path(__file__).resolve().parent))
        self._history_dirs: List[Path] = [default_data_dir / "historical", Path(settings.database.personal_portfolio_db).parent]
        self._history_dirs = [path for path in self._history_dirs if path and path.exists()]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_all_stock_symbols(self) -> List[str]:
        """Return all supported stock symbols."""

        return list(self.jp_stock_codes.keys())

    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch historical OHLCV data for *symbol*.

        The result always contains the ``Symbol``, ``ActualTicker`` and
        ``CompanyName`` columns in addition to the usual OHLCV fields.
        """

        cache_key = f"stock::{symbol}::{period}"
        cache = get_cache()
        cached = cache.get(cache_key)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached

        data: Optional[pd.DataFrame] = None
        actual_ticker: Optional[str] = None

        if self._should_use_local_first(symbol):
            local = self._load_first_available_csv(symbol)
            if local is not None:
                data, actual_ticker = local

        if data is None or data.empty:
            data, actual_ticker = self._download_via_yfinance(symbol, period)

        if data is None or data.empty:
            raise DataFetchError(symbol, "No historical data available")

        prepared = self._prepare_history_frame(data, symbol, actual_ticker)
        cache.set(cache_key, prepared, ttl=self.cache_ttl)
        return prepared

    def get_multiple_stocks(
        self, symbols: Iterable[str], period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols, skipping those that fail."""

        result: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                result[symbol] = self.get_stock_data(symbol, period)
            except DataFetchError:
                logger.debug("Skipping symbol %s due to missing data", symbol)
        return result

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate a selection of technical indicators."""

        if data is None or data.empty:
            return data.copy()

        df = data.copy()
        close = df["Close"].astype("float")
        df["Close"] = close
        high = df.get("High", close)
        low = df.get("Low", close)

        df["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
        df["SMA_50"] = close.rolling(window=50, min_periods=1).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace({0: pd.NA})
        df["RSI"] = (100 - (100 / (1 + rs))).fillna(100.0)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["ATR"] = true_range.rolling(window=14, min_periods=1).mean()

        return df

    def get_financial_metrics(self, symbol: str) -> Dict[str, Optional[float]]:
        """Return a dictionary with lightweight financial metrics."""

        cache_key = f"financial::{symbol}"
        cache = get_cache()
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            return cached

        metrics: Dict[str, Optional[float]] = {
            "symbol": symbol,
            "company_name": self.jp_stock_codes.get(symbol, symbol),
            "market_cap": None,
            "pe_ratio": None,
            "dividend_yield": None,
            "beta": None,
            "last_price": None,
            "previous_close": None,
            "ten_day_average_volume": None,
            "actual_ticker": None,
        }

        for ticker in self._ticker_formats(symbol):
            try:
                ticker_obj = yf.Ticker(ticker)
                info = getattr(ticker_obj, "info", {}) or {}
                fast = getattr(ticker_obj, "fast_info", None)

                metrics.update(
                    {
                        "market_cap": info.get("marketCap", metrics["market_cap"]),
                        "pe_ratio": info.get("forwardPE") or info.get("trailingPE"),
                        "dividend_yield": info.get("dividendYield"),
                        "beta": info.get("beta", metrics["beta"]),
                    }
                )

                if fast is not None:
                    metrics["last_price"] = getattr(fast, "last_price", metrics["last_price"])
                    metrics["previous_close"] = getattr(
                        fast, "previous_close", metrics["previous_close"]
                    )
                    metrics["ten_day_average_volume"] = getattr(
                        fast, "ten_day_average_volume", metrics["ten_day_average_volume"]
                    )

                metrics["actual_ticker"] = ticker
                break
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to retrieve financial metrics for %s via %s: %s", symbol, ticker, exc)
                continue

        cache.set(cache_key, metrics, ttl=self.cache_ttl)
        return metrics

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _ticker_formats(self, symbol: str) -> List[str]:
        normalized = symbol.upper()
        if "." in normalized:
            base, _ = normalized.split(".", 1)
        else:
            base = normalized

        variants = [normalized, f"{base}.T", f"{base}.TO", base]
        seen: Dict[str, None] = {}
        result: List[str] = []
        for candidate in variants:
            if candidate not in seen:
                seen[candidate] = None
                result.append(candidate)
        return result

    def _should_use_local_first(self, symbol: str) -> bool:  # pragma: no cover - logic is trivial
        prefer_local = os.getenv("CLSTOCK_PREFER_LOCAL_DATA", "0").lower()
        return prefer_local in {"1", "true", "yes"}

    def _load_first_available_csv(self, symbol: str) -> Optional[Tuple[pd.DataFrame, str]]:
        for ticker in self._ticker_formats(symbol):
            filename = f"{ticker}.csv"
            for directory in self._history_dirs:
                candidate = directory / filename
                if candidate.exists():
                    try:
                        df = pd.read_csv(candidate, index_col=0, parse_dates=True)
                        return df, ticker
                    except Exception as exc:  # pragma: no cover - disk errors
                        logger.debug("Failed to load local CSV %s: %s", candidate, exc)
        return None

    def _download_via_yfinance(self, symbol: str, period: str) -> Tuple[pd.DataFrame, Optional[str]]:
        last_error: Optional[Exception] = None
        for ticker in self._ticker_formats(symbol):
            try:
                ticker_obj = yf.Ticker(ticker)
                history = ticker_obj.history(period=period)
                if isinstance(history, pd.DataFrame) and not history.empty:
                    return history, ticker
            except Exception as exc:  # pragma: no cover - depends on yfinance
                last_error = exc
                logger.debug("Failed to download %s via yfinance: %s", ticker, exc)
        if last_error:
            logger.warning("All yfinance attempts failed for %s: %s", symbol, last_error)
        return pd.DataFrame(), None

    def _prepare_history_frame(
        self, data: pd.DataFrame, symbol: str, actual_ticker: Optional[str]
    ) -> pd.DataFrame:
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        for column in ["Open", "High", "Low", "Close", "Volume"]:
            if column not in df:
                df[column] = pd.NA

        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Close"] = df["Close"].ffill().bfill()
        for column in ["Open", "High", "Low"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df[column] = df[column].fillna(df["Close"])

        if "Volume" in df:
            df["Volume"] = (
                pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype("int64")
            )

        df["Symbol"] = symbol
        df["ActualTicker"] = actual_ticker or self._ticker_formats(symbol)[0]
        df["CompanyName"] = self.jp_stock_codes.get(symbol, symbol)
        return df


# Backwards compatibility helper -------------------------------------------------

_stock_data_provider: Optional[StockDataProvider] = None


def get_stock_data_provider() -> StockDataProvider:
    """Singleton-style access used by legacy modules."""

    global _stock_data_provider
    if _stock_data_provider is None:
        _stock_data_provider = StockDataProvider()
    return _stock_data_provider


    def _generate_demo_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        デモデータを生成する
        """
        import pandas as pd
        from datetime import datetime, timedelta
        import numpy as np

        # period に応じた日数を計算
        period_map = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "10y": 3650,
            "max": 3650,  # maxも10年分とする
        }
        days = period_map.get(period, 365)  # デフォルトは1年

        # データ生成
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
        base_price = np.random.uniform(100, 1000)  # 基準価格
        prices = [base_price]
        volumes = []

        for _ in range(1, len(dates)):
            change_percent = np.random.uniform(-0.05, 0.05)  # -5% 〜 +5% の変動
            new_price = prices[-1] * (1 + change_percent)
            prices.append(new_price)
            volumes.append(np.random.randint(1000, 100000))  # ランダムな取引量

        # DataFrame に変換
        df = pd.DataFrame({
            "Open": prices,
            "High": [p * np.random.uniform(1, 1.02) for p in prices],  # HighはOpenより少し高い
            "Low": [p * np.random.uniform(0.98, 1) for p in prices],   # LowはOpenより少し低い
            "Close": prices,
            "Volume": volumes,
        }, index=dates)

        # 一部の値をNaNに置き換えて、検証用に使う
        if np.random.random() > 0.9:  # 10%の確率で
            df.loc[df.index[:5], "Close"] = np.nan  # 最初の5行をNaN

        return df
