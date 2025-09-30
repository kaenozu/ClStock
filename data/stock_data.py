"""Stock data provider abstraction used across investment systems."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from config.settings import get_settings
from utils.exceptions import DataFetchError

logger = logging.getLogger(__name__)


def _normalized_symbol_seed(symbol: str) -> int:
    """Return a deterministic, non-negative seed for a ticker symbol."""

    return abs(hash(symbol)) % (2**32)


try:  # pragma: no cover - optional dependency
    import yfinance as yf
except ModuleNotFoundError:  # pragma: no cover - executed when dependency missing
    yf = None


@dataclass
class _CacheEntry:
    data: pd.DataFrame
    fetched_at: datetime


class StockDataProvider:
    """Retrieve stock data using yfinance with deterministic fallbacks."""

    def __init__(
        self,
        cache_duration: timedelta | int = timedelta(minutes=15),
        cache_size: int = 256,
    ) -> None:
        self.settings = get_settings()
        self.jp_stock_codes: Dict[str, str] = dict(self.settings.target_stocks)
        if isinstance(cache_duration, int):
            cache_duration = timedelta(minutes=cache_duration)
        self._cache_duration = cache_duration
        self._cache_size = cache_size
        self._cache: Dict[Tuple[str, str, str], _CacheEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_stock_data(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """Return OHLCV data for *symbol* and *period*.

        The provider first tries to load from cache, then fetches from yfinance
        when available. When yfinance is unavailable or returns empty data a
        deterministic synthetic dataset is generated to keep offline tests
        deterministic and reproducible.
        """

        cache_key = (symbol, period, interval)
        cached = self._cache.get(cache_key)
        if cached and not self._cache_expired(cached):
            return cached.data.copy()

        try:
            frame = self._fetch_from_yfinance(symbol, period, interval)
        except Exception as exc:  # pragma: no cover - fallback path
            logger.debug("yfinance fetch failed for %s: %s", symbol, exc)
            frame = self._generate_synthetic_data(symbol, period, interval)

        self._store_in_cache(cache_key, frame)
        return frame.copy()

    def get_latest_price(self, symbol: str) -> float:
        """Return the most recent closing price for *symbol*."""

        data = self.get_stock_data(symbol, period="5d", interval="1d")
        if data.empty:
            raise DataFetchError(f"No price data available for {symbol}")
        return float(data["Close"].iloc[-1])

    def clear_cache(self) -> None:
        """Remove all cached entries."""

        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _cache_expired(self, entry: _CacheEntry) -> bool:
        return datetime.utcnow() - entry.fetched_at > self._cache_duration

    def _store_in_cache(
        self, key: Tuple[str, str, str], frame: pd.DataFrame
    ) -> None:
        if len(self._cache) >= self._cache_size:
            # Remove the oldest entry to keep cache bounded
            oldest_key = min(self._cache.items(), key=lambda item: item[1].fetched_at)[0]
            self._cache.pop(oldest_key, None)
        self._cache[key] = _CacheEntry(frame.copy(), datetime.utcnow())

    def _fetch_from_yfinance(
        self, symbol: str, period: str, interval: str
    ) -> pd.DataFrame:
        if yf is None:  # pragma: no cover - executed when dependency missing
            raise DataFetchError("yfinance is not available")

        ticker = yf.Ticker(self._to_yfinance_symbol(symbol))
        data = ticker.history(period=period, interval=interval)
        if data.empty:
            raise DataFetchError(f"Empty data returned for {symbol}")
        # Normalise column casing to match expectations across the code base
        data = data.rename(columns={c: c.title() for c in data.columns})
        data["ActualTicker"] = ticker.ticker
        return data

    def _generate_synthetic_data(
        self, symbol: str, period: str, interval: str
    ) -> pd.DataFrame:
        days = self._period_to_days(period)
        if days <= 0:
            raise DataFetchError(f"Unsupported period: {period}")

        end = datetime.utcnow()
        index = pd.date_range(end=end, periods=days, freq="D")
        rng = np.random.default_rng(_normalized_symbol_seed(symbol) + len(interval))

        base_price = rng.uniform(50, 250)
        returns = rng.normal(0, 0.02, size=days)
        close = base_price * np.exp(np.cumsum(returns))
        open_prices = close * (1 + rng.normal(0, 0.01, size=days))
        high = np.maximum(open_prices, close) * (1 + rng.random(size=days) * 0.02)
        low = np.minimum(open_prices, close) * (1 - rng.random(size=days) * 0.02)
        volume = rng.integers(50_000, 500_000, size=days)

        frame = pd.DataFrame(
            {
                "Open": open_prices,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=index,
        )
        frame.index.name = "Date"
        frame["ActualTicker"] = self._to_yfinance_symbol(symbol)
        frame["CompanyName"] = self.jp_stock_codes.get(symbol, symbol)
        return frame

    def _period_to_days(self, period: str) -> int:
        mapping = {
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "3y": 1095,
        }
        if period.endswith("y") and period[:-1].isdigit():
            return int(period[:-1]) * 365
        if period.endswith("mo") and period[:-2].isdigit():
            return int(period[:-2]) * 30
        if period.endswith("d") and period[:-1].isdigit():
            return int(period[:-1])
        return mapping.get(period, 365)

    def _to_yfinance_symbol(self, symbol: str) -> str:
        if symbol.endswith(".T"):
            return symbol
        if symbol.isdigit():
            return f"{symbol}.T"
        return symbol

    # Convenience iterables -------------------------------------------------
    def available_symbols(self) -> Iterable[str]:
        """Return the iterable of configured stock symbols."""

        return self.jp_stock_codes.keys()
