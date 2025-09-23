"""
Asynchronous stock data provider
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
from utils.logger_config import setup_logger
from utils.exceptions import DataFetchError, InvalidSymbolError, NetworkError
from utils.connection_pool import get_http_pool

logger = setup_logger(__name__)


class AsyncStockDataProvider:
    """Asynchronous stock data provider for I/O-bound operations"""

    def __init__(self):
        from config.settings import get_settings

        settings = get_settings()
        self.jp_stock_codes: Dict[str, str] = settings.target_stocks
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp client session"""
        if self._session is None or self._session.closed:
            # Create connection pool
            pool = get_http_pool(max_connections=20, connection_timeout=30)
            # For now, we'll create a simple session - in a full implementation,
            # we would integrate with the connection pool
            connector = aiohttp.TCPConnector(limit=20, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._session

    async def close(self):
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Asynchronously fetch stock data"""
        from utils.cache import get_cache

        # Generate cache key
        cache_key = f"stock_data_{symbol}_{period}"
        cache = get_cache()

        # Try to get from cache first
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            if symbol in self.jp_stock_codes:
                ticker = f"{symbol}.T"
            else:
                ticker = symbol

            logger.info(f"Fetching data for {symbol} (period: {period})")

            # Use yfinance with async wrapper
            data = await self._fetch_with_yfinance_async(ticker, period)

            if data.empty:
                raise DataFetchError(symbol, "No historical data available")

            data["Symbol"] = symbol
            data["CompanyName"] = self.jp_stock_codes.get(symbol, symbol)

            # Save to cache (30 minutes)
            cache.set(cache_key, data, ttl=1800)

            return data

        except DataFetchError:
            raise
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise DataFetchError(symbol, "Unexpected error during data fetch", str(e))

    async def _fetch_with_yfinance_async(
        self, ticker: str, period: str
    ) -> pd.DataFrame:
        """Fetch data using yfinance with async wrapper"""
        # yfinance is not async by nature, so we'll run it in a thread pool
        loop = asyncio.get_event_loop()

        def fetch_data():
            import yfinance as yf

            stock = yf.Ticker(ticker)
            return stock.history(period=period)

        # Run in thread pool to avoid blocking
        data = await loop.run_in_executor(None, fetch_data)
        return data

    async def get_multiple_stocks(
        self, symbols: List[str], period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """Asynchronously fetch data for multiple stocks"""
        from utils.exceptions import DataFetchError, InvalidSymbolError

        result: Dict[str, pd.DataFrame] = {}
        failed_symbols: List[str] = []

        # Create tasks for all symbols
        tasks = [self._fetch_single_stock_async(symbol, period) for symbol in symbols]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for symbol, result_data in zip(symbols, results):
            if isinstance(result_data, Exception):
                logger.warning(f"Failed to fetch data for {symbol}: {result_data}")
                failed_symbols.append(symbol)
            elif result_data is not None:
                result[symbol] = result_data

        if failed_symbols:
            logger.info(f"Failed to fetch data for symbols: {failed_symbols}")

        return result

    async def _fetch_single_stock_async(
        self, symbol: str, period: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single stock asynchronously"""
        try:
            return await self.get_stock_data(symbol, period)
        except (DataFetchError, InvalidSymbolError) as e:
            logger.warning(f"Failed to fetch data for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {symbol}: {e}")
            return None

    async def calculate_technical_indicators_async(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Asynchronously calculate technical indicators"""
        from utils.cache import get_cache

        if data.empty:
            return data

        # Generate cache key with symbol information
        symbol = data["Symbol"].iloc[0] if "Symbol" in data.columns else "unknown"
        cache_key = f"technical_indicators_{symbol}_{hash(str(data.index.tolist()))}"
        cache = get_cache()

        # Try to get from cache first
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # In a real implementation, we would make this truly async
        # For now, we'll just run the existing synchronous function
        df = data.copy()

        # Moving averages (vectorized for speed)
        df["SMA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()

        # RSI (efficient calculation)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()

        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD (efficient exponential moving average)
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # ATR (efficient True Range calculation)
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = true_range.rolling(window=14, min_periods=1).mean()

        # Save to cache (30 minutes)
        cache.set(cache_key, df, ttl=1800)

        return df


# Global instance
_async_stock_data_provider: Optional[AsyncStockDataProvider] = None


def get_async_stock_data_provider() -> AsyncStockDataProvider:
    """Get the global async stock data provider instance"""
    global _async_stock_data_provider
    if _async_stock_data_provider is None:
        _async_stock_data_provider = AsyncStockDataProvider()
    return _async_stock_data_provider


async def close_async_stock_data_provider():
    """Close the global async stock data provider"""
    global _async_stock_data_provider
    if _async_stock_data_provider:
        await _async_stock_data_provider.close()
        _async_stock_data_provider = None
