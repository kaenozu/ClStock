"""Asynchronous stock data provider"""

import asyncio
from typing import Dict, List, Optional, Tuple

import aiohttp

import pandas as pd
from data.stock_data import StockDataProvider
from utils.exceptions import DataFetchError, InvalidSymbolError, NetworkError
from utils.logger_config import setup_logger

logger = setup_logger(__name__)


class AsyncStockDataProvider:
    """Asynchronous stock data provider for I/O-bound operations"""

    def __init__(self):
        from config.settings import get_settings

        settings = get_settings()
        self.jp_stock_codes: Dict[str, str] = settings.target_stocks
        self._session = None
        self._sync_provider: Optional[StockDataProvider] = None
        self._last_yfinance_error: Optional[str] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp client session"""
        if self._session is None or self._session.closed:
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
            sync_provider = self._get_sync_provider()
            ticker_candidates = sync_provider._ticker_formats(symbol)

            if symbol in self.jp_stock_codes:
                ticker = f"{symbol}.T"
            else:
                ticker = symbol

            logger.info(f"Fetching data for {symbol} (period: {period})")

            trusted_data: Optional[pd.DataFrame] = None
            actual_ticker: Optional[str] = None

            try:
                trusted_data, actual_ticker = await self._fetch_trusted_source_async(
                    sync_provider,
                    symbol,
                    ticker_candidates,
                    period,
                )
            except DataFetchError:
                raise
            except Exception as exc:
                logger.debug("Trusted source fetch failed for %s: %s", symbol, exc)

            if trusted_data is not None and not trusted_data.empty:
                prepared = sync_provider._prepare_history_frame(
                    trusted_data,
                    symbol,
                    actual_ticker,
                )
                cache.set(cache_key, prepared, ttl=1800)
                return prepared

            data = await self._fetch_with_yfinance_async(ticker, period)

            if data.empty:
                raise DataFetchError(
                    symbol,
                    "Failed to fetch data via yfinance",
                    self._last_yfinance_error or "yfinance returned empty dataset",
                )

            prepared = sync_provider._prepare_history_frame(data, symbol, ticker)

            cache.set(cache_key, prepared, ttl=1800)

            return prepared

        except InvalidSymbolError as e:
            logger.error(f"Invalid symbol error for {symbol}: {e!s}")
            raise
        except NetworkError as e:
            logger.error(f"Network error for {symbol}: {e!s}")
            raise
        except DataFetchError as e:
            logger.error(f"Data fetch error for {symbol}: {e!s}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {symbol}: {e!s}")
            raise DataFetchError(symbol, "Unexpected error during data fetch", str(e))

    async def _fetch_with_yfinance_async(
        self,
        ticker: str,
        period: str,
    ) -> pd.DataFrame:
        """Fetch data using yfinance with async wrapper"""
        # yfinance is not async by nature, so we'll run it in a thread pool
        loop = asyncio.get_event_loop()

        # リトライ回数の設定
        max_retries = 3
        retry_count = 0
        self._last_yfinance_error = None
        last_exception: Optional[Exception] = None

        def fetch_data_with_retry(ticker: str, period: str):
            import yfinance as yf

            # yfinanceのインスタンスを取得
            stock = yf.Ticker(ticker)

            # ヒストリカルデータを取得
            data = stock.history(period=period)

            return data

        while retry_count < max_retries:
            try:
                # 関数実行前に少し待機（API負荷軽減のため）
                await asyncio.sleep(0.1)

                # スレッドプールで実行
                data = await loop.run_in_executor(
                    None,
                    fetch_data_with_retry,
                    ticker,
                    period,
                )

                # データ構造を確認
                if not data.empty:
                    logger.info(
                        f"Successfully fetched {len(data)} data points for {ticker}",
                    )
                    # 最初と最後の日付をログに出力
                    if len(data) > 0:
                        logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
                    return data
                logger.warning(
                    f"No data found for ticker: {ticker} (async, attempt {retry_count + 1}/{max_retries})",
                )
                # 次のリトライのために少し待機
                await asyncio.sleep(0.5)
                retry_count += 1
                continue
            except Exception as e:
                last_exception = e
                self._last_yfinance_error = str(e) or repr(e)
                # より詳細なエラー情報をログに出力
                logger.error(
                    f"Failed to fetch data for {ticker} (async): {e!s} (type: {type(e).__name__}), attempt {retry_count + 1}/{max_retries}",
                )
                # 例外の詳細をログに出力
                import traceback

                logger.error(
                    f"Full traceback for {ticker} (async): {traceback.format_exc()}",
                )

                # HTTPエラーか確認（一時的なエラーの可能性）
                if "429" in str(e) or "404" in str(e) or "50" in str(e):
                    logger.warning(
                        f"HTTP error detected, will retry for {ticker} (async)",
                    )
                elif "ConnectionError" in str(type(e)) or "Timeout" in str(e):
                    logger.warning(
                        f"Connection error detected, will retry for {ticker} (async)",
                    )
                else:
                    # その他のエラーの場合はリトライしない
                    logger.error(
                        f"Non-retryable error for {ticker} (async), stopping retries",
                    )
                    raise

                # 次のリトライのために少し待機
                await asyncio.sleep(1.0 * (retry_count + 1))  # 非同期対応のスリープ
                retry_count += 1

        # 全てのリトライが失敗した場合
        logger.error(f"All retry attempts failed for {ticker} (async)")
        if last_exception is not None and not self._last_yfinance_error:
            self._last_yfinance_error = str(last_exception) or repr(last_exception)
        if self._last_yfinance_error is None:
            self._last_yfinance_error = "yfinance returned empty dataset"
        return pd.DataFrame()  # 空のDataFrameを返す

    def _get_sync_provider(self) -> StockDataProvider:
        if self._sync_provider is None:
            self._sync_provider = StockDataProvider()
        return self._sync_provider

    async def _fetch_trusted_source_async(
        self,
        sync_provider: StockDataProvider,
        symbol: str,
        ticker_candidates: List[str],
        period: Optional[str],
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        start_ts = None
        end_ts = None
        config = getattr(sync_provider, "market_data_config", None)

        if config is None:
            if sync_provider._should_use_local_first(symbol):
                return await self._fetch_from_local_async(
                    sync_provider,
                    symbol,
                    period,
                    start_ts,
                    end_ts,
                )
            return pd.DataFrame(), None

        provider_name = getattr(config, "provider", "local_csv").lower()

        if provider_name == "local_csv":
            return await self._fetch_from_local_async(
                sync_provider,
                symbol,
                period,
                start_ts,
                end_ts,
            )

        if provider_name == "http_api":
            return await self._fetch_from_http_async(
                sync_provider,
                symbol,
                ticker_candidates,
                period,
                start=None,
                end=None,
                start_ts=start_ts,
                end_ts=end_ts,
            )

        if provider_name == "hybrid":
            local_data, actual = await self._fetch_from_local_async(
                sync_provider,
                symbol,
                period,
                start_ts,
                end_ts,
            )
            if not local_data.empty:
                return local_data, actual
            return await self._fetch_from_http_async(
                sync_provider,
                symbol,
                ticker_candidates,
                period,
                start=None,
                end=None,
                start_ts=start_ts,
                end_ts=end_ts,
            )

        if sync_provider._should_use_local_first(symbol):
            return await self._fetch_from_local_async(
                sync_provider,
                symbol,
                period,
                start_ts,
                end_ts,
            )
        return pd.DataFrame(), None

    async def _fetch_from_local_async(
        self,
        sync_provider: StockDataProvider,
        symbol: str,
        period: Optional[str],
        start_ts: Optional[pd.Timestamp],
        end_ts: Optional[pd.Timestamp],
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            sync_provider._fetch_from_local_csv,
            symbol,
            period,
            start_ts,
            end_ts,
        )

    async def _fetch_from_http_async(
        self,
        sync_provider: StockDataProvider,
        symbol: str,
        ticker_candidates: List[str],
        period: Optional[str],
        start: Optional[str],
        end: Optional[str],
        start_ts: Optional[pd.Timestamp],
        end_ts: Optional[pd.Timestamp],
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            sync_provider._fetch_from_http_api,
            symbol,
            ticker_candidates,
            period,
            start,
            end,
            start_ts,
            end_ts,
        )

    async def get_multiple_stocks(
        self,
        symbols: List[str],
        period: str = "1y",
    ) -> Dict[str, pd.DataFrame]:
        """Asynchronously fetch data for multiple stocks"""
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
        self,
        symbol: str,
        period: str,
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
        self,
        data: pd.DataFrame,
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
