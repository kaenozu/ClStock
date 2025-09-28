<<<<<<< Updated upstream
<<<<<<< HEAD
try:
        try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
    except ModuleNotFoundError:  # pragma: no cover - fallback for test environments
        import pandas as pd  # type: ignore

        class _DummyTicker:
            def __init__(self, *_, **__):
                pass

            def history(self, *_, **__):
                return pd.DataFrame()

        class _DummyYFinance:
            def Ticker(self, *args, **kwargs):
                return _DummyTicker()

        yf = _DummyYFinance()  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments
    class _DummyTicker:
        def __init__(self, *_, **__):
            pass

        def history(self, *_, **__):
            import pandas as pd  # type: ignore

            return pd.DataFrame()

    class _DummyYFinance:
        def Ticker(self, *args, **kwargs):
            return _DummyTicker()

    yf = _DummyYFinance()  # type: ignore
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
from utils.logger_config import setup_logger

logger = setup_logger(__name__)


class StockDataProvider:
    def __init__(self):
        from config.settings import get_settings

        settings = get_settings()
        self.jp_stock_codes: Dict[str, str] = settings.target_stocks

    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        from utils.exceptions import DataFetchError, InvalidSymbolError, NetworkError
        from utils.cache import get_cache

        # キャッシュキーを生成
        cache_key = f"stock_data_{symbol}_{period}"
        cache = get_cache()

        # キャッシュから取得を試行
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # 日本株の場合は複数の形式を試行
            if symbol in self.jp_stock_codes:
                ticker_formats = [f"{symbol}.T", f"{symbol}.TO", symbol]
            else:
                ticker_formats = [symbol]

            data = pd.DataFrame()
            successful_ticker = None

            for ticker in ticker_formats:
                logger.info(f"Fetching data for {ticker} (period: {period})")
                try:
                    stock = yf.Ticker(ticker)
                    data = stock.history(period=period)

                    if not data.empty:
                        successful_ticker = ticker
                        logger.info(f"Successfully fetched data using ticker: {ticker}")
                        break
                    else:
                        logger.warning(f"No data found for ticker format: {ticker}")
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {ticker}: {str(e)}")
                    continue

            if data.empty:
                # 最後の手段：より長い期間で試行
                logger.info(f"Trying longer period for {symbol}")
                try:
                    for ticker in ticker_formats:
                        stock = yf.Ticker(ticker)
                        data = stock.history(period="2y")
                        if not data.empty:
                            logger.info(f"Found data with 2y period for {ticker}")
                            successful_ticker = ticker
                            break
                except Exception:
                    pass

            if data.empty:
                # 最終的にデモデータを生成
                logger.warning(
                    f"No real data available for {symbol}, generating demo data"
                )
                data = self._generate_demo_data(symbol, period)
                successful_ticker = "demo_data"

            # データが取得できた場合の処理
            data["Symbol"] = symbol
            data["CompanyName"] = self.jp_stock_codes.get(symbol, symbol)
            data["ActualTicker"] = successful_ticker

            logger.info(
                f"Successfully processed {len(data)} data points for {symbol} using {successful_ticker}"
            )

            # キャッシュに保存（30分）
            cache.set(cache_key, data, ttl=1800)

            return data

        except DataFetchError:
            raise
        except ConnectionError as e:
            raise NetworkError(f"yfinance API for {symbol}", str(e))
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise DataFetchError(symbol, "Unexpected error during data fetch", str(e))

    def _resolve_period_days(self, period: str) -> int:
        period = (period or '').lower()
        mapping = {
            '1d': 1,
            '5d': 5,
            '1wk': 5,
            '1mo': 22,
            '3mo': 66,
            '6mo': 126,
            '1y': 252,
            '2y': 504,
            '5y': 1260,
            '10y': 2520,
        }
        if period in mapping:
            return mapping[period]
        if period.endswith('y') and period[:-1].isdigit():
            return max(1, int(period[:-1]) * 252)
        if period.endswith('mo') and period[:-2].isdigit():
            return max(1, int(period[:-2]) * 22)
        if period == 'max':
            return 2520
        if period == 'ytd':
            start_of_year = datetime(datetime.now().year, 1, 1)
            return max(1, (datetime.now() - start_of_year).days)
        return 120

    def _generate_demo_data(self, symbol: str, period: str) -> pd.DataFrame:
        days = self._resolve_period_days(period)
        end_date = datetime.now()
        index = pd.date_range(end=end_date, periods=days, freq='B')
        base_price = 100 + np.random.rand() * 20
        returns = np.random.normal(loc=0.0005, scale=0.02, size=len(index))
        close = base_price * np.exp(np.cumsum(returns))
        high = close * (1 + np.random.uniform(0.001, 0.01, size=len(index)))
        low = close * (1 - np.random.uniform(0.001, 0.01, size=len(index)))
        open_prices = close * (1 + np.random.uniform(-0.005, 0.005, size=len(index)))
        volume = np.random.randint(10_000, 200_000, size=len(index))

        df = pd.DataFrame(
            {
                'Open': open_prices,
                'High': np.maximum.reduce([open_prices, high, close]),
                'Low': np.minimum.reduce([open_prices, low, close]),
                'Close': close,
                'Adj Close': close,
                'Volume': volume,
            },
            index=index,
        )
        df.index.name = 'Date'
        logger.info(f"Generated demo data for {symbol} with {len(df)} points")
        return df


    def get_multiple_stocks(
        self, symbols: List[str], period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        from utils.exceptions import DataFetchError, InvalidSymbolError

        result: Dict[str, pd.DataFrame] = {}
        failed_symbols: List[str] = []

        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, period)
                if not data.empty:
                    result[symbol] = data
            except DataFetchError as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                failed_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Unexpected error for {symbol}: {e}")
                failed_symbols.append(symbol)

        if failed_symbols:
            logger.info(f"Failed to fetch data for symbols: {failed_symbols}")

        return result

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        from utils.cache import get_cache

        if data.empty:
            return data

        # シンボル情報を含むキャッシュキーを生成
        symbol = data["Symbol"].iloc[0] if "Symbol" in data.columns else "unknown"
        cache_key = f"technical_indicators_{symbol}_{hash(str(data.index.tolist()))}"
        cache = get_cache()

        # キャッシュから取得を試行
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        df = data.copy()

        # 移動平均線（ベクトル化で高速化）
        df["SMA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()

        # RSI（効率的な計算）
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()

        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD（効率的な指数移動平均）
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # ATR（効率的なTrue Range計算）
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = true_range.rolling(window=14, min_periods=1).mean()

        # キャッシュに保存（30分）
        cache.set(cache_key, df, ttl=1800)

        return df

    def get_financial_metrics(self, symbol: str) -> Dict[str, Union[str, int, float]]:
        try:
            if symbol in self.jp_stock_codes:
                ticker = f"{symbol}.T"
            else:
                ticker = symbol

            stock = yf.Ticker(ticker)
            info = stock.info

            metrics: Dict[str, Union[str, int, float]] = {
                "symbol": symbol,
                "company_name": self.jp_stock_codes.get(symbol, symbol),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "pb_ratio": info.get("priceToBook", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "roe": info.get("returnOnEquity", 0),
                "current_price": info.get("currentPrice", 0),
                "target_price": info.get("targetMeanPrice", 0),
                "recommendation": info.get("recommendationMean", 0),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error fetching financial metrics for {symbol}: {str(e)}")
            return {}

    def get_all_stock_symbols(self) -> List[str]:
        return list(self.jp_stock_codes.keys())
=======
try:
        try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
    except ModuleNotFoundError:  # pragma: no cover - fallback for test environments
        import pandas as pd  # type: ignore

        class _DummyTicker:
            def __init__(self, *_, **__):
                pass

            def history(self, *_, **__):
                return pd.DataFrame()

        class _DummyYFinance:
            def Ticker(self, *args, **kwargs):
                return _DummyTicker()

        yf = _DummyYFinance()  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments
    class _DummyTicker:
        def __init__(self, *_, **__):
            pass

        def history(self, *_, **__):
            import pandas as pd  # type: ignore

            return pd.DataFrame()

    class _DummyYFinance:
        def Ticker(self, *args, **kwargs):
            return _DummyTicker()

    yf = _DummyYFinance()  # type: ignore
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
from utils.logger_config import setup_logger

logger = setup_logger(__name__)


class StockDataProvider:
    def __init__(self):
        from config.settings import get_settings

        settings = get_settings()
        self.jp_stock_codes: Dict[str, str] = settings.target_stocks

    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        from utils.exceptions import DataFetchError, InvalidSymbolError, NetworkError
        from utils.cache import get_cache

        # キャッシュキーを生成
        cache_key = f"stock_data_{symbol}_{period}"
        cache = get_cache()

        # キャッシュから取得を試行
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # 日本株の場合は複数の形式を試行
            if symbol in self.jp_stock_codes:
                ticker_formats = [f"{symbol}.T", f"{symbol}.TO", symbol]
            else:
                ticker_formats = [symbol]

            data = pd.DataFrame()
            successful_ticker = None

            for ticker in ticker_formats:
                logger.info(f"Fetching data for {ticker} (period: {period})")
                try:
                    stock = yf.Ticker(ticker)
                    data = stock.history(period=period)

                    if not data.empty:
                        successful_ticker = ticker
                        logger.info(f"Successfully fetched data using ticker: {ticker}")
                        break
                    else:
                        logger.warning(f"No data found for ticker format: {ticker}")
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {ticker}: {str(e)}")
                    continue

            if data.empty:
                # 最後の手段：より長い期間で試行
                logger.info(f"Trying longer period for {symbol}")
                try:
                    for ticker in ticker_formats:
                        stock = yf.Ticker(ticker)
                        data = stock.history(period="2y")
                        if not data.empty:
                            logger.info(f"Found data with 2y period for {ticker}")
                            successful_ticker = ticker
                            break
                except Exception:
                    pass

            if data.empty:
                # 最終的にデモデータを生成
                logger.warning(
                    f"No real data available for {symbol}, generating demo data"
                )
                data = self._generate_demo_data(symbol, period)
                successful_ticker = "demo_data"

            # データが取得できた場合の処理
            data["Symbol"] = symbol
            data["CompanyName"] = self.jp_stock_codes.get(symbol, symbol)
            data["ActualTicker"] = successful_ticker

            logger.info(
                f"Successfully processed {len(data)} data points for {symbol} using {successful_ticker}"
            )

            # キャッシュに保存（30分）
            cache.set(cache_key, data, ttl=1800)

            return data

        except DataFetchError:
            raise
        except ConnectionError as e:
            raise NetworkError(f"yfinance API for {symbol}", str(e))
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise DataFetchError(symbol, "Unexpected error during data fetch", str(e))

    def get_multiple_stocks(
        self, symbols: List[str], period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        from utils.exceptions import DataFetchError, InvalidSymbolError

        result: Dict[str, pd.DataFrame] = {}
        failed_symbols: List[str] = []

        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, period)
                if not data.empty:
                    result[symbol] = data
            except DataFetchError as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                failed_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Unexpected error for {symbol}: {e}")
                failed_symbols.append(symbol)

        if failed_symbols:
            logger.info(f"Failed to fetch data for symbols: {failed_symbols}")

        return result

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        from utils.cache import get_cache

        if data.empty:
            return data

        # シンボル情報を含むキャッシュキーを生成
        symbol = data["Symbol"].iloc[0] if "Symbol" in data.columns else "unknown"
        cache_key = f"technical_indicators_{symbol}_{hash(str(data.index.tolist()))}"
        cache = get_cache()

        # キャッシュから取得を試行
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        df = data.copy()

        # 移動平均線（ベクトル化で高速化）
        df["SMA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()

        # RSI（効率的な計算）
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()

        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD（効率的な指数移動平均）
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # ATR（効率的なTrue Range計算）
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = true_range.rolling(window=14, min_periods=1).mean()

        # キャッシュに保存（30分）
        cache.set(cache_key, df, ttl=1800)

        return df

    def get_financial_metrics(self, symbol: str) -> Dict[str, Union[str, int, float]]:
        try:
            if symbol in self.jp_stock_codes:
                ticker = f"{symbol}.T"
            else:
                ticker = symbol

            stock = yf.Ticker(ticker)
            info = stock.info

            metrics: Dict[str, Union[str, int, float]] = {
                "symbol": symbol,
                "company_name": self.jp_stock_codes.get(symbol, symbol),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "pb_ratio": info.get("priceToBook", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "roe": info.get("returnOnEquity", 0),
                "current_price": info.get("currentPrice", 0),
                "target_price": info.get("targetMeanPrice", 0),
                "recommendation": info.get("recommendationMean", 0),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error fetching financial metrics for {symbol}: {str(e)}")
            return {}

    def get_all_stock_symbols(self) -> List[str]:
        return list(self.jp_stock_codes.keys())
>>>>>>> origin/codex/fix-ci-errors-in-pull-request
=======
﻿import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
    except ModuleNotFoundError:  # pragma: no cover - fallback for test environments
        import pandas as pd  # type: ignore

        class _DummyTicker:
            def __init__(self, *_, **__):
                pass

            def history(self, *_, **__):
                return pd.DataFrame()

        class _DummyYFinance:
            def Ticker(self, *args, **kwargs):
                return _DummyTicker()

        yf = _DummyYFinance()  # type: ignore

from utils.logger_config import setup_logger
from utils.cache import get_cache


logger = setup_logger(__name__)


class StockDataProvider:
    def __init__(self):
        from config.settings import get_settings

        settings = get_settings()
        self.jp_stock_codes: Dict[str, str] = settings.target_stocks
        self.yfinance_failure_log: set = set()  # yfinance縺ｧ蜿門ｾ励↓螟ｱ謨励＠縺滄釜譟・ｒ險倬鹸

        # yfinance逕ｨ繧ｻ繝・す繝ｧ繝ｳ・・ser-Agent繧剃ｸ頑嶌縺搾ｼ・
        import requests

        user_agent = os.getenv(
            "CLSTOCK_YF_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        )
        self.yf_session = requests.Session()
        self.yf_session.headers.update({"User-Agent": user_agent})  # yfinance縺ｧ蜿門ｾ励↓螟ｱ謨励＠縺滄釜譟・ｒ險倬鹸

    def _ticker_formats(self, symbol: str) -> List[str]:
        if symbol in self.jp_stock_codes:
            return [f"{symbol}.T", f"{symbol}.TO", symbol]
        return [symbol]

    def _csv_candidates(self, symbol: str, ticker_formats: List[str]) -> List[Path]:
        candidate_names: List[str] = [
            f"stock_{symbol}.csv",
            f"stock_{symbol}.T.csv",
            f"stock_{symbol}.TO.csv",
        ]

        for ticker in ticker_formats:
            candidate_names.append(f"stock_{ticker}.csv")
            sanitized = ticker.replace(".", "")
            if sanitized and sanitized != ticker:
                candidate_names.append(f"stock_{sanitized}.csv")

        unique_candidates: List[Path] = []
        seen_keys: set[str] = set()
        for name in candidate_names:
            candidate = (Path("data") / name)
            key = str(candidate.resolve(strict=False))
            if key in seen_keys:
                continue
            unique_candidates.append(candidate)
            seen_keys.add(key)
        return unique_candidates

    def _load_first_available_csv(
        self,
        symbol: str,
        candidates: List[Path],
    ) -> Optional[Tuple[pd.DataFrame, str]]:
        for candidate in candidates:
            if candidate.exists():
                logger.info("Loading data for %s from %s", symbol, candidate)
                csv_data = pd.read_csv(candidate, index_col="Date", parse_dates=True)
                candidate_ticker = candidate.stem.replace("stock_", "")

                if "Symbol" not in csv_data.columns:
                    csv_data["Symbol"] = symbol
                if "CompanyName" not in csv_data.columns:
                    csv_data["CompanyName"] = self.jp_stock_codes.get(symbol, symbol)
                if "ActualTicker" not in csv_data.columns:
                    csv_data["ActualTicker"] = candidate_ticker

                logger.info(
                    "Successfully loaded %d data points for %s from CSV (%s).",
                    len(csv_data),
                    symbol,
                    candidate.name,
                )
                return csv_data, candidate_ticker
        return None

    def _should_use_local_first(self, candidates: List[Path]) -> bool:
        prefer_local_setting = os.getenv("CLSTOCK_PREFER_LOCAL_DATA", "auto").strip().lower()
        if prefer_local_setting in {"1", "true", "yes"}:
            return True
        if prefer_local_setting in {"0", "false", "no"}:
            return False
        return any(candidate.exists() for candidate in candidates)

    def _download_single_ticker(self, ticker: str, period: str) -> pd.DataFrame:
        import time

        max_retries = 3
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker, session=self.yf_session)
                time.sleep(0.1)
                data = stock.history(period=period)
                if not data.empty:
                    logger.info(
                        "Successfully fetched %d data points for %s (period=%s)",
                        len(data),
                        ticker,
                        period,
                    )
                    logger.info("Date range: %s to %s", data.index[0], data.index[-1])
                    return data

                logger.warning(
                    "No data found for ticker format: %s (attempt %d/%d)",
                    ticker,
                    attempt + 1,
                    max_retries,
                )
            except Exception as exc:
                import traceback

                logger.error(
                    "Failed to fetch data for %s: %s (type: %s), attempt %d/%d",
                    ticker,
                    exc,
                    type(exc).__name__,
                    attempt + 1,
                    max_retries,
                )
                logger.error("Full traceback for %s: %s", ticker, traceback.format_exc())

                error_text = str(exc)
                if any(code in error_text for code in ("429", "404", "50")):
                    logger.warning("HTTP error detected, will retry for %s", ticker)
                elif "ConnectionError" in error_text or "Timeout" in error_text:
                    logger.warning("Connection error detected, will retry for %s", ticker)
                else:
                    logger.error("Non-retryable error for %s, stopping retries", ticker)
                    break

            time.sleep(0.5 * (attempt + 1))

        self.yfinance_failure_log.add(ticker)
        return pd.DataFrame()

    def _download_via_yfinance(
        self,
        symbol: str,
        ticker_formats: List[str],
        period: str,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        for ticker in ticker_formats:
            logger.info("Fetching data for %s (period: %s)", ticker, period)
            data = self._download_single_ticker(ticker, period)
            if not data.empty:
                return data, ticker
            logger.warning("All retry attempts failed for ticker format: %s", ticker)

        logger.info("Trying longer period for %s", symbol)
        for ticker in ticker_formats:
            data = self._download_single_ticker(ticker, "2y")
            if not data.empty:
                logger.info("Found data with 2y period for %s", ticker)
                return data, ticker

        return pd.DataFrame(), None

    def get_all_tickers(self):
        """Return configured ticker metadata."""

        class TickerInfo:
            def __init__(self, symbol: str, name: str) -> None:
                self.symbol = symbol
                self.name = name

        return [TickerInfo(code, name) for code, name in self.jp_stock_codes.items()]

    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        from utils.exceptions import DataFetchError, InvalidSymbolError, NetworkError


        cache_key = f"stock_data_{symbol}_{period}"
        cache = get_cache()

        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            ticker_formats = self._ticker_formats(symbol)
            csv_candidates = self._csv_candidates(symbol, ticker_formats)

            data = pd.DataFrame()
            successful_ticker: Optional[str] = None

            if self._should_use_local_first(csv_candidates):
                loaded = self._load_first_available_csv(symbol, csv_candidates)
                if loaded is not None:
                    data, successful_ticker = loaded

            if data.empty:
                data, successful_ticker = self._download_via_yfinance(symbol, ticker_formats, period)

            if data.empty:
                loaded = self._load_first_available_csv(symbol, csv_candidates)
                if loaded is not None:
                    data, successful_ticker = loaded

            if data.empty:
                logger.error(
                    "No real data available for %s after trying multiple ticker formats, periods, and CSV files.",
                    symbol,
                )
                raise ValueError(
                    f"Failed to retrieve data for {symbol} from yfinance and CSV files."
                )

            if "Symbol" not in data.columns:
                data["Symbol"] = symbol
            if "CompanyName" not in data.columns:
                data["CompanyName"] = self.jp_stock_codes.get(symbol, symbol)
            if "ActualTicker" not in data.columns:
                fallback = ticker_formats[0] if ticker_formats else symbol
                data["ActualTicker"] = successful_ticker if successful_ticker else fallback

            logger.info(
                "Successfully processed %d data points for %s using %s",
                len(data),
                symbol,
                "CSV" if successful_ticker is None else successful_ticker,
            )

            cache.set(cache_key, data, ttl=1800)
            return data

        except DataFetchError:
            raise
        except ConnectionError as exc:
            raise NetworkError(f"yfinance API for {symbol}", str(exc))
        except Exception as exc:
            logger.error("Error fetching data for %s: %s", symbol, exc)
            raise DataFetchError(symbol, "Unexpected error during data fetch", str(exc))

    def _resolve_period_days(self, period: str) -> int:
        period = (period or '').lower()
        mapping = {
            '1d': 1,
            '5d': 5,
            '1wk': 5,
            '1mo': 22,
            '3mo': 66,
            '6mo': 126,
            '1y': 252,
            '2y': 504,
            '5y': 1260,
            '10y': 2520,
        }
        if period in mapping:
            return mapping[period]
        if period.endswith('y') and period[:-1].isdigit():
            return max(1, int(period[:-1]) * 252)
        if period.endswith('mo') and period[:-2].isdigit():
            return max(1, int(period[:-2]) * 22)
        if period == 'max':
            return 2520
        if period == 'ytd':
            start_of_year = datetime(datetime.now().year, 1, 1)
            return max(1, (datetime.now() - start_of_year).days)
        return 120

    def _generate_demo_data(self, symbol: str, period: str) -> pd.DataFrame:
        days = self._resolve_period_days(period)
        end_date = datetime.now()
        index = pd.date_range(end=end_date, periods=days, freq='B')
        base_price = 100 + np.random.rand() * 20
        returns = np.random.normal(loc=0.0005, scale=0.02, size=len(index))
        close = base_price * np.exp(np.cumsum(returns))
        high = close * (1 + np.random.uniform(0.001, 0.01, size=len(index)))
        low = close * (1 - np.random.uniform(0.001, 0.01, size=len(index)))
        open_prices = close * (1 + np.random.uniform(-0.005, 0.005, size=len(index)))
        volume = np.random.randint(10_000, 200_000, size=len(index))

        df = pd.DataFrame(
            {
                'Open': open_prices,
                'High': np.maximum.reduce([open_prices, high, close]),
                'Low': np.minimum.reduce([open_prices, low, close]),
                'Close': close,
                'Adj Close': close,
                'Volume': volume,
            },
            index=index,
        )
        df.index.name = 'Date'
        logger.info(f"Generated demo data for {symbol} with {len(df)} points")
        return df


    def get_multiple_stocks(
        self, symbols: List[str], period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        from utils.exceptions import DataFetchError, InvalidSymbolError

        result: Dict[str, pd.DataFrame] = {}
        failed_symbols: List[str] = []

        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, period)
                if not data.empty:
                    result[symbol] = data
            except ValueError as e:
                # get_stock_data縺ｧ繝・・繧ｿ縺瑚ｦ九▽縺九ｉ縺ｪ縺・ｴ蜷医・ValueError繧貞・逅・
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                failed_symbols.append(symbol)
            except DataFetchError as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                failed_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Unexpected error for {symbol}: {e}")
                failed_symbols.append(symbol)

        if failed_symbols:
            logger.info(f"Failed to fetch data for symbols: {failed_symbols}")

        return result

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:



        if data.empty:
            return data

        # 繧ｷ繝ｳ繝懊Ν諠・ｱ繧貞性繧繧ｭ繝｣繝・す繝･繧ｭ繝ｼ繧堤函謌・
        symbol = data["Symbol"].iloc[0] if "Symbol" in data.columns else "unknown"
        cache_key = f"technical_indicators_{symbol}_{hash(str(data.index.tolist()))}"
        cache = get_cache()

        # 繧ｭ繝｣繝・す繝･縺九ｉ蜿門ｾ励ｒ隧ｦ陦・
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        df = data.copy()

        # 遘ｻ蜍募ｹｳ蝮・ｷ夲ｼ医・繧ｯ繝医Ν蛹悶〒鬮倬溷喧・・
        df["SMA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()

        # RSI・亥柑邇・噪縺ｪ險育ｮ暦ｼ・
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()

        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD・亥柑邇・噪縺ｪ謖・焚遘ｻ蜍募ｹｳ蝮・ｼ・
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # ATR・亥柑邇・噪縺ｪTrue Range險育ｮ暦ｼ・
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = true_range.rolling(window=14, min_periods=1).mean()

        # 繧ｭ繝｣繝・す繝･縺ｫ菫晏ｭ假ｼ・0蛻・ｼ・
        cache.set(cache_key, df, ttl=1800)

        return df

    def get_financial_metrics(self, symbol: str) -> Dict[str, Union[str, int, float]]:
        max_retries = 3
        retry_count = 0
        
        if symbol in self.jp_stock_codes:
            ticker = f"{symbol}.T"
        else:
            ticker = symbol

        while retry_count < max_retries:
            try:
                import time
                # 繝ｪ繧ｯ繧ｨ繧ｹ繝亥燕縺ｫ蟆代＠蠕・ｩ滂ｼ・PI雋闕ｷ霆ｽ貂帙・縺溘ａ・・
                time.sleep(0.1)
                
                stock = yf.Ticker(ticker, session=self.yf_session)
                info = stock.info

                metrics: Dict[str, Union[str, int, float]] = {
                    "symbol": symbol,
                    "company_name": self.jp_stock_codes.get(symbol, symbol),
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE", 0),
                    "pb_ratio": info.get("priceToBook", 0),
                    "dividend_yield": info.get("dividendYield", 0),
                    "roe": info.get("returnOnEquity", 0),
                    "current_price": info.get("currentPrice", 0),
                    "target_price": info.get("targetMeanPrice", 0),
                    "recommendation": info.get("recommendationMean", 0),
                }

                return metrics

            except Exception as e:
                logger.error(f"Error fetching financial metrics for {symbol} (attempt {retry_count + 1}/{max_retries}): {str(e)}")
                
                # HTTP繧ｨ繝ｩ繝ｼ縺狗｢ｺ隱・
                if "429" in str(e) or "404" in str(e) or "50" in str(e):
                    logger.warning(f"HTTP error detected, will retry for {symbol}")
                elif "ConnectionError" in str(type(e)) or "Timeout" in str(e):
                    logger.warning(f"Connection error detected, will retry for {symbol}")
                else:
                    # 縺昴・莉悶・繧ｨ繝ｩ繝ｼ縺ｮ蝣ｴ蜷医・繝ｪ繝医Λ繧､縺励↑縺・
                    logger.error(f"Non-retryable error for {symbol}, stopping retries")
                    break
                
                # 谺｡縺ｮ繝ｪ繝医Λ繧､縺ｮ縺溘ａ縺ｫ蟆代＠蠕・ｩ・
                time.sleep(1.0 * (retry_count + 1))
                retry_count += 1

        # 蜈ｨ縺ｦ縺ｮ繝ｪ繝医Λ繧､縺悟､ｱ謨励＠縺溷ｴ蜷・
        logger.error(f"All retry attempts failed for financial metrics of {symbol}")
        return {}

    def get_all_stock_symbols(self) -> List[str]:
        return list(self.jp_stock_codes.keys())



>>>>>>> Stashed changes
