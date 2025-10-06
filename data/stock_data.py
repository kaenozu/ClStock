"""Stock market data utilities used throughout the project.

The provider integrates with the configurable ``MarketDataConfig`` settings to
prioritise trusted market data sources such as local CSV caches or approved
HTTP APIs.  When those sources are unavailable it can fall back to live
``yfinance`` requests if the dependency is installed.  Pseudo-random fallback
data generation has been removed to ensure production runs always use real
market data.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from config.settings import get_settings
from utils.cache import get_cache
from utils.exceptions import BatchDataFetchError, DataFetchError

logger = logging.getLogger(__name__)


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


def _normalize_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value)
    if isinstance(ts, pd.DatetimeIndex):
        ts = ts[0]
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


try:  # pragma: no cover - exercised indirectly through tests
    import yfinance as yf  # type: ignore

    YFINANCE_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - hit in constrained envs
    yf = None  # type: ignore
    YFINANCE_AVAILABLE = False


class StockDataProvider:
    """Fetch and enrich historical stock data.

    The provider favours cached and local data when available but can fall back
    to ``yfinance`` for live downloads.  It also exposes helper utilities for
    technical indicators and basic financial metrics.
    """

    def __init__(self) -> None:
        settings = get_settings()
        # Always keep codes sorted for deterministic behaviour
        self.jp_stock_codes: Dict[str, str] = dict(
            sorted(settings.target_stocks.items()),
        )
        self.cache_ttl = int(os.getenv("CLSTOCK_STOCK_CACHE_TTL", "1800"))
        default_data_dir = Path(
            os.getenv("CLSTOCK_DATA_DIR", Path(__file__).resolve().parent),
        )
        default_history = default_data_dir / "historical"

        market_config = getattr(settings, "market_data", None)
        configured_dirs: List[Path] = []
        if market_config is not None:
            if getattr(market_config, "local_cache_dir", None):
                configured_dirs.append(Path(market_config.local_cache_dir))
            for extra in getattr(market_config, "extra_cache_dirs", []):
                configured_dirs.append(Path(extra))

        portfolio_parent = Path(settings.database.personal_portfolio_db).parent
        candidate_dirs: List[Path] = []
        candidate_dirs.extend(configured_dirs)
        candidate_dirs.append(default_history)
        candidate_dirs.append(portfolio_parent)

        self._history_dirs = self._deduplicate_existing_paths(candidate_dirs)
        self.market_data_config = market_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_all_stock_symbols(self) -> List[str]:
        """Return all supported stock symbols."""
        return list(self.jp_stock_codes.keys())

    def get_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for *symbol*."""
        if start is not None or end is not None:
            cache_key = f"stock::{symbol}::start_{start}::end_{end}"
        else:
            cache_key = f"stock::{symbol}::{period}"
        cache = get_cache()
        cached = cache.get(cache_key)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached

        data: Optional[pd.DataFrame] = None
        actual_ticker: Optional[str] = None
        ticker_candidates = self._ticker_formats(symbol)

        try:
            data, actual_ticker = self._fetch_trusted_source(
                symbol, ticker_candidates, period, start, end,
            )
        except DataFetchError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Trusted data source failed for %s: %s", symbol, exc)

        if (data is None or data.empty) and YFINANCE_AVAILABLE:
            if start is not None or end is not None:
                data, actual_ticker = self._download_via_yfinance(
                    symbol, period=None, start=start, end=end,
                )
            else:
                data, actual_ticker = self._download_via_yfinance(symbol, period)

        if data is None or data.empty:
            raise DataFetchError(symbol, "No historical data available")

        if start is not None and end is not None:
            actual_start = data.index.min()
            actual_end = data.index.max()
            logger.debug(
                "StockDataProvider window for %s: requested %s-%s, actual %s-%s",
                symbol,
                start,
                end,
                actual_start,
                actual_end,
            )

        prepared = self._prepare_history_frame(data, symbol, actual_ticker)
        cache.set(cache_key, prepared, ttl=self.cache_ttl)
        return prepared

    def get_multiple_stocks(
        self, symbols: Iterable[str], period: str = "1y",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols, raising on missing data."""
        results: Dict[str, pd.DataFrame] = {}
        failures: Dict[str, DataFetchError] = {}

        for symbol in symbols:
            try:
                results[symbol] = self.get_stock_data(symbol, period)
            except DataFetchError as exc:
                failures[symbol] = exc

        if failures:
            details = "; ".join(str(exc) for exc in failures.values())
            raise BatchDataFetchError(
                failed_symbols=list(failures.keys()),
                partial_results=results,
                details=details or None,
            )

        return results

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

        if not YFINANCE_AVAILABLE or yf is None:
            cache.set(cache_key, metrics, ttl=self.cache_ttl)
            return metrics

        for ticker in self._ticker_formats(symbol):
            try:
                if self._fetch_financial_metrics_via_yfinance(ticker, metrics):
                    break
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "Failed to retrieve financial metrics for %s via %s: %s",
                    symbol,
                    ticker,
                    exc,
                )
                continue

        cache.set(cache_key, metrics, ttl=self.cache_ttl)
        return metrics

    def _fetch_financial_metrics_via_yfinance(
        self, ticker: str, metrics: Dict[str, Optional[float]],
    ) -> bool:
        """Populate *metrics* using ``yfinance`` for the given *ticker*.

        The helper guards against missing dependencies or partial responses and
        returns ``True`` only when some data was successfully retrieved.
        """
        if not YFINANCE_AVAILABLE or yf is None:
            return False

        try:
            ticker_obj = yf.Ticker(ticker)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Failed to instantiate yfinance ticker %s: %s", ticker, exc)
            return False

        info = getattr(ticker_obj, "info", {}) or {}
        fast = getattr(ticker_obj, "fast_info", None)

        has_data = False

        market_cap = info.get("marketCap")
        if market_cap is not None:
            metrics["market_cap"] = market_cap
            has_data = True

        pe_ratio = info.get("forwardPE") or info.get("trailingPE")
        if pe_ratio is not None:
            metrics["pe_ratio"] = pe_ratio
            has_data = True

        dividend_yield = info.get("dividendYield")
        if dividend_yield is not None:
            metrics["dividend_yield"] = dividend_yield
            has_data = True

        beta = info.get("beta")
        if beta is not None:
            metrics["beta"] = beta
            has_data = True

        if fast is not None:
            last_price = getattr(fast, "last_price", None)
            if last_price is not None:
                metrics["last_price"] = last_price
                has_data = True

            previous_close = getattr(fast, "previous_close", None)
            if previous_close is not None:
                metrics["previous_close"] = previous_close
                has_data = True

            ten_day_average_volume = getattr(fast, "ten_day_average_volume", None)
            if ten_day_average_volume is not None:
                metrics["ten_day_average_volume"] = ten_day_average_volume
                has_data = True

        if has_data:
            metrics["actual_ticker"] = ticker

        return has_data

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _fetch_trusted_source(
        self,
        symbol: str,
        ticker_candidates: Sequence[str],
        period: Optional[str],
        start: Optional[str],
        end: Optional[str],
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        config = getattr(self, "market_data_config", None)
        start_ts = _normalize_timestamp(start)
        end_ts = _normalize_timestamp(end)

        if config is None:
            if self._should_use_local_first(symbol):
                return self._fetch_from_local_csv(symbol, period, start_ts, end_ts)
            return pd.DataFrame(), None

        provider = getattr(config, "provider", "local_csv").lower()

        if provider == "local_csv":
            return self._fetch_from_local_csv(symbol, period, start_ts, end_ts)

        if provider == "http_api":
            return self._fetch_from_http_api(
                symbol, ticker_candidates, period, start, end, start_ts, end_ts,
            )

        if provider == "hybrid":
            local_data, actual = self._fetch_from_local_csv(
                symbol, period, start_ts, end_ts,
            )
            if not local_data.empty:
                return local_data, actual
            return self._fetch_from_http_api(
                symbol, ticker_candidates, period, start, end, start_ts, end_ts,
            )

        if self._should_use_local_first(symbol):
            return self._fetch_from_local_csv(symbol, period, start_ts, end_ts)
        return pd.DataFrame(), None

    def _fetch_from_local_csv(
        self,
        symbol: str,
        period: Optional[str],
        start_ts: Optional[pd.Timestamp],
        end_ts: Optional[pd.Timestamp],
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        local = self._load_first_available_csv(symbol)
        if local is None:
            return pd.DataFrame(), None
        data, actual_ticker = local
        if data.empty:
            return data.copy(), actual_ticker
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        data.sort_index(inplace=True)
        trimmed = self._trim_time_window(data, period, start_ts, end_ts)
        return trimmed, actual_ticker

    def _fetch_from_http_api(
        self,
        symbol: str,
        ticker_candidates: Sequence[str],
        period: Optional[str],
        start: Optional[str],
        end: Optional[str],
        start_ts: Optional[pd.Timestamp],
        end_ts: Optional[pd.Timestamp],
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        config = getattr(self, "market_data_config", None)
        if config is None or not getattr(config, "api_base_url", None):
            return pd.DataFrame(), None
        try:
            import requests  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
            raise DataFetchError(
                symbol,
                "requests library is required for HTTP market data retrieval",
                str(exc),
            ) from exc

        headers: Dict[str, str] = {}
        if getattr(config, "api_token", None):
            headers["Authorization"] = f"Bearer {config.api_token}"
        if getattr(config, "api_key", None):
            headers["X-API-KEY"] = config.api_key
        if getattr(config, "api_secret", None):
            headers["X-API-SECRET"] = config.api_secret

        base_url = str(config.api_base_url).rstrip("/")
        request_errors: List[str] = []
        format_errors: List[str] = []

        for ticker in ticker_candidates:
            params = {"symbol": ticker}
            if period:
                params["period"] = period
            if start:
                params["start"] = start
            if end:
                params["end"] = end

            try:
                response = requests.get(
                    f"{base_url}/historical",
                    params=params,
                    headers=headers,
                    timeout=getattr(config, "api_timeout", 10.0),
                    verify=getattr(config, "verify_ssl", True),
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:  # pragma: no cover - network errors
                request_errors.append(f"{ticker}: {exc}")
                continue

            records = payload.get("data") if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                format_errors.append(f"{ticker}: unexpected response format")
                continue

            frame = pd.DataFrame.from_records(records)
            if frame.empty:
                continue

            if "Date" in frame.columns:
                frame.index = pd.to_datetime(frame.pop("Date"))
            elif "date" in frame.columns:
                frame.index = pd.to_datetime(frame.pop("date"))
            else:
                frame.index = pd.to_datetime(frame.index)

            frame.sort_index(inplace=True)
            frame = self._trim_time_window(frame, period, start_ts, end_ts)
            if frame.empty:
                continue

            return frame, ticker

        if request_errors:
            raise DataFetchError(
                symbol,
                "Trusted market data request failed",
                "; ".join(request_errors),
            )

        provider = getattr(config, "provider", "local_csv").lower()
        if format_errors and provider == "http_api":
            raise DataFetchError(
                symbol,
                "Trusted market data response format invalid",
                "; ".join(format_errors),
            )

        return pd.DataFrame(), None

    def _trim_time_window(
        self,
        data: pd.DataFrame,
        period: Optional[str],
        start_ts: Optional[pd.Timestamp],
        end_ts: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        if data.empty:
            return data.copy()

        result = data.copy()
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.to_datetime(result.index)

        if start_ts is not None:
            result = result[result.index >= start_ts]
        if end_ts is not None:
            result = result[result.index <= end_ts]

        if period and start_ts is None and end_ts is None and not result.empty:
            days = _period_to_days(period)
            if days > 0:
                cutoff = result.index.max() - pd.to_timedelta(days, unit="D")
                result = result[result.index >= cutoff]

        return result

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

    def _deduplicate_existing_paths(self, directories: Sequence[Path]) -> List[Path]:
        seen = set()
        existing: List[Path] = []
        for directory in directories:
            if directory is None:
                continue
            path = Path(directory).expanduser().resolve()
            if path in seen:
                continue
            seen.add(path)
            if path.exists():
                existing.append(path)
        return existing

    def _should_use_local_first(
        self, symbol: str,
    ) -> bool:  # pragma: no cover - logic is trivial
        prefer_local = os.getenv("CLSTOCK_PREFER_LOCAL_DATA", "auto").lower()
        if prefer_local in {"1", "true", "yes"}:
            return True
        if prefer_local in {"0", "false", "no"}:
            return False

        config = getattr(self, "market_data_config", None)
        if config is not None and getattr(config, "provider", "local_csv").lower() in {
            "local_csv",
            "hybrid",
        }:
            return True
        return False

    def _load_first_available_csv(
        self, symbol: str,
    ) -> Optional[Tuple[pd.DataFrame, str]]:
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

    def _download_via_yfinance(
        self,
        symbol: str,
        period: Optional[str] = "1y",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        if not YFINANCE_AVAILABLE or yf is None:
            raise DataFetchError(
                symbol,
                "yfinance is not available for live market data retrieval",
            )
        max_attempts = 3
        base_backoff = 0.5
        last_error: Optional[Exception] = None
        attempt_messages: List[str] = []

        for ticker in self._ticker_formats(symbol):
            for attempt in range(1, max_attempts + 1):
                try:
                    ticker_obj = yf.Ticker(ticker)
                    if start is not None or end is not None:
                        history = ticker_obj.history(start=start, end=end)
                    else:
                        history = ticker_obj.history(period=period)
                    if isinstance(history, pd.DataFrame) and not history.empty:
                        return history, ticker
                    logger.debug(
                        "yfinance returned empty history for %s via %s (attempt %d/%d)",
                        symbol,
                        ticker,
                        attempt,
                        max_attempts,
                    )
                except Exception as exc:  # pragma: no cover - depends on yfinance
                    last_error = exc
                    attempt_messages.append(f"{ticker} attempt {attempt}: {exc}")
                    logger.debug(
                        "yfinance attempt %d/%d for %s via %s failed: %s",
                        attempt,
                        max_attempts,
                        symbol,
                        ticker,
                        exc,
                    )
                else:
                    # Empty history without exception should not trigger a retry delay.
                    continue

                if attempt < max_attempts:
                    backoff = base_backoff * (2 ** (attempt - 1))
                    logger.debug(
                        "Retrying yfinance download for %s via %s in %.2fs",
                        symbol,
                        ticker,
                        backoff,
                    )
                    time.sleep(backoff)

            logger.debug(
                "Moving to next ticker candidate for %s after %s attempts",
                symbol,
                ticker,
            )

        if last_error is not None:
            aggregated = "; ".join(attempt_messages) or str(last_error)
            logger.warning(
                "All yfinance attempts failed for %s after %d retries: %s",
                symbol,
                max(len(attempt_messages), 1),
                aggregated,
            )
            raise DataFetchError(
                symbol,
                "Failed to download data via yfinance",
                str(last_error),
            )

        raise DataFetchError(
            symbol,
            "yfinance returned no data",
        )

    def _prepare_history_frame(
        self, data: pd.DataFrame, symbol: str, actual_ticker: Optional[str],
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

    # ------------------------------------------------------------------
    # Extended data methods
    # ------------------------------------------------------------------
    def get_extended_financial_metrics(self, symbol: str) -> Dict[str, Optional[float]]:
        """Return a dictionary with extended financial metrics."""
        # 既存の get_financial_metrics をベースに、infoから追加のデータを取得
        base_metrics = self.get_financial_metrics(symbol)
        actual_ticker = base_metrics.get("actual_ticker")

        if not actual_ticker:
            # tickerが見つからなかった場合は、Noneを返す
            return {}

        cache_key = f"extended_financial::{symbol}"
        cache = get_cache()
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            return cached

        extended_metrics = base_metrics.copy()

        if not YFINANCE_AVAILABLE or yf is None:
            cache.set(cache_key, extended_metrics, ttl=self.cache_ttl)
            return extended_metrics

        for ticker in self._ticker_formats(symbol):
            try:
                ticker_obj = yf.Ticker(ticker)
                info = getattr(ticker_obj, "info", {}) or {}

                # 基本的な財務指標以外のデータを取得
                # 例: revenue, grossProfits, operatingCashflow, ebitda, totalDebt, totalCash, debtToEquity
                # 他にも、ROE, ROA, ROICなどの計算に必要な生データも取得
                keys_to_fetch = [
                    "revenue",
                    "grossProfits",
                    "operatingCashflow",
                    "ebitda",
                    "totalDebt",
                    "totalCash",
                    "debtToEquity",
                    "returnOnAssets",
                    "returnOnEquity",
                    "earningsQuarterlyGrowth",
                    "quarterlyEarningsGrowth",
                    "quarterlyRevenueGrowth",
                    # 必要に応じて他のキーも追加
                ]

                for key in keys_to_fetch:
                    value = info.get(key)
                    if value is not None:
                        # value は数値または文字列の可能性があるため、数値に変換を試みる
                        try:
                            if isinstance(value, str):
                                # 文字列が '1.23M', '45.67B' のような形式の場合、数値化が必要
                                # 今回は数値として解釈できない場合はNoneのままとする
                                float_value = float(value)
                                extended_metrics[key] = float_value
                            elif isinstance(value, (int, float)):
                                extended_metrics[key] = float(value)
                            else:
                                extended_metrics[key] = None
                        except (ValueError, TypeError):
                            extended_metrics[key] = None
                    else:
                        extended_metrics[key] = None

                # データが取得できたtickerを記録
                if (
                    extended_metrics.get("revenue") is not None
                    or extended_metrics.get("totalCash") is not None
                ):
                    extended_metrics["actual_ticker"] = ticker
                    break
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "Failed to retrieve extended financial metrics for %s via %s: %s",
                    symbol,
                    ticker,
                    exc,
                )
                continue

        cache.set(cache_key, extended_metrics, ttl=self.cache_ttl)
        return extended_metrics

    def get_dividend_data(self, symbol: str) -> Dict[str, Optional[float]]:
        """Return a dictionary with dividend-related data."""
        cache_key = f"dividend::{symbol}"
        cache = get_cache()
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            return cached

        dividend_data: Dict[str, Optional[float]] = {
            "symbol": symbol,
            "dividend_rate": None,  # dividendRate
            "dividend_yield": None,  # dividendYield (already in get_financial_metrics but included here for completeness)
            "ex_dividend_date": None,  # exDividendDate (date string)
            "actual_ticker": None,
        }

        if not YFINANCE_AVAILABLE or yf is None:
            cache.set(cache_key, dividend_data, ttl=self.cache_ttl)
            return dividend_data

        for ticker in self._ticker_formats(symbol):
            try:
                ticker_obj = yf.Ticker(ticker)
                info = getattr(ticker_obj, "info", {}) or {}

                # dividendRate, dividendYield, exDividendDate を取得
                # exDividendDate は文字列で返ってくる可能性があるため、str型も考慮
                dividend_data["dividend_rate"] = info.get("dividendRate")
                dividend_data["dividend_yield"] = info.get("dividendYield")
                ex_date_raw = info.get("exDividendDate")
                if ex_date_raw:
                    try:
                        # exDividendDate が数値 (timestamp) か文字列 (YYYY-MM-DD) か確認
                        if isinstance(ex_date_raw, (int, float)):
                            dividend_data["ex_dividend_date"] = (
                                pd.to_datetime(ex_date_raw, unit="s").date().isoformat()
                            )
                        elif isinstance(ex_date_raw, str):
                            # strの場合は、既に YYYY-MM-DD 形式であると仮定
                            dividend_data["ex_dividend_date"] = ex_date_raw
                        else:
                            dividend_data["ex_dividend_date"] = None
                    except Exception:
                        dividend_data["ex_dividend_date"] = None
                else:
                    dividend_data["ex_dividend_date"] = None

                dividend_data["actual_ticker"] = ticker
                break
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "Failed to retrieve dividend data for %s via %s: %s",
                    symbol,
                    ticker,
                    exc,
                )
                continue

        cache.set(cache_key, dividend_data, ttl=self.cache_ttl)
        return dividend_data

    def get_news_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Return a list of news articles for a symbol."""
        cache_key = f"news::{symbol}"
        cache = get_cache()
        cached = cache.get(cache_key)
        if isinstance(cached, list):
            return cached

        news_data: List[Dict[str, Any]] = []

        if not YFINANCE_AVAILABLE or yf is None:
            cache.set(cache_key, news_data, ttl=self.cache_ttl)
            return news_data

        for ticker in self._ticker_formats(symbol):
            try:
                ticker_obj = yf.Ticker(ticker)
                # news プロパティが存在するか確認
                if hasattr(ticker_obj, "news"):
                    news = getattr(ticker_obj, "news", [])
                    if news:
                        # news は list of dict
                        # 必要に応じてフィルタリングや整形を行う
                        # 例: providerPublishTime を datetime に変換
                        processed_news = []
                        for item in news:
                            processed_item = {
                                "title": item.get("title"),
                                "publisher": item.get("publisher"),
                                "link": item.get("link"),
                                "type": item.get("type"),
                                "id": item.get("id"),
                                "publish_time": item.get("providerPublishTime"),
                                # 'provider' や他のキーも必要に応じて展開
                            }
                            if processed_item["publish_time"]:
                                try:
                                    # timestamp が秒単位かミリ秒単位か確認
                                    # Yahoo Finance は秒単位で返す模様
                                    processed_item["publish_time"] = pd.to_datetime(
                                        processed_item["publish_time"], unit="s",
                                    )
                                except Exception:
                                    # 変換失敗時は元の値を保持
                                    pass
                            processed_news.append(processed_item)

                        news_data = processed_news
                        break  # 一番最初に取得できたニュースを使用
                else:
                    logger.debug(
                        f"'news' attribute not found for ticker {ticker}. This might be due to yfinance version or API limitations.",
                    )
                    news_data = []
                    break

            except AttributeError as e:
                # 'news' attribute がない場合
                logger.debug(f"AttributeError for ticker {ticker}: {e}")
                news_data = []
                break
            except Exception as exc:  # pragma: no cover - depends on yfinance
                logger.debug(
                    "Failed to retrieve news data for %s via %s: %s",
                    symbol,
                    ticker,
                    exc,
                )
                continue

        cache.set(cache_key, news_data, ttl=self.cache_ttl)
        return news_data


# Backwards compatibility helper -------------------------------------------------

_stock_data_provider: Optional[StockDataProvider] = None


def get_stock_data_provider() -> StockDataProvider:
    """Singleton-style access used by legacy modules."""
    global _stock_data_provider
    if _stock_data_provider is None:
        _stock_data_provider = StockDataProvider()
    return _stock_data_provider

    def _generate_demo_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """デモデータを生成する
        """
        import numpy as np
        import pandas as pd

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
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": [
                    p * np.random.uniform(1, 1.02) for p in prices
                ],  # HighはOpenより少し高い
                "Low": [
                    p * np.random.uniform(0.98, 1) for p in prices
                ],  # LowはOpenより少し低い
                "Close": prices,
                "Volume": volumes,
            },
            index=dates,
        )

        # 一部の値をNaNに置き換えて、検証用に使う
        if np.random.random() > 0.9:  # 10%の確率で
            df.loc[df.index[:5], "Close"] = np.nan  # 最初の5行をNaN

        return df
