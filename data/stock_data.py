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
from typing import Dict, Iterable, List, Optional, Tuple, Any

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

        def history(
            self,
            period: str = "1y",
            interval: str = "1d",
            start: Optional[str] = None,
            end: Optional[str] = None,
        ) -> pd.DataFrame:
            def _empty_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "Open": pd.Series(dtype="float64"),
                        "High": pd.Series(dtype="float64"),
                        "Low": pd.Series(dtype="float64"),
                        "Close": pd.Series(dtype="float64"),
                        "Volume": pd.Series(dtype="int64"),
                    },
                    index=index,
                )

            def _normalize(value: Optional[str]) -> Optional[pd.Timestamp]:
                if value is None:
                    return None
                ts = pd.to_datetime(value)
                if isinstance(ts, pd.DatetimeIndex):  # defensive, though unlikely
                    ts = ts[0]
                if ts.tzinfo is not None:
                    ts = ts.tz_convert(None)
                return ts.normalize()

            start_ts = _normalize(start)
            end_ts = _normalize(end)

            if start_ts is not None or end_ts is not None:
                if end_ts is None:
                    end_ts = pd.Timestamp.utcnow().normalize()
                if start_ts is None:
                    default_length = max(5, _period_to_days(period))
                    start_ts = (end_ts - pd.tseries.offsets.BDay(default_length - 1)).normalize()
                if end_ts < start_ts:
                    return _empty_frame(pd.DatetimeIndex([], dtype="datetime64[ns]"))
                index = pd.bdate_range(start=start_ts, end=end_ts)
            else:
                length = max(5, _period_to_days(period))
                end_date = datetime.utcnow().date()
                index = pd.bdate_range(end_date - timedelta(days=length + 5), periods=length)

            base = 90 + (self._seed % 60)
            variation = (self._seed % 13) - 6
            close = pd.Series(
                [base + variation * (i % 5) * 0.5 for i in range(len(index))],
                index=index,
                dtype="float",
            )
            if close.empty:
                return _empty_frame(index)

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

    def get_stock_data(self, symbol: str, period: str = "1y", start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Fetch historical OHLCV data for *symbol*.

        The result always contains the ``Symbol``, ``ActualTicker`` and
        ``CompanyName`` columns in addition to the usual OHLCV fields.
        """

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

        # start/end が指定されている場合は、ローカルCSVは使用しない (期間指定が一致するとは限らないため)
        if start is None and end is None and self._should_use_local_first(symbol):
            local = self._load_first_available_csv(symbol)
            if local is not None:
                data, actual_ticker = local

        if data is None or data.empty:
            if start is not None or end is not None:
                # yfinance の start と end パラメータを直接使用して、指定された期間のデータを取得
                # yfinance は start と end の日付文字列を解析し、該当する期間のデータを取得する
                # ただし、休場日やデータの可用性により、実際の取得期間が要求期間と異なる場合がある
                data, actual_ticker = self._download_via_yfinance(symbol, period=None, start=start, end=end)
            else:
                data, actual_ticker = self._download_via_yfinance(symbol, period)

        if data is None or data.empty:
            raise DataFetchError(symbol, "No historical data available")

        # yfinance が実際に取得した期間を確認
        if start is not None and end is not None:
            actual_start = data.index.min()
            actual_end = data.index.max()
            print(f"StockDataProvider: Requested: {start} to {end}, Actual: {actual_start} to {actual_end}")
            # 必要に応じて、取得範囲が要求範囲を満たしているか確認するロジックを追加
            # 例: 休場日などの関係で、要求した期間より狭くなることは許容するが、
            #     意図しない期間が取得されないよう、ある程度のチェックを行う。
            # if actual_start > start or actual_end < end:
            #     print(f"Warning: Actual data range differs significantly from requested range for {symbol}.")

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

    def _download_via_yfinance(self, symbol: str, period: Optional[str] = "1y", start: Optional[str] = None, end: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[str]]:
        last_error: Optional[Exception] = None
        for ticker in self._ticker_formats(symbol):
            try:
                ticker_obj = yf.Ticker(ticker)
                if start is not None or end is not None:
                    history = ticker_obj.history(start=start, end=end)
                else:
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

        for ticker in self._ticker_formats(symbol):
            try:
                ticker_obj = yf.Ticker(ticker)
                info = getattr(ticker_obj, "info", {}) or {}

                # 基本的な財務指標以外のデータを取得
                # 例: revenue, grossProfits, operatingCashflow, ebitda, totalDebt, totalCash, debtToEquity
                # 他にも、ROE, ROA, ROICなどの計算に必要な生データも取得
                keys_to_fetch = [
                    "revenue", "grossProfits", "operatingCashflow", "ebitda", "totalDebt", "totalCash",
                    "debtToEquity", "returnOnAssets", "returnOnEquity", "earningsQuarterlyGrowth",
                    "quarterlyEarningsGrowth", "quarterlyRevenueGrowth"
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
                if extended_metrics.get("revenue") is not None or extended_metrics.get("totalCash") is not None:
                    extended_metrics["actual_ticker"] = ticker
                    break
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to retrieve extended financial metrics for %s via %s: %s", symbol, ticker, exc)
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
            "dividend_rate": None, # dividendRate
            "dividend_yield": None, # dividendYield (already in get_financial_metrics but included here for completeness)
            "ex_dividend_date": None, # exDividendDate (date string)
            "actual_ticker": None,
        }

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
                            dividend_data["ex_dividend_date"] = pd.to_datetime(ex_date_raw, unit='s').date().isoformat()
                        elif isinstance(ex_date_raw, str):
                            # strの場合は、既に YYYY-MM-DD 形式であると仮定
                            dividend_data["ex_dividend_date"] = ex_date_raw
                        else:
                            dividend_data["ex_dividend_date"] = None
                    except:
                        dividend_data["ex_dividend_date"] = None
                else:
                    dividend_data["ex_dividend_date"] = None

                dividend_data["actual_ticker"] = ticker
                break
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to retrieve dividend data for %s via %s: %s", symbol, ticker, exc)
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

        for ticker in self._ticker_formats(symbol):
            try:
                ticker_obj = yf.Ticker(ticker)
                # news プロパティが存在するか確認
                if hasattr(ticker_obj, 'news'):
                    news = getattr(ticker_obj, 'news', [])
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
                                    processed_item["publish_time"] = pd.to_datetime(processed_item["publish_time"], unit='s')
                                except:
                                    # 変換失敗時は元の値を保持
                                    pass
                            processed_news.append(processed_item)

                        news_data = processed_news
                        break  # 一番最初に取得できたニュースを使用
                else:
                    logger.debug(f"'news' attribute not found for ticker {ticker}. This might be due to yfinance version or API limitations.")
                    news_data = []
                    break

            except AttributeError as e:
                # 'news' attribute がない場合
                logger.debug(f"AttributeError for ticker {ticker}: {e}")
                news_data = []
                break
            except Exception as exc:  # pragma: no cover - depends on yfinance
                logger.debug("Failed to retrieve news data for %s via %s: %s", symbol, ticker, exc)
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
