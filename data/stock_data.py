import yfinance as yf
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
            if symbol in self.jp_stock_codes:
                ticker = f"{symbol}.T"
            else:
                ticker = symbol

            logger.info(f"Fetching data for {symbol} (period: {period})")
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)

            if data.empty:
                raise DataFetchError(symbol, "No historical data available")

            data["Symbol"] = symbol
            data["CompanyName"] = self.jp_stock_codes.get(symbol, symbol)

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
