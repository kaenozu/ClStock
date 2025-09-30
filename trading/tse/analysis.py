"""Stock analysis utilities for the TSE 4000 optimizer."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ANNUAL_TRADING_DAYS = 252
LARGE_CAP_THRESHOLD = 1e12
MID_CAP_THRESHOLD = 1e11

TREND_MA_PERIOD_SHORT = 20
TREND_MA_PERIOD_LONG = 60

PROFIT_WEIGHT = 0.5
DIVERSITY_WEIGHT = 0.3
STABILITY_WEIGHT = 0.2
STABILITY_MULTIPLIER = 10

MAX_WORKERS = 10
PROGRESS_REPORT_INTERVAL = 10

LOW_VOLATILITY = 0.2
MID_VOLATILITY = 0.4


@dataclass
class StockProfile:
    """Aggregated analytics for a single stock."""

    symbol: str
    sector: str
    market_cap: float
    volatility: float
    profit_potential: float
    diversity_score: float
    combined_score: float


class StockAnalyzer:
    """Performs per-stock analysis and scoring."""

    def __init__(self, data_provider, tse_universe: Dict[str, List[str]]):
        self.data_provider = data_provider
        self.tse_universe = tse_universe

    def parallel_analysis(self, symbols: List[str]) -> List[StockProfile]:
        """Analyze stocks in parallel and return their profiles."""

        print("並列分析開始...")
        profiles: List[StockProfile] = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_symbol = {
                executor.submit(self.analyze_stock_profile, symbol): symbol
                for symbol in symbols
            }

            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    profile = future.result()
                    if profile:
                        profiles.append(profile)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logging.warning(f"分析失敗 {symbol}: {exc}")
                finally:
                    completed += 1
                    if completed % PROGRESS_REPORT_INTERVAL == 0:
                        print(f"分析完了: {completed}/{len(symbols)}")

        return profiles

    def analyze_stock_profile(self, symbol: str) -> Optional[StockProfile]:
        """Run a full analysis for an individual stock."""

        try:
            stock_data = self._get_stock_data_for_analysis(symbol)
            if stock_data.empty:
                return None

            return self._create_stock_profile(symbol, stock_data)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.warning(f"Error analyzing {symbol}: {exc}")
            return None

    def _get_stock_data_for_analysis(self, symbol: str) -> pd.DataFrame:
        """Fetch historical price data used for analysis."""

        return self.data_provider.get_stock_data(symbol, "2y")

    def _create_stock_profile(
        self, symbol: str, stock_data: pd.DataFrame
    ) -> StockProfile:
        """Transform historical data into a StockProfile."""

        close = stock_data["Close"]
        volume = stock_data["Volume"]

        volatility = self._calculate_volatility(close)
        profit_potential = self._calculate_profit_potential(close)
        sector = self.determine_sector(symbol)
        market_cap = self._estimate_market_cap(close, volume)
        diversity_score = self.calculate_diversity_score(symbol, volatility, market_cap)
        combined_score = self._calculate_combined_score(
            profit_potential, diversity_score, volatility
        )

        return StockProfile(
            symbol=symbol,
            sector=sector,
            market_cap=market_cap,
            volatility=volatility,
            profit_potential=profit_potential,
            diversity_score=diversity_score,
            combined_score=combined_score,
        )

    def _calculate_volatility(self, close: pd.Series) -> float:
        """Calculate annualised volatility."""

        returns = close.pct_change().dropna()
        return returns.std() * np.sqrt(ANNUAL_TRADING_DAYS)

    def _calculate_profit_potential(self, close: pd.Series) -> float:
        """Calculate trend and momentum based profit potential."""

        ma_short = close.rolling(TREND_MA_PERIOD_SHORT).mean()
        ma_long = close.rolling(TREND_MA_PERIOD_LONG).mean()

        trend_strength = self._calculate_trend_strength(ma_short, ma_long)
        momentum = self._calculate_momentum(close)

        return (trend_strength + momentum) * 100

    def _calculate_trend_strength(
        self, ma_short: pd.Series, ma_long: pd.Series
    ) -> float:
        """Return relative strength of the short moving average."""

        if ma_long.iloc[-1] > 0:
            return (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
        return 0

    def _calculate_momentum(self, close: pd.Series) -> float:
        """Compute simple momentum over the last month."""

        if close.iloc[-20] > 0:
            return (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
        return 0

    def _estimate_market_cap(self, close: pd.Series, volume: pd.Series) -> float:
        """Approximate market cap via price and volume."""

        return close.iloc[-1] * volume.mean()

    def _calculate_combined_score(
        self, profit_potential: float, diversity_score: float, volatility: float
    ) -> float:
        """Create a weighted combined score for the stock."""

        stability = 1 / (volatility + 0.01)
        return (
            profit_potential * PROFIT_WEIGHT
            + diversity_score * DIVERSITY_WEIGHT
            + stability * STABILITY_MULTIPLIER * STABILITY_WEIGHT
        )

    def determine_sector(self, symbol: str) -> str:
        """Look up the sector for a stock symbol."""

        for sector, symbols in self.tse_universe.items():
            if symbol in symbols:
                return sector
        return "other"

    def calculate_diversity_score(
        self, symbol: str, volatility: float, market_cap: float
    ) -> float:
        """Calculate diversity score based on sector, cap and volatility."""

        sector_weight = self._get_sector_weight(symbol)
        cap_score = self._get_cap_score(market_cap)
        vol_score = self._get_volatility_score(volatility)

        return sector_weight * cap_score * vol_score

    def _get_sector_weight(self, symbol: str) -> float:
        sector = self.determine_sector(symbol)
        sector_weights = {
            "finance": 1.0,
            "tech": 1.2,
            "automotive": 1.0,
            "manufacturing": 0.9,
            "consumer": 1.1,
            "energy": 1.3,
            "healthcare": 1.2,
            "realestate": 0.8,
            "telecom": 1.0,
            "chemicals": 0.9,
            "food": 0.7,
            "transport": 1.1,
        }
        return sector_weights.get(sector, 1.0)

    def _get_cap_score(self, market_cap: float) -> float:
        if market_cap > LARGE_CAP_THRESHOLD:
            return 1.0
        if market_cap > MID_CAP_THRESHOLD:
            return 1.2
        return 1.1

    def _get_volatility_score(self, volatility: float) -> float:
        if volatility < LOW_VOLATILITY:
            return 1.0
        if volatility < MID_VOLATILITY:
            return 1.1
        return 1.2
