"""Portfolio backtesting module for ClStock."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import pandas as pd

from data.stock_data import StockDataProvider


class PortfolioBacktester:
    """Perform a lightweight equal-weight portfolio backtest."""

    _CLOSE_CANDIDATES: Sequence[str] = ("Close", "Adj Close", "close", "adj_close")

    def __init__(self, data_provider: StockDataProvider):
        self.data_provider = data_provider

    def backtest_portfolio(
        self, symbols: Sequence[str], period: str = "1y"
    ) -> Dict[str, Any]:
        """Backtest *symbols* for the specified *period*.

        The implementation intentionally keeps the calculation lightweight so it
        can run in constrained CI environments while still returning useful
        summary statistics for the portfolio.
        """

        if not symbols:
            raise ValueError("At least one symbol must be provided for backtesting")

        history = self.data_provider.get_multiple_stocks(symbols, period=period)

        returns: List[pd.Series] = []
        used_symbols: List[str] = []

        for symbol in symbols:
            frame = history.get(symbol)
            if frame is None or frame.empty:
                continue

            close = self._extract_close_prices(frame)
            if close is None or close.shape[0] < 2:
                continue

            close = close.sort_index()
            daily_returns = close.pct_change().dropna()
            if daily_returns.empty:
                continue

            daily_returns.name = symbol
            returns.append(daily_returns)
            used_symbols.append(symbol)

        if not returns:
            raise ValueError("Price history is unavailable for the requested symbols")

        aligned = pd.concat(returns, axis=1, join="inner")
        if aligned.empty:
            raise ValueError("No overlapping price history to compute portfolio returns")

        aligned = aligned.sort_index()
        portfolio_returns = aligned.mean(axis=1)
        if portfolio_returns.empty:
            raise ValueError("Not enough return observations to compute statistics")

        cumulative = (1 + portfolio_returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()

        volatility = float(portfolio_returns.std(ddof=0))
        avg_return = float(portfolio_returns.mean())
        sharpe_ratio = 0.0
        if volatility > 0:
            sharpe_ratio = (avg_return / volatility) * (252 ** 0.5)

        return {
            "symbols": used_symbols,
            "period": period,
            "observations": int(portfolio_returns.shape[0]),
            "return_rate": float(total_return * 100),
            "total_return": float(total_return),
            "volatility": volatility,
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
        }

    def _extract_close_prices(self, frame: pd.DataFrame) -> pd.Series | None:
        """Return a price series from *frame* if available."""

        for candidate in self._CLOSE_CANDIDATES:
            if candidate in frame.columns:
                series = pd.to_numeric(frame[candidate], errors="coerce")
                series = series.dropna()
                return series

        if frame.shape[1] == 1:
            series = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
            return series.dropna()

        return None
