import pandas as pd
import pytest

from trading.tse.backtester import PortfolioBacktester


class StubProvider:
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def get_multiple_stocks(self, symbols, period="1y"):
        self.calls.append((tuple(symbols), period))
        return {
            symbol: self.frames[symbol] for symbol in symbols if symbol in self.frames
        }


@pytest.fixture
def sample_frames():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    return {
        "AAA": pd.DataFrame({"Close": [100, 110, 105, 120]}, index=index),
        "BBB": pd.DataFrame({"Close": [200, 190, 195, 210]}, index=index),
    }


def test_backtest_portfolio_computes_metrics(sample_frames):
    provider = StubProvider(sample_frames)
    backtester = PortfolioBacktester(provider)

    result = backtester.backtest_portfolio(["AAA", "BBB"], period="6mo")

    assert provider.calls == [(("AAA", "BBB"), "6mo")]
    assert result["symbols"] == ["AAA", "BBB"]
    assert result["period"] == "6mo"
    assert result["observations"] == 3
    assert pytest.approx(result["return_rate"], rel=1e-3) == 12.675
    assert pytest.approx(result["total_return"], rel=1e-3) == 0.12675
    assert pytest.approx(result["volatility"], rel=1e-3) == 0.05019
    assert pytest.approx(result["sharpe_ratio"], rel=1e-3) == 13.2123
    assert pytest.approx(result["max_drawdown"], rel=1e-3) == -0.009567


def test_backtest_portfolio_requires_prices(sample_frames):
    empty_provider = StubProvider(
        {symbol: df.iloc[:1] for symbol, df in sample_frames.items()}
    )
    backtester = PortfolioBacktester(empty_provider)

    with pytest.raises(ValueError):
        backtester.backtest_portfolio(["AAA", "BBB"])
