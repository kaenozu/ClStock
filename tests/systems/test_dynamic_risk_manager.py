from datetime import datetime, timedelta

import pytest

import pandas as pd
from systems.dynamic_risk_manager import AdvancedRiskManager


class FakeProvider:
    """Simple data provider that serves deterministic OHLC data for tests."""

    def __init__(self):
        self._data = {}

    def add_series(
        self, symbol: str, data: pd.DataFrame, period: str | None = None,
    ) -> None:
        key = (symbol, period)
        self._data[key] = data.copy()

    def get_stock_data(self, symbol: str, period: str) -> pd.DataFrame:
        data = self._data.get((symbol, period))
        if data is None:
            data = self._data.get((symbol, None))
        if data is None:
            data = pd.DataFrame(columns=["Close"])
        return data.copy()


@pytest.fixture
def risk_manager():
    manager = AdvancedRiskManager(initial_capital=100_000)
    manager.data_provider = FakeProvider()
    return manager


def _linear_close_series(
    start: float, step: float, periods: int, start_date: datetime | None = None,
):
    start_date = start_date or datetime(2024, 1, 1)
    index = pd.date_range(start_date, periods=periods, freq="D")
    close = start + step * pd.Series(range(periods), index=index)
    return pd.DataFrame({"Close": close})


class TestDynamicStopLoss:
    def test_calculate_dynamic_stop_loss_adjusts_factors(self, risk_manager):
        high_confidence = risk_manager.calculate_dynamic_stop_loss(
            "AAA", 100.0, confidence=0.9, volatility=0.02,
        )
        low_confidence = risk_manager.calculate_dynamic_stop_loss(
            "AAA", 100.0, confidence=0.55, volatility=0.02,
        )
        high_volatility = risk_manager.calculate_dynamic_stop_loss(
            "AAA", 100.0, confidence=0.6, volatility=0.12,
        )

        for result in (high_confidence, low_confidence, high_volatility):
            assert 0.02 <= result["stop_loss_rate"] <= 0.15

        assert (
            high_confidence["confidence_factor"] < low_confidence["confidence_factor"]
        )
        assert high_confidence["pattern_factor"] < low_confidence["pattern_factor"]
        assert low_confidence["volatility_factor"] == pytest.approx(0.5)
        assert high_volatility["volatility_factor"] == pytest.approx(2.0)


class TestPortfolioRiskMetrics:
    def test_calculate_portfolio_var_with_deterministic_data(self, risk_manager):
        provider = risk_manager.data_provider
        provider.add_series("AAA", _linear_close_series(100, 1.2, 90))
        provider.add_series("BBB", _linear_close_series(200, 0.8, 90))

        risk_manager.positions = {
            "AAA": {
                "size": 100,
                "avg_price": 95.0,
                "entry_date": datetime.now() - timedelta(days=30),
            },
            "BBB": {
                "size": 50,
                "avg_price": 205.0,
                "entry_date": datetime.now() - timedelta(days=45),
            },
        }

        result = risk_manager.calculate_portfolio_var(confidence_level=0.05)

        assert result["var"] > 0
        assert result["cvar"] > 0
        assert result["portfolio_volatility"] > 0
        assert result["total_portfolio_value"] > 0

    def test_calculate_portfolio_var_returns_zero_with_insufficient_data(
        self, risk_manager,
    ):
        provider = risk_manager.data_provider
        provider.add_series("CCC", _linear_close_series(50, 1.0, 10))

        risk_manager.positions = {
            "CCC": {
                "size": 10,
                "avg_price": 48.0,
                "entry_date": datetime.now() - timedelta(days=5),
            },
        }

        result = risk_manager.calculate_portfolio_var(confidence_level=0.05)

        assert result["var"] == 0
        assert result["cvar"] == 0
        assert result["portfolio_volatility"] == 0

    def test_monitor_correlation_changes_flags_high_correlation(self, risk_manager):
        provider = risk_manager.data_provider
        provider.add_series("AAA", _linear_close_series(100, 0.5, 120))
        provider.add_series("BBB", _linear_close_series(150, 0.5, 120))

        risk_manager.positions = {
            "AAA": {
                "size": 200,
                "avg_price": 98.0,
                "entry_date": datetime.now() - timedelta(days=20),
            },
            "BBB": {
                "size": 150,
                "avg_price": 148.0,
                "entry_date": datetime.now() - timedelta(days=40),
            },
        }

        result = risk_manager.monitor_correlation_changes(lookback_days=60)

        assert result["correlation_risk"] == "極高"
        assert result["max_correlation"] == pytest.approx(1.0, abs=1e-3)
        assert any(
            pair[0] == "AAA" and pair[1] == "BBB"
            for pair in result["high_correlation_pairs"]
        )


class TestKellyAndDrawdown:
    def test_calculate_position_sizing_kelly_positive_only_when_edge_crossed(
        self, risk_manager,
    ):
        positive = risk_manager.calculate_position_sizing_kelly(
            win_probability=0.6, avg_win=2.0, avg_loss=1.0, current_capital=50_000,
        )
        boundary = risk_manager.calculate_position_sizing_kelly(
            win_probability=0.4, avg_win=1.5, avg_loss=1.0, current_capital=50_000,
        )

        assert positive["kelly_fraction"] > 0
        assert positive["optimal_size"] == pytest.approx(
            50_000 * positive["kelly_fraction"],
        )
        assert boundary["kelly_fraction"] == pytest.approx(0, abs=1e-12)
        assert boundary["optimal_size"] == pytest.approx(0, abs=1e-6)

    def test_assess_maximum_drawdown_risk_levels_and_recommendations(
        self, risk_manager,
    ):
        provider = risk_manager.data_provider
        provider.add_series("AAA", _linear_close_series(100, -10.0, 2), period="1d")
        provider.add_series("BBB", _linear_close_series(200, -15.0, 2), period="1d")

        risk_manager.positions = {
            "AAA": {
                "size": 300,
                "avg_price": 110.0,
                "entry_date": datetime.now() - timedelta(days=1),
            },
            "BBB": {
                "size": 200,
                "avg_price": 210.0,
                "entry_date": datetime.now() - timedelta(days=2),
            },
        }

        result = risk_manager.assess_maximum_drawdown_risk()

        assert result["risk_level"] in {"高", "極高"}
        assert result["recommendation"] == risk_manager._get_drawdown_recommendation(
            result["current_drawdown"],
        )


class TestComprehensiveRiskReport:
    def test_generate_comprehensive_risk_report_aggregates_metrics(
        self, risk_manager, monkeypatch,
    ):
        monkeypatch.setattr(
            risk_manager,
            "calculate_portfolio_var",
            lambda: {
                "var": 15_000,
                "cvar": 20_000,
                "var_rate": -0.15,
                "cvar_rate": -0.2,
                "portfolio_volatility": 0.04,
                "total_portfolio_value": 100_000,
                "confidence_level": 0.05,
            },
        )
        monkeypatch.setattr(
            risk_manager,
            "monitor_correlation_changes",
            lambda: {
                "correlation_risk": "高",
                "max_correlation": 0.85,
                "avg_correlation": 0.8,
                "high_correlation_pairs": [("AAA", "BBB", 0.85)],
            },
        )
        monkeypatch.setattr(
            risk_manager,
            "assess_maximum_drawdown_risk",
            lambda: {
                "current_drawdown": -0.12,
                "risk_level": "高",
                "recommendation": "警告: 損切り基準の再検討",
            },
        )

        report = risk_manager.generate_comprehensive_risk_report()

        assert report["overall_risk_level"] == "極高"
        assert report["overall_risk_score"] > 0
        assert isinstance(report["timestamp"], datetime)
        assert {
            "VaRが10%超過: ポジションサイズ縮小推奨",
            "高相関リスク: 分散投資の見直し推奨",
            "ドローダウン10%超過: 損切り実行検討",
        }.issubset(set(report["recommendations"]))

    def test_generate_comprehensive_risk_report_handles_zero_metrics(
        self, risk_manager, monkeypatch,
    ):
        monkeypatch.setattr(
            risk_manager,
            "calculate_portfolio_var",
            lambda: {"var": 0, "cvar": 0, "portfolio_volatility": 0},
        )
        monkeypatch.setattr(
            risk_manager,
            "monitor_correlation_changes",
            lambda: {"correlation_risk": "低", "max_correlation": 0.0},
        )
        monkeypatch.setattr(
            risk_manager,
            "assess_maximum_drawdown_risk",
            lambda: {"current_drawdown": 0.0, "risk_level": "低"},
        )

        report = risk_manager.generate_comprehensive_risk_report()

        assert report["overall_risk_score"] == 0
        assert report["overall_risk_level"] == "低"
        assert report["recommendations"] == ["リスク水準正常: 現在の戦略継続"]
