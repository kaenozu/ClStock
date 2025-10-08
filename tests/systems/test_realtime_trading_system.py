import sys
import types
from datetime import date, datetime

import pytest

import numpy as np
import pandas as pd

# Provide lightweight sklearn stubs to avoid optional dependency imports during testing.
sklearn_stub = types.ModuleType("sklearn")
preprocessing_stub = types.ModuleType("sklearn.preprocessing")
linear_model_stub = types.ModuleType("sklearn.linear_model")


class _DummyScaler:
    pass


class _DummyLogisticRegression:
    pass


preprocessing_stub.StandardScaler = _DummyScaler
linear_model_stub.LogisticRegression = _DummyLogisticRegression
sklearn_stub.preprocessing = preprocessing_stub
sklearn_stub.linear_model = linear_model_stub

sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.preprocessing", preprocessing_stub)
sys.modules.setdefault("sklearn.linear_model", linear_model_stub)


from systems import realtime_trading_system as rtts
from systems.realtime_trading_system import RealTimeTradingSystem, RiskManager


class DummyDateTime(datetime):
    """Helper datetime subclass to control now()."""

    _now = None

    @classmethod
    def set_now(cls, new_now: datetime) -> None:
        cls._now = new_now

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        if cls._now is None:
            return super().now(tz=tz)
        if tz is not None and cls._now.tzinfo is None:
            return cls._now.replace(tzinfo=tz)
        return cls._now


@pytest.fixture
def reset_datetime(monkeypatch):
    monkeypatch.setattr(rtts, "datetime", DummyDateTime)
    yield
    DummyDateTime.set_now(None)
    monkeypatch.setattr(rtts, "datetime", datetime)


def make_history(size: int = 60, close: float = 500.0) -> pd.DataFrame:
    index = pd.date_range(end=datetime(2024, 1, 1), periods=size, freq="D")
    data = pd.DataFrame(
        {
            "Open": np.linspace(close - 5, close + 5, size),
            "High": np.linspace(close - 2, close + 8, size),
            "Low": np.linspace(close - 8, close + 2, size),
            "Close": np.linspace(close - 4, close + 4, size),
            "Volume": np.random.randint(1000, 5000, size),
        },
        index=index,
    )
    return data


class DummyDataProvider:
    def __init__(self, history: pd.DataFrame, realtime_close: float):
        self.history = history
        self.realtime_close = realtime_close
        self.history_calls = []
        self.realtime_calls = []

    def get_historical_context(self, symbol: str):
        self.history_calls.append(symbol)
        return self.history.copy()

    def get_realtime_data(self, symbol: str):
        self.realtime_calls.append(symbol)
        return pd.DataFrame(
            {"Close": [self.realtime_close]},
            index=[datetime(2024, 1, 1, 10, 0)],
        )


class DummyPatternDetector:
    def __init__(self, signal: int, confidence: float, reason: str, price: float):
        self.signal = signal
        self.confidence = confidence
        self.reason = reason
        self.price = price
        self.calls = []

    def detect_846_pattern(self, data):
        self.calls.append(len(data))
        return {
            "signal": self.signal,
            "confidence": self.confidence,
            "reason": self.reason,
            "current_price": self.price,
        }


class DummyOrderExecutor:
    def __init__(self, status: str):
        self.status = status
        self.order_history = []

    def execute_order(self, symbol, action, size, price, confidence):
        order = {
            "symbol": symbol,
            "action": action,
            "size": size,
            "price": price,
            "confidence": confidence,
            "status": self.status,
        }
        self.order_history.append(order)
        return order


class TestRiskManager:
    def test_calculate_position_size_respects_total_exposure_margin(self):
        manager = RiskManager(initial_capital=1_000_000)
        manager.positions = {
            "AAA": {"value": 600_000, "size": 1200, "avg_price": 500.0},
        }
        manager.current_capital = 400_000

        result = manager.calculate_position_size(
            symbol="BBB",
            signal=1,
            confidence=0.846,
            current_price=500.0,
        )

        assert result["size"] > 0, "Expected additional position below exposure limit"

    def test_calculate_position_size_returns_min_unit_reason(self):
        manager = RiskManager(initial_capital=100_000)
        manager.current_capital = 20_000

        result = manager.calculate_position_size(
            symbol="CCC",
            signal=1,
            confidence=0.423,
            current_price=2_000.0,
        )

        assert result == {"size": 0, "reason": "最小単位未満"}

    def test_check_daily_reset_resets_counters(self, reset_datetime):
        manager = RiskManager(initial_capital=500_000)
        manager.daily_trades = 3
        manager.last_reset_date = date(2024, 1, 1)

        DummyDateTime.set_now(datetime(2024, 1, 2, 9, 0))
        manager.check_daily_reset()

        assert manager.daily_trades == 0
        assert manager.last_reset_date == date(2024, 1, 2)


class TestRealTimeTradingSystemProcessSymbol:
    def setup_system(self, monkeypatch, order_status="executed"):
        system = RealTimeTradingSystem(initial_capital=1_000_000)

        history = make_history()
        data_provider = DummyDataProvider(history, realtime_close=550.0)
        pattern = DummyPatternDetector(
            signal=1,
            confidence=0.9,
            reason="dummy",
            price=550.0,
        )
        executor = DummyOrderExecutor(status=order_status)

        system.data_provider = data_provider
        system.pattern_detector = pattern
        system.order_executor = executor

        def fake_calculate(self, symbol, signal, confidence, current_price):
            return {"size": 100, "value": 55_000.0, "ratio": 0.1, "reason": "ok"}

        monkeypatch.setattr(
            system.risk_manager,
            "calculate_position_size",
            types.MethodType(fake_calculate, system.risk_manager),
        )

        return system

    def test_process_symbol_executes_and_updates_state(self, monkeypatch):
        system = self.setup_system(monkeypatch, order_status="executed")

        system._process_symbol("7203")

        assert system.order_executor.order_history == [
            {
                "symbol": "7203",
                "action": "BUY",
                "size": 100,
                "price": 550.0,
                "confidence": 0.9,
                "status": "executed",
            },
        ]
        assert "7203" in system.risk_manager.positions
        assert system.risk_manager.positions["7203"]["size"] == 100

    def test_process_symbol_skips_position_update_on_rejection(self, monkeypatch):
        system = self.setup_system(monkeypatch, order_status="rejected")

        calls = []

        def track_update(self, symbol, action, size, price):
            calls.append((symbol, action, size, price))

        monkeypatch.setattr(
            system.risk_manager,
            "update_positions",
            types.MethodType(track_update, system.risk_manager),
        )

        system._process_symbol("7203")

        assert system.order_executor.order_history[0]["status"] == "rejected"
        assert calls == []
        assert system.risk_manager.positions == {}


class TestRealTimeTradingSystemStatus:
    def test_is_market_hours(self, reset_datetime):
        system = RealTimeTradingSystem(initial_capital=1_000_000)

        DummyDateTime.set_now(datetime(2024, 1, 3, 10, 0))
        assert system._is_market_hours() is True

        DummyDateTime.set_now(datetime(2024, 1, 3, 8, 30))
        assert system._is_market_hours() is False

        DummyDateTime.set_now(datetime(2024, 1, 6, 10, 0))
        assert system._is_market_hours() is False

    def test_get_status_report_aggregates_positions(self, monkeypatch):
        system = RealTimeTradingSystem(initial_capital=1_000_000)
        system.is_running = True
        system.risk_manager.current_capital = 900_000
        system.risk_manager.positions = {
            "7203": {"size": 100, "avg_price": 500.0, "value": 50_000},
            "6758": {"size": 200, "avg_price": 300.0, "value": 60_000},
        }

        def fake_realtime(symbol):
            close = 600.0 if symbol == "7203" else 310.0
            return pd.DataFrame({"Close": [close]}, index=[datetime(2024, 1, 3, 10, 0)])

        monkeypatch.setattr(system.data_provider, "get_realtime_data", fake_realtime)

        report = system.get_status_report()

        expected_value = 900_000 + 100 * 600.0 + 200 * 310.0
        expected_return = (expected_value - 1_000_000) / 1_000_000 * 100

        assert report["status"] == "running"
        assert report["total_value"] == expected_value
        assert report["total_return_pct"] == pytest.approx(expected_return)
        assert report["order_history_count"] == len(system.order_executor.order_history)
