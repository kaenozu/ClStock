import math

from trading.portfolio_manager import DemoPortfolioManager
from trading.risk_manager import DemoRiskManager
from trading.trading_strategy import SignalType


def test_add_position_rejects_when_cash_insufficient():
    manager = DemoPortfolioManager(initial_capital=1_000)

    assert manager.add_position("AAA", 5, 100.0, SignalType.BUY) is True
    assert math.isclose(manager.current_cash, 500.0)

    # Attempt to spend more cash than available should fail and leave balance unchanged
    assert manager.add_position("BBB", 10, 100.0, SignalType.BUY) is False
    assert math.isclose(manager.current_cash, 500.0)


def test_risk_manager_register_updates_capital_and_positions():
    risk_manager = DemoRiskManager(initial_capital=1_000)

    assert risk_manager.can_open_position(
        "AAA", position_size=80.0, confidence=0.8, precision=87.0
    )

    risk_manager.register_trade_open("AAA", quantity=4, price=100.0)
    assert math.isclose(risk_manager.current_capital, 600.0)
    assert risk_manager.positions["AAA"]["quantity"] == 4

    risk_manager.register_trade_close("AAA", quantity=4, price=120.0)
    assert math.isclose(risk_manager.current_capital, 1_080.0)
    assert "AAA" not in risk_manager.positions

    # After closing, opening another trade should respect the new larger capital base
    assert risk_manager.current_capital > 1_000
