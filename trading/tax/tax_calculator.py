from __future__ import annotations

"""税務計算ユーティリティ"""

from datetime import datetime
from typing import List, Optional

from trading.models import TaxCalculation, TradeRecord


class TaxCalculator:
    """簡易税務計算ロジック"""

    def calculate(
        self,
        trades: List[TradeRecord],
        tax_year: Optional[int] = None,
    ) -> TaxCalculation:
        if tax_year is None:
            tax_year = datetime.now().year

        start_date = datetime(tax_year, 1, 1)
        end_date = datetime(tax_year, 12, 31)

        relevant_trades = [
            trade
            for trade in trades
            if start_date <= trade.timestamp <= end_date and trade.action == "CLOSE"
        ]

        if not relevant_trades:
            return TaxCalculation(
                total_realized_gains=0.0,
                total_realized_losses=0.0,
                net_capital_gains=0.0,
                short_term_gains=0.0,
                long_term_gains=0.0,
                wash_sales=0.0,
                estimated_tax_liability=0.0,
                deductible_expenses=0.0,
            )

        gains = [trade.profit_loss for trade in relevant_trades if trade.profit_loss and trade.profit_loss > 0]
        losses = [abs(trade.profit_loss) for trade in relevant_trades if trade.profit_loss and trade.profit_loss < 0]

        total_realized_gains = float(sum(gains)) if gains else 0.0
        total_realized_losses = float(sum(losses)) if losses else 0.0
        net_capital_gains = total_realized_gains - total_realized_losses

        short_term_gains = 0.0
        long_term_gains = 0.0
        for trade in relevant_trades:
            if trade.profit_loss and trade.profit_loss > 0:
                holding_days = self._calculate_holding_days(trade)
                if holding_days < 365:
                    short_term_gains += trade.profit_loss
                else:
                    long_term_gains += trade.profit_loss

        deductible_expenses = sum(sum(trade.trading_costs.values()) for trade in relevant_trades)
        taxable_gains = max(net_capital_gains - deductible_expenses, 0)
        estimated_tax_liability = taxable_gains * 0.20315

        return TaxCalculation(
            total_realized_gains=total_realized_gains,
            total_realized_losses=total_realized_losses,
            net_capital_gains=net_capital_gains,
            short_term_gains=short_term_gains,
            long_term_gains=long_term_gains,
            wash_sales=0.0,
            estimated_tax_liability=estimated_tax_liability,
            deductible_expenses=deductible_expenses,
        )

    def _calculate_holding_days(self, trade: TradeRecord) -> int:
        # 実際には対応するOPEN取引との日数差を計算する必要があるが簡略化
        return 1
