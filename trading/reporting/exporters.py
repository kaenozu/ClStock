from __future__ import annotations

"""取引データエクスポートユーティリティ"""

import json
from dataclasses import asdict
from datetime import datetime
from typing import List, Optional

import pandas as pd
from trading.models import PerformanceMetrics, TradeRecord


class ExportService:
    """取引履歴のエクスポート機能"""

    def export_to_csv(
        self,
        trades: List[TradeRecord],
        filepath: str,
    ) -> bool:
        if not trades:
            return False

        csv_data = []
        for trade in trades:
            csv_data.append(
                {
                    "trade_id": trade.trade_id,
                    "session_id": trade.session_id,
                    "timestamp": trade.timestamp.isoformat(),
                    "symbol": trade.symbol,
                    "action": trade.action,
                    "signal_type": trade.signal_type.value,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "confidence": trade.confidence,
                    "precision": trade.precision,
                    "precision_87_achieved": trade.precision_87_achieved,
                    "expected_return": trade.expected_return,
                    "actual_return": trade.actual_return,
                    "profit_loss": trade.profit_loss,
                    "position_size": trade.position_size,
                    "market_value": trade.market_value,
                    "reasoning": trade.reasoning,
                    "commission": trade.trading_costs.get("commission", 0),
                    "spread": trade.trading_costs.get("spread", 0),
                    "total_cost": trade.trading_costs.get("total_cost", 0),
                },
            )

        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        return True

    def export_to_json(
        self,
        trades: List[TradeRecord],
        filepath: str,
        session_id: Optional[str] = None,
        include_performance: bool = True,
        performance_metrics: Optional[PerformanceMetrics] = None,
    ) -> bool:
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "trade_count": len(trades),
            "trades": [asdict(trade) for trade in trades],
        }

        if include_performance and performance_metrics is not None:
            export_data["performance_metrics"] = asdict(performance_metrics)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

        return True
