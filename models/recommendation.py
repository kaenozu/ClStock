from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict


@dataclass
class StockRecommendation:
    """Minimal representation of a stock recommendation used in tests."""

    rank: int = 0
    symbol: str = ""
    company_name: str = ""
    buy_timing: str = ""
    target_price: float = 0.0
    stop_loss: float = 0.0
    profit_target_1: float = 0.0
    profit_target_2: float = 0.0
    holding_period: str = ""
    score: float = 0.0
    current_price: float = 0.0
    recommendation_reason: str = ""
    recommendation_level: str = "neutral"
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, object]:
        return {
            "rank": self.rank,
            "symbol": self.symbol,
            "company_name": self.company_name,
            "buy_timing": self.buy_timing,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "profit_target_1": self.profit_target_1,
            "profit_target_2": self.profit_target_2,
            "holding_period": self.holding_period,
            "score": self.score,
            "current_price": self.current_price,
            "recommendation_reason": self.recommendation_reason,
            "recommendation_level": self.recommendation_level,
            "generated_at": self.generated_at.isoformat(),
        }
