"""Stock analysis module for ClStock."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class StockProfile:
    """Stock profile dataclass."""

    symbol: str
    sector: str
    market_cap: float
    volatility: float
    profit_potential: float
    diversity_score: float
    combined_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "sector": self.sector,
            "market_cap": self.market_cap,
            "volatility": self.volatility,
            "profit_potential": self.profit_potential,
            "diversity_score": self.diversity_score,
            "combined_score": self.combined_score,
        }
