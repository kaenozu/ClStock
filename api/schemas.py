"""Pydantic response schemas for the public API."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class StockRecommendationSchema(BaseModel):
    """Serializable representation of a stock recommendation."""

    rank: int
    symbol: str
    company_name: str
    buy_timing: str
    target_price: float
    stop_loss: float
    profit_target_1: float
    profit_target_2: float
    holding_period: str
    score: float
    current_price: float
    recommendation_reason: str
    recommendation_level: Optional[str] = None
    generated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class RecommendationResponse(BaseModel):
    """Top level schema returned by the recommendations endpoint."""

    recommendations: List[StockRecommendationSchema]
    generated_at: datetime
    market_status: str

    model_config = ConfigDict(from_attributes=True)
