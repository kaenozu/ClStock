"""Pydantic response schemas for the public API."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class EntryPriceSchema(BaseModel):
    """Entry price information."""

    min: float
    max: float


class TargetSchema(BaseModel):
    """Target price information."""

    label: str
    price: float


class ChartRefsSchema(BaseModel):
    """Chart reference information."""

    support_levels: List[float]
    resistance_levels: List[float]
    indicators: List[str]


class StockRecommendationSchema(BaseModel):
    """Serializable representation of a stock recommendation."""

    rank: int
    symbol: str
    name: str  # company_name -> name に変更
    sector: Optional[str] = None  # 追加
    score: float
    action: str  # 追加
    action_text: Optional[str] = None  # 追加 (CUI向け)
    entry_condition: Optional[str] = None  # 追加
    entry_price: EntryPriceSchema  # target_price -> entry_price (object) に変更
    stop_loss: float
    targets: List[
        TargetSchema
    ]  # profit_target_1, profit_target_2 -> targets (array<object>) に変更
    holding_period_days: int  # holding_period (str) -> holding_period_days (int) に変更
    confidence: Optional[float] = None  # 追加
    rationale: Optional[str] = None  # recommendation_reason -> rationale に変更
    notes: Optional[str] = None  # 追加
    risk_level: Optional[str] = None  # 追加
    chart_refs: Optional[ChartRefsSchema] = None  # 追加

    # current_price は一旦維持
    current_price: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class RecommendationResponse(BaseModel):
    """Top level schema returned by the recommendations endpoint."""

    items: List[StockRecommendationSchema]  # recommendations -> items に変更
    generated_at: datetime
    market_status: str
    top_n: int

    model_config = ConfigDict(from_attributes=True)
