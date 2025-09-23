from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class StockRecommendation(BaseModel):
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


class RecommendationResponse(BaseModel):
    recommendations: List[StockRecommendation]
    generated_at: datetime
    market_status: str
