from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
import pandas as pd

from models.recommendation import RecommendationResponse, StockRecommendation
from models.predictor import StockPredictor
from data.stock_data import StockDataProvider

router = APIRouter()


@router.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    top_n: int = Query(5, ge=1, le=10, description="推奨銘柄の上位N件を取得")
):
    try:
        predictor = StockPredictor()
        recommendations = predictor.get_top_recommendations(top_n)

        return RecommendationResponse(
            recommendations=recommendations,
            generated_at=datetime.now(),
            market_status=(
                "市場営業時間外"
                if datetime.now().hour < 9 or datetime.now().hour > 15
                else "市場営業中"
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"推奨銘柄の取得に失敗しました: {str(e)}"
        )


@router.get("/recommendation/{symbol}", response_model=StockRecommendation)
async def get_single_recommendation(symbol: str):
    """特定の銘柄の推奨情報を取得"""
    from utils.exceptions import InvalidSymbolError, PredictionError

    try:
        predictor = StockPredictor()
        data_provider = StockDataProvider()

        if symbol not in data_provider.get_all_stock_symbols():
            raise InvalidSymbolError(symbol, data_provider.get_all_stock_symbols())

        recommendation = predictor.generate_recommendation(symbol)
        recommendation.rank = 1

        return recommendation
    except InvalidSymbolError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PredictionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"推奨情報の取得に失敗しました: {str(e)}"
        )


@router.get("/stocks")
async def get_available_stocks():
    try:
        data_provider = StockDataProvider()
        return {
            "stocks": [
                {"symbol": symbol, "name": name}
                for symbol, name in data_provider.jp_stock_codes.items()
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"銘柄一覧の取得に失敗しました: {str(e)}"
        )


@router.get("/stock/{symbol}/data")
async def get_stock_data(
    symbol: str,
    period: str = Query(
        "1y", description="期間 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"
    ),
):
    try:
        data_provider = StockDataProvider()

        if symbol not in data_provider.get_all_stock_symbols():
            raise HTTPException(
                status_code=404, detail=f"銘柄コード {symbol} が見つかりません"
            )

        data = data_provider.get_stock_data(symbol, period)

        if data.empty:
            raise HTTPException(
                status_code=404, detail=f"銘柄 {symbol} のデータが見つかりません"
            )

        technical_data = data_provider.calculate_technical_indicators(data)
        financial_metrics = data_provider.get_financial_metrics(symbol)

        return {
            "symbol": symbol,
            "company_name": data_provider.jp_stock_codes.get(symbol, symbol),
            "current_price": float(technical_data["Close"].iloc[-1]),
            "price_change": float(
                technical_data["Close"].iloc[-1] - technical_data["Close"].iloc[-2]
            ),
            "price_change_percent": float(
                (technical_data["Close"].iloc[-1] - technical_data["Close"].iloc[-2])
                / technical_data["Close"].iloc[-2]
                * 100
            ),
            "volume": int(technical_data["Volume"].iloc[-1]),
            "technical_indicators": {
                "sma_20": (
                    float(technical_data["SMA_20"].iloc[-1])
                    if not pd.isna(technical_data["SMA_20"].iloc[-1])
                    else None
                ),
                "sma_50": (
                    float(technical_data["SMA_50"].iloc[-1])
                    if not pd.isna(technical_data["SMA_50"].iloc[-1])
                    else None
                ),
                "rsi": (
                    float(technical_data["RSI"].iloc[-1])
                    if not pd.isna(technical_data["RSI"].iloc[-1])
                    else None
                ),
                "macd": (
                    float(technical_data["MACD"].iloc[-1])
                    if not pd.isna(technical_data["MACD"].iloc[-1])
                    else None
                ),
            },
            "financial_metrics": financial_metrics,
            "last_updated": datetime.now(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"株価データの取得に失敗しました: {str(e)}"
        )
