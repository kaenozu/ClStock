from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.security import HTTPAuthorizationCredentials

from datetime import datetime, time
from dataclasses import dataclass
from typing import List
import pandas as pd

# セキュリティと検証機能
from api.security import security, verify_token
from utils.validators import (
    validate_stock_symbol,
    validate_period,
    ValidationError,
    log_validation_error,
)
from utils.exceptions import DataFetchError

from models.core import MLStockPredictor
from models.recommendation import StockRecommendation
from api.schemas import RecommendationResponse
from data.stock_data import StockDataProvider


router = APIRouter()


@router.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    top_n: int = Query(5, ge=1, le=50, description="推奨銘柄の上位N件を取得"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        # Authentication verification
        verify_token(credentials.credentials)

        predictor = MLStockPredictor()
        recommendations = predictor.get_top_recommendations(top_n)

        current_time = datetime.now()
        market_open_time = time(9, 0)
        market_close_time = time(15, 0)

        return RecommendationResponse(
            recommendations=recommendations,
            generated_at=current_time,
            market_status=(
                "市場営業時間外"
                if current_time.time() < market_open_time
                or current_time.time() >= market_close_time
                else "市場営業中"
            ),
        )
    except ValidationError as e:
        log_validation_error(e, {"endpoint": "/recommendations", "top_n": top_n})
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"推奨銘柄の取得に失敗しました: {str(e)}",
        )


@router.get("/recommendation/{symbol}")
async def get_single_recommendation(
    symbol: str, credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get recommendation information for a specific symbol"""
    from utils.exceptions import InvalidSymbolError, PredictionError

    try:
        # Authentication verification
        verify_token(credentials.credentials)

        # Input validation
        validated_symbol = validate_stock_symbol(symbol)
        base_symbol = validated_symbol.split(".")[0]

        predictor = MLStockPredictor()
        data_provider = StockDataProvider()

        available_symbols = data_provider.get_all_stock_symbols()
        if base_symbol not in available_symbols:
            raise InvalidSymbolError(validated_symbol, available_symbols)

        recommendation = predictor.generate_recommendation(base_symbol)
        recommendation.rank = 1
        recommendation.symbol = validated_symbol
        recommendation.company_name = data_provider.jp_stock_codes.get(
            base_symbol, recommendation.company_name
        )

        return recommendation
    except InvalidSymbolError as e:
        raise HTTPException(
            status_code=404, detail=f"銘柄コード {symbol} が見つかりません"
        )
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
        validated_symbol = validate_stock_symbol(symbol)
        base_symbol = validated_symbol.split(".")[0]
        validated_period = validate_period(period)

        data_provider = StockDataProvider()

        available_symbols = set(data_provider.get_all_stock_symbols())
        lookup_symbol = None

        if validated_symbol in available_symbols:
            lookup_symbol = validated_symbol
        else:
            base_symbol = validated_symbol.split(".")[0]
            if base_symbol in available_symbols:
                lookup_symbol = base_symbol

        if lookup_symbol is None:
            raise HTTPException(
                status_code=404, detail=f"銘柄コード {symbol} が見つかりません"
            )

        data = data_provider.get_stock_data(lookup_symbol, validated_period)

        if data.empty:
            raise HTTPException(
                status_code=404, detail=f"銘柄 {symbol} のデータが見つかりません"
            )

        technical_data = data_provider.calculate_technical_indicators(data)
        financial_metrics = data_provider.get_financial_metrics(lookup_symbol)

        current_price = float(technical_data["Close"].iloc[-1])
        price_change = 0.0
        price_change_percent = 0.0

        if len(technical_data) >= 2:
            previous_close = technical_data["Close"].iloc[-2]
            if not pd.isna(previous_close):
                price_change = float(current_price - previous_close)
                if previous_close != 0:
                    price_change_percent = float(price_change / previous_close * 100)

        return {
            "symbol": validated_symbol,
            "company_name": data_provider.jp_stock_codes.get(
                lookup_symbol, lookup_symbol
            ),
            "current_price": current_price,
            "price_change": price_change,
            "price_change_percent": price_change_percent,
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
    except ValidationError as e:
        log_validation_error(
            e,
            {
                "endpoint": "/stock/{symbol}/data",
                "symbol": symbol,
                "period": period,
            },
        )
        raise HTTPException(status_code=400, detail=str(e))
    except DataFetchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"株価データの取得に失敗しました: {str(e)}"
        )
