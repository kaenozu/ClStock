"""
セキュアなAPIエンドポイント
認証・認可・入力検証を実装した安全なエンドポイント群
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Body
from fastapi.security import HTTPAuthorizationCredentials
from typing import Dict, Any
from datetime import datetime
import logging

# セキュリティと検証機能
from api.security import security, verify_token
from utils.validators import (
    validate_stock_symbol,
    validate_period,
    validate_symbols_list,
    ValidationError,
    log_validation_error,
)

# データプロバイダー
from data.stock_data import StockDataProvider

# ログ設定
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/secure", tags=["secure"])


@router.get("/stock/{symbol}/data")
async def get_stock_data(
    symbol: str,
    period: str = Query(
        "1y", description="期間 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y)"
    ),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    セキュアな株価データ取得エンドポイント
    認証済みユーザーのみアクセス可能
    """
    try:
        # 認証検証
        user_info = verify_token(credentials.credentials)

        # 入力検証
        symbol = validate_stock_symbol(symbol)
        period = validate_period(period)

        # データ取得
        data_provider = StockDataProvider()
        data = data_provider.get_stock_data(symbol, period)

        if data.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for symbol: {symbol}"
            )

        # レスポンス作成
        response = {
            "symbol": symbol,
            "period": period,
            "data_points": len(data),
            "date_range": {
                "start": data.index[0].isoformat(),
                "end": data.index[-1].isoformat(),
            },
            "latest_price": float(data["Close"].iloc[-1]),
            "price_change": float(data["Close"].iloc[-1] - data["Close"].iloc[0]),
            "user": user_info,
        }

        # ユーザータイプに応じて詳細データを追加
        if user_info == "administrator":
            response["detailed_data"] = data.to_dict(orient="records")

        logger.info(f"Stock data request: {symbol} by {user_info}")
        return response

    except ValidationError as e:
        log_validation_error(
            e, {"endpoint": "/secure/stock/data", "symbol": symbol, "period": period}
        )
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/stocks/batch")
async def get_batch_stock_data(
    symbols_data: Dict[str, Any] = Body(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    複数銘柄の一括データ取得（セキュア版）
    """
    try:
        # 認証検証
        user_info = verify_token(credentials.credentials)

        # 入力検証
        symbols = symbols_data.get("symbols", [])
        period = symbols_data.get("period", "1y")

        symbols = validate_symbols_list(symbols, max_count=50)  # 制限を強化
        period = validate_period(period)

        # 管理者以外は制限を設ける
        if user_info != "administrator" and len(symbols) > 10:
            raise ValidationError("Non-admin users limited to 10 symbols per request")

        # データ取得
        data_provider = StockDataProvider()
        results = {}
        errors = {}

        for symbol in symbols:
            try:
                data = data_provider.get_stock_data(symbol, period)
                if not data.empty:
                    results[symbol] = {
                        "data_points": len(data),
                        "latest_price": float(data["Close"].iloc[-1]),
                        "date_range": {
                            "start": data.index[0].isoformat(),
                            "end": data.index[-1].isoformat(),
                        },
                    }
                else:
                    errors[symbol] = "No data available"
            except Exception as e:
                errors[symbol] = str(e)

        logger.info(f"Batch stock data request: {len(symbols)} symbols by {user_info}")

        return {
            "requested_symbols": len(symbols),
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
            "user": user_info,
        }

    except ValidationError as e:
        log_validation_error(e, {"endpoint": "/secure/stocks/batch"})
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch stock data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analysis/{symbol}")
async def get_stock_analysis(
    symbol: str, credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    株式分析エンドポイント（管理者専用）
    """
    try:
        # 認証検証
        user_info = verify_token(credentials.credentials)

        # 管理者権限チェック
        if user_info != "administrator":
            raise HTTPException(status_code=403, detail="Administrator access required")

        # 入力検証
        symbol = validate_stock_symbol(symbol)

        # 分析実行
        data_provider = StockDataProvider()
        data = data_provider.get_stock_data(symbol, "1y")

        if data.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for symbol: {symbol}"
            )

        # テクニカル指標を計算
        data_with_indicators = data_provider.calculate_technical_indicators(data)

        # 分析結果
        analysis = {
            "symbol": symbol,
            "analysis_date": datetime.now().isoformat(),
            "price_analysis": {
                "current_price": float(data["Close"].iloc[-1]),
                "52_week_high": float(data["Close"].max()),
                "52_week_low": float(data["Close"].min()),
                "average_volume": float(data["Volume"].mean()),
            },
            "technical_indicators": {
                "rsi": (
                    float(data_with_indicators["RSI"].iloc[-1])
                    if "RSI" in data_with_indicators
                    else None
                ),
                "sma_20": (
                    float(data_with_indicators["SMA_20"].iloc[-1])
                    if "SMA_20" in data_with_indicators
                    else None
                ),
                "sma_50": (
                    float(data_with_indicators["SMA_50"].iloc[-1])
                    if "SMA_50" in data_with_indicators
                    else None
                ),
            },
            "analyst": user_info,
        }

        logger.info(f"Advanced analysis request: {symbol} by {user_info}")
        return analysis

    except ValidationError as e:
        log_validation_error(e, {"endpoint": "/secure/analysis", "symbol": symbol})
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in stock analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health_check():
    """
    ヘルスチェックエンドポイント（認証不要）
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "ClStock Secure API",
    }
