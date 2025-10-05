from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.security import HTTPAuthorizationCredentials

from datetime import datetime, time
from zoneinfo import ZoneInfo
import pandas as pd

from utils.logger_config import get_logger

logger = get_logger(__name__)

# セキュリティと検証機能
from api.security import security, verify_token
from utils.validators import (
    validate_stock_symbol,
    validate_period,
    ValidationError,
    log_validation_error,
)
from utils.exceptions import DataFetchError

from models.legacy_core import MLStockPredictor
from models.recommendation import StockRecommendation
from api.schemas import RecommendationResponse, StockRecommendationSchema, EntryPriceSchema, TargetSchema
from data.stock_data import StockDataProvider


router = APIRouter()


def stock_recommendation_to_schema(stock_rec: StockRecommendation) -> StockRecommendationSchema:
    """
    Convert StockRecommendation dataclass to StockRecommendationSchema pydantic model.
    """
    # holding_period (str) -> holding_period_days (int) の変換 (例: "1～2か月" -> 平均して30+45=75日など)
    # 簡易的な変換ロジック (より正確な変換が必要な場合は修正)
    period_str = stock_rec.holding_period
    if "1～2か月" in period_str:
        holding_period_days = 45
    elif "2～3か月" in period_str:
        holding_period_days = 75
    elif "3～4か月" in period_str:
        holding_period_days = 105
    else:
        # デフォルトまたは不明な場合は、数値以外の文字列から日数を推定 (簡易)
        try:
            # "1か月" から "1" を取り出し、30日として計算
            import re
            matches = re.findall(r'(\d+)', period_str)
            if matches:
                months = int(matches[0])  # 最初に見つかった数字を月数とする
                holding_period_days = months * 30
            else:
                holding_period_days = 30  # デフォルト
        except:
            holding_period_days = 30  # エラー時デフォルト

    # target_price (float) -> entry_price (object) の変換 (価格帯をどう設定するか)
    # 簡易的に、target_priceを中央値として、±5%の範囲をmin/maxとする
    price_range_pct = 0.05
    entry_price_min = stock_rec.target_price * (1 - price_range_pct)
    entry_price_max = stock_rec.target_price * (1 + price_range_pct)
    entry_price = EntryPriceSchema(min=entry_price_min, max=entry_price_max)

    # profit_target_1, profit_target_2 (float) -> targets (List[object]) の変換
    targets = [
        TargetSchema(label="base", price=stock_rec.profit_target_1),
        TargetSchema(label="stretch", price=stock_rec.profit_target_2),
    ]

    # recommendation_level (str) -> action (str) のマッピング (例: "strong_buy" -> "buy")
    # _score_to_level メソッド (models/legacy_core.py内) を参考にする
    # "strong_buy", "buy", "neutral", "watch", "avoid"
    rec_level = stock_rec.recommendation_level
    if rec_level in ["strong_buy", "buy"]:
        action = "buy"
    elif rec_level in ["avoid"]:
        action = "sell"  # "avoid" は sell に近いと見なす
    else: # "neutral", "watch"
        action = "hold"

    # buy_timing (str) -> entry_condition または action_text
    # buy_timing は "押し目買いを検討" など、action_textに近いが、entry_conditionとして扱う
    entry_condition = stock_rec.buy_timing
    action_text = None # APIレスポンスには含まれない (CUI用)

    # sector はデータソースから取得 (例: data_provider)
    # StockRecommendation オブジェクトには含まれていないので、関数の外から渡すかデフォルト値
    # 今回はデフォルト値 (None) とする。必要に応じて data_provider から取得するロジックを追加。

    return StockRecommendationSchema(
        rank=stock_rec.rank,
        symbol=stock_rec.symbol,
        name=stock_rec.company_name, # company_name -> name
        sector=None, # デフォルト
        score=stock_rec.score,
        action=action,
        action_text=action_text,
        entry_condition=entry_condition,
        entry_price=entry_price,
        stop_loss=stock_rec.stop_loss,
        targets=targets,
        holding_period_days=holding_period_days, # holding_period (str) -> holding_period_days (int)
        confidence=None, # デフォルト
        rationale=stock_rec.recommendation_reason, # recommendation_reason -> rationale
        notes=None, # デフォルト
        risk_level=None, # デフォルト
        chart_refs=None, # デフォルト
        current_price=stock_rec.current_price,
    )


@router.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    top_n: int = Query(10, ge=1, le=50, description="推奨銘柄の上位N件を取得"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        # Authentication verification
        verify_token(credentials.credentials)

        predictor = MLStockPredictor()
        recommendations_raw = predictor.get_top_recommendations(top_n)
        recommendations = [stock_recommendation_to_schema(rec) for rec in recommendations_raw]

        current_time = datetime.now(ZoneInfo("Asia/Tokyo"))
        market_open_time = time(9, 0)
        market_close_time = time(15, 0)
        is_weekend = current_time.weekday() >= 5
        is_market_hours = (
            market_open_time <= current_time.time() < market_close_time
        )

        return RecommendationResponse(
            items=recommendations, # recommendations -> items に変更
            generated_at=current_time,
            market_status=(
                "市場営業時間外"
                if is_weekend or not is_market_hours
                else "市場営業中"
            ),
            top_n=top_n, # top_n フィールドも追加
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


@router.get("/recommendation/{symbol}", response_model=StockRecommendationSchema) # response_model 追加
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

        return stock_recommendation_to_schema(recommendation) # 変換して返す
    except InvalidSymbolError as e:
        raise HTTPException(
            status_code=404, detail=f"銘柄コード {symbol} が見つかりません"
        )
    except PredictionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"推奨情報の取得中にエラーが発生しました: {str(e)}")
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
        actual_ticker_value = lookup_symbol
        if "ActualTicker" in technical_data.columns:
            latest_actual_ticker = technical_data["ActualTicker"].dropna()
            if not latest_actual_ticker.empty:
                actual_ticker_value = str(latest_actual_ticker.iloc[-1])
        raw_financial_metrics = (
            data_provider.get_financial_metrics(lookup_symbol) or {}
        )
        company_name = data_provider.jp_stock_codes.get(
            lookup_symbol, lookup_symbol
        )
        financial_metrics = dict(raw_financial_metrics)
        financial_metrics["symbol"] = validated_symbol
        if "company_name" in financial_metrics or company_name != lookup_symbol:
            financial_metrics["company_name"] = company_name
        if not financial_metrics.get("actual_ticker"):
            financial_metrics["actual_ticker"] = actual_ticker_value

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
            "company_name": company_name,
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
