"""入力検証ユーティリティ
セキュリティ強化のための検証関数群
"""

import logging
import re
from collections.abc import Mapping
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """検証エラー"""



def validate_stock_symbol(symbol: str) -> str:
    """株式銘柄コードの検証

    Args:
        symbol: 検証する銘柄コード

    Returns:
        str: 正規化された銘柄コード

    Raises:
        ValidationError: 無効な銘柄コードの場合

    """
    if not symbol:
        raise ValidationError("Stock symbol cannot be empty")

    # 文字列に変換
    symbol = str(symbol).strip().upper()

    # 基本的な形式チェック（英数字とハイフン、ドットのみ）
    if not re.match(r"^[A-Z0-9\.\-]+$", symbol):
        raise ValidationError(f"Invalid stock symbol format: {symbol}")

    # 長さチェック
    if not (1 <= len(symbol) <= 20):
        raise ValidationError(f"Stock symbol length must be 1-20 characters: {symbol}")

    return symbol


def validate_period(period: str) -> str:
    """期間パラメータの検証

    Args:
        period: 検証する期間

    Returns:
        str: 検証済み期間

    Raises:
        ValidationError: 無効な期間の場合

    """
    if not period:
        raise ValidationError("Period cannot be empty")

    # 許可された期間のリスト
    valid_periods = [
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    ]

    period = str(period).lower().strip()

    if period not in valid_periods:
        raise ValidationError(
            f"Invalid period: {period}. Valid periods: {', '.join(valid_periods)}",
        )

    return period


def validate_numeric_range(
    value: Union[str, float],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    field_name: str = "value",
) -> float:
    """数値範囲の検証

    Args:
        value: 検証する値
        min_val: 最小値
        max_val: 最大値
        field_name: フィールド名（エラーメッセージ用）

    Returns:
        float: 検証済み数値

    Raises:
        ValidationError: 無効な数値の場合

    """
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        raise ValidationError(f"Invalid {field_name}: must be a number")

    if min_val is not None and num_value < min_val:
        raise ValidationError(f"{field_name} must be >= {min_val}")

    if max_val is not None and num_value > max_val:
        raise ValidationError(f"{field_name} must be <= {max_val}")

    return num_value


def validate_email(email: str) -> str:
    """メールアドレスの検証

    Args:
        email: 検証するメールアドレス

    Returns:
        str: 正規化されたメールアドレス

    Raises:
        ValidationError: 無効なメールアドレスの場合

    """
    if not email:
        raise ValidationError("Email cannot be empty")

    email = str(email).strip().lower()

    # 基本的なメール形式チェック
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_pattern, email):
        raise ValidationError(f"Invalid email format: {email}")

    # 長さチェック
    if len(email) > 254:
        raise ValidationError("Email address too long")

    return email


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """文字列のサニタイズ

    Args:
        value: サニタイズする文字列
        max_length: 最大長

    Returns:
        str: サニタイズされた文字列

    Raises:
        ValidationError: 無効な文字列の場合

    """
    if not isinstance(value, str):
        value = str(value)

    # 制御文字を除去
    value = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", value)

    # 長さチェック
    if len(value) > max_length:
        raise ValidationError(f"String too long: maximum {max_length} characters")

    return value.strip()


def validate_api_key(api_key: str) -> str:
    """API キーの検証

    Args:
        api_key: 検証するAPI キー

    Returns:
        str: 検証済みAPI キー

    Raises:
        ValidationError: 無効なAPI キーの場合

    """
    if not api_key:
        raise ValidationError("API key cannot be empty")

    api_key = str(api_key).strip()

    # 基本的な形式チェック（英数字とハイフンのみ）
    if not re.match(r"^[a-zA-Z0-9\-_]+$", api_key):
        raise ValidationError("Invalid API key format")

    # 長さチェック
    if not (8 <= len(api_key) <= 64):
        raise ValidationError("API key length must be 8-64 characters")

    return api_key


def validate_symbols_list(symbols: List[str], max_count: int = 100) -> List[str]:
    """銘柄コードリストの検証

    Args:
        symbols: 銘柄コードのリスト
        max_count: 最大件数

    Returns:
        List[str]: 検証済み銘柄コードリスト

    Raises:
        ValidationError: 無効なリストの場合

    """
    if not symbols:
        raise ValidationError("Symbols list cannot be empty")

    if not isinstance(symbols, list):
        raise ValidationError("Symbols must be a list")

    if len(symbols) > max_count:
        raise ValidationError(f"Too many symbols: maximum {max_count}")

    validated_symbols = []
    for symbol in symbols:
        validated_symbols.append(validate_stock_symbol(symbol))

    # 重複を除去
    unique_symbols = list(dict.fromkeys(validated_symbols))

    return unique_symbols


def validate_portfolio_allocations(
    allocations: Mapping[str, Union[int, float]],
    *,
    tolerance: float = 1e-6,
) -> Dict[str, float]:
    """ポートフォリオ配分の検証

    Args:
        allocations: 銘柄コードと配分値のマッピング
        tolerance: 合計が1となるために許容される誤差

    Returns:
        Dict[str, float]: 検証済みで正規化された配分

    Raises:
        ValidationError: 入力が無効な場合
    """

    if not isinstance(allocations, Mapping):
        raise ValidationError("Allocations must be a mapping of symbols to weights")

    if not allocations:
        raise ValidationError("Allocations cannot be empty")

    normalized_allocations: Dict[str, float] = {}
    total_weight = 0.0

    for symbol, weight in allocations.items():
        validated_symbol = validate_stock_symbol(symbol)

        try:
            weight_value = float(weight)
        except (TypeError, ValueError):
            raise ValidationError(
                f"Allocation for {validated_symbol} must be a number"
            ) from None

        if weight_value < 0:
            raise ValidationError(
                f"Allocation for {validated_symbol} must be non-negative"
            )

        normalized_allocations[validated_symbol] = weight_value
        total_weight += weight_value

    if total_weight <= 0:
        raise ValidationError("Total allocation must be greater than zero")

    if abs(total_weight - 1.0) > tolerance:
        raise ValidationError(
            "Total allocation must sum to 1.0 within tolerance "
            f"{tolerance}, got {total_weight:.6f}"
        )

    return {
        symbol: weight / total_weight for symbol, weight in normalized_allocations.items()
    }


def validate_date_range(
    start_date: Optional[str] = None, end_date: Optional[str] = None,
) -> tuple:
    """日付範囲の検証

    Args:
        start_date: 開始日 (YYYY-MM-DD)
        end_date: 終了日 (YYYY-MM-DD)

    Returns:
        tuple: (開始日, 終了日)

    Raises:
        ValidationError: 無効な日付の場合

    """
    from datetime import datetime

    date_format = "%Y-%m-%d"

    if start_date:
        try:
            start_dt = datetime.strptime(start_date, date_format)
        except ValueError:
            raise ValidationError(
                f"Invalid start date format: {start_date}. Use YYYY-MM-DD",
            )
    else:
        start_dt = None

    if end_date:
        try:
            end_dt = datetime.strptime(end_date, date_format)
        except ValueError:
            raise ValidationError(
                f"Invalid end date format: {end_date}. Use YYYY-MM-DD",
            )
    else:
        end_dt = None

    if start_dt and end_dt:
        if start_dt > end_dt:
            raise ValidationError("Start date must be before end date")

        # 期間が長すぎないかチェック（10年以内）
        if (end_dt - start_dt).days > 3650:
            raise ValidationError("Date range too long: maximum 10 years")

    return (start_date, end_date)


def log_validation_error(error: ValidationError, context: dict = None):
    """検証エラーをログに記録

    Args:
        error: 検証エラー
        context: 追加のコンテキスト情報

    """
    context = context or {}
    logger.warning(f"Validation error: {error!s}", extra=context)
