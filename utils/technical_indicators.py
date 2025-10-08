#!/usr/bin/env python3
"""テクニカル指標計算ユーティリティ
重複する計算関数を統合
"""

import pandas as pd


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) 計算"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception:
        return pd.Series([50] * len(prices), index=prices.index)


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD計算"""
    try:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line

        return pd.DataFrame(
            {"MACD": macd, "Signal": signal_line, "Histogram": histogram},
        )
    except Exception:
        return pd.DataFrame(
            {
                "MACD": [0] * len(prices),
                "Signal": [0] * len(prices),
                "Histogram": [0] * len(prices),
            },
            index=prices.index,
        )


def calculate_bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    std_dev: float = 2,
) -> pd.DataFrame:
    """ボリンジャーバンド計算"""
    try:
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        return pd.DataFrame(
            {
                "Upper": sma + (std * std_dev),
                "Middle": sma,
                "Lower": sma - (std * std_dev),
            },
        )
    except Exception:
        return pd.DataFrame({"Upper": prices, "Middle": prices, "Lower": prices})


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """ストキャスティクス計算"""
    try:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return pd.DataFrame({"%K": k_percent, "%D": d_percent})
    except Exception:
        return pd.DataFrame(
            {"%K": [50] * len(close), "%D": [50] * len(close)},
            index=close.index,
        )


def calculate_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ウィリアムズ%R計算"""
    try:
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return wr
    except Exception:
        return pd.Series([-50] * len(close), index=close.index)


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range計算"""
    try:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr
    except Exception:
        return pd.Series([1.0] * len(close), index=close.index)


def calculate_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """モメンタム計算"""
    try:
        return prices / prices.shift(period) - 1
    except Exception:
        return pd.Series([0] * len(prices), index=prices.index)


def calculate_roc(prices: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change計算"""
    try:
        return ((prices - prices.shift(period)) / prices.shift(period)) * 100
    except Exception:
        return pd.Series([0] * len(prices), index=prices.index)


def calculate_keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """ケルトナーチャネル計算"""
    try:
        typical_price = (high + low + close) / 3
        middle_band = typical_price.ewm(span=window, adjust=False).mean()
        atr = calculate_atr(high, low, close, period=window)

        upper_band = middle_band + (multiplier * atr)
        lower_band = middle_band - (multiplier * atr)

        return pd.DataFrame(
            {"Upper": upper_band, "Middle": middle_band, "Lower": lower_band},
        )
    except Exception:
        return pd.DataFrame({"Upper": close, "Middle": close, "Lower": close})
