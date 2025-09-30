import pandas as pd
import numpy as np

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common technical indicators (SMA, RSI, MACD, Bollinger Bands).
    Assumes 'Close' column is present in the DataFrame.
    """
    df_copy = df.copy()

    # Simple Moving Average (SMA)
    df_copy["SMA_5"] = df_copy["Close"].rolling(window=5).mean()
    df_copy["SMA_20"] = df_copy["Close"].rolling(window=20).mean()

    # Relative Strength Index (RSI)
    delta = df_copy["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df_copy["RSI"] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp1 = df_copy["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df_copy["Close"].ewm(span=26, adjust=False).mean()
    df_copy["MACD"] = exp1 - exp2
    df_copy["Signal_Line"] = df_copy["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df_copy["BB_Middle"] = df_copy["Close"].rolling(window=20).mean()
    df_copy["BB_Upper"] = df_copy["BB_Middle"] + (df_copy["Close"].rolling(window=20).std() * 2)
    df_copy["BB_Lower"] = df_copy["BB_Middle"] - (df_copy["Close"].rolling(window=20).std() * 2)

    return df_copy
