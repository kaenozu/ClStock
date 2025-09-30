import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple

import pandas as pd

from config.settings import get_settings
from utils.exceptions import DataFetchError


def _normalized_symbol_seed(symbol: str) -> int:
    """Return a deterministic, non-negative seed for a ticker symbol."""

    return abs(hash(symbol)) % (2**32)

# yfinance を try-except でインポート
try:
    import yfinance as yf
except ModuleNotFoundError:
    # yfinance が見つからない場合、ダミーのクラスを提供
    class yf:
        @staticmethod
        def Ticker(symbol):