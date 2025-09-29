"""Utility to build Google Colab helper scripts for manual data retrieval."""

from pathlib import Path
from textwrap import dedent
from typing import Iterable, List


def _format_symbol(symbol: str) -> str:
    clean = symbol.strip().upper()
    return clean


def generate_colab_data_retrieval_script(
    missing_symbols: Iterable[str],
    period: str = "1y",
    output_dir: str = "data",
) -> str:
    """Return a small Python script for downloading historical data in Colab.

    The script relies on :mod:`yfinance` and saves CSV files into ``output_dir``.
    """

    symbols: List[str] = [
        _format_symbol(symbol) for symbol in missing_symbols if symbol
    ]
    if not symbols:
        return ""

    output_path = Path(output_dir).as_posix()
    symbol_list = ", ".join(f'"{symbol}"' for symbol in symbols)

    script = f"""
    from pathlib import Path

    import yfinance as yf

    SYMBOLS = [{symbol_list}]
    PERIOD = "{period}"
    OUTPUT_DIR = Path(r"{output_path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def download(symbol: str) -> None:
        ticker = symbol if symbol.endswith(".T") else f"{symbol}.T"
        print(f"Fetching {{ticker}} for period {{PERIOD}}...")
        data = yf.Ticker(ticker).history(period=PERIOD)
        if data.empty:
            print(f"No data returned for {{ticker}}")
            return
        file_path = OUTPUT_DIR / f"{{symbol}}_{{PERIOD}}.csv"
        data.to_csv(file_path)
        print(f"Saved {{file_path}}")

    for sym in SYMBOLS:
        download(sym)
    """

    return dedent(script).strip() + "\n"
