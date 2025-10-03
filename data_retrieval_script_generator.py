"""Utility to build Google Colab helper scripts for manual data retrieval."""

from pathlib import Path
from textwrap import dedent
from typing import Iterable, List
import requests  # 追加


def _format_symbol(symbol: str) -> str:
    """Clean and normalize a stock symbol, ensuring Japanese stocks have the .T suffix.

    For Japanese stock codes, which are typically numeric, the '.T' suffix is appended
    to indicate the Tokyo Stock Exchange.
    """
    clean = symbol.strip().upper()
    # Append .T suffix for Japanese stock codes (assumed to be numeric)
    if clean.isdigit() and not clean.endswith(".T"):
        clean += ".T"
    return clean


def generate_colab_data_retrieval_script(
    missing_symbols: Iterable[str],
    period: str = "1y",
    output_dir: str = "data",
) -> str:
    """Return a small Python script for downloading historical data in Colab.

    The script relies on :mod:`yfinance` and saves CSV files into ``output_dir``.
    For Japanese stock codes (numeric only), the '.T' suffix is automatically appended
    to indicate the Tokyo Stock Exchange.
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
    import requests  # 追加

    SYMBOLS = [{symbol_list}]
    PERIOD = "{period}"
    OUTPUT_DIR = Path(r"{output_path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def download(symbol: str) -> None:
        ticker = symbol  # _format_symbol によって .T が既に追加されている
        print(f"Fetching {{ticker}} for period {{PERIOD}}...")
        try:
            data = yf.Ticker(ticker).history(period=PERIOD)
            if data.empty:
                print(f"No data returned for {{ticker}}")
                return
            file_path = OUTPUT_DIR / f"{{symbol}}_{{PERIOD}}.csv"
            data.to_csv(file_path)
            print(f"Saved {{file_path}}")
        except requests.exceptions.RequestException as e:  # ネットワークエラーなど
            print(f"Network error occurred while fetching data for {{ticker}}: {{e}}")
        except Exception as e:  # その他のエラー
            print(f"An unexpected error occurred while fetching data for {{ticker}}: {{e}}")

    for sym in SYMBOLS:
        download(sym)
    """

    return dedent(script).strip() + "\n"