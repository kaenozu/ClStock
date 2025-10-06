"""Utility to build Google Colab helper scripts for manual data retrieval of Japanese stocks.

This module is specifically designed for Japanese stock codes (e.g., 7203.T).
It automatically appends the .T suffix to numeric stock codes to indicate the Tokyo Stock Exchange.
"""

from pathlib import Path
from textwrap import dedent
from typing import Iterable, List


def _format_symbol(symbol: str) -> str:
    """Clean and normalize a Japanese stock symbol, ensuring it has the .T suffix.

    This function is specifically designed for Japanese stock codes.
    For Japanese stock codes, which are typically numeric, the '.T' suffix is appended
    to indicate the Tokyo Stock Exchange.

    Args:
        symbol: A Japanese stock symbol (e.g., "7203" or "7203.T")

    Returns:
        A normalized Japanese stock symbol with .T suffix (e.g., "7203.T")

    Examples:
        >>> _format_symbol("7203")
        '7203.T'
        >>> _format_symbol("7203.T")
        '7203.T'
        >>> _format_symbol(" 7203 ")
        '7203.T'

    """
    clean = symbol.strip().upper()

    # 日本株式コードの検証（数字のみ、または数字+.T）
    if clean.isdigit():
        # 数字のみの場合、.Tを追加
        clean += ".T"
    elif clean.endswith(".T") and clean[:-2].isdigit():
        # 数字+.Tの場合、そのまま使用
        pass
    else:
        # その他の形式（.T以外の接尾辞や、数字以外を含むなど）は警告
        print(
            f"Warning: Symbol '{symbol}' may not be a valid Japanese stock code. Expected numeric code with optional .T suffix.",
        )

    return clean


def generate_colab_data_retrieval_script(
    missing_symbols: Iterable[str],
    period: str = "1y",
    output_dir: str = "data",
) -> str:
    """Return a small Python script for downloading historical data of Japanese stocks in Colab.

    This function is specifically designed for Japanese stock codes (e.g., 7203.T).
    The script relies on :mod:`yfinance` and saves CSV files into ``output_dir``.
    For Japanese stock codes (numeric only), the '.T' suffix is automatically appended
    to indicate the Tokyo Stock Exchange.

    Args:
        missing_symbols: An iterable of Japanese stock symbols (e.g., ["7203", "6758.T"])
        period: The period for which to retrieve data (default: "1y")
        output_dir: The directory to save the CSV files (default: "data")

    Returns:
        A Python script as a string for downloading historical data in Google Colab

    Examples:
        >>> script = generate_colab_data_retrieval_script(["7203", "6758.T"])
        >>> print(script)  # 生成されたスクリプトを表示

    """
    symbols: List[str] = [
        _format_symbol(symbol) for symbol in missing_symbols if symbol
    ]
    if not symbols:
        return ""

    output_path = Path(output_dir).as_posix()
    symbol_list = ", ".join(f'"{symbol}"' for symbol in symbols)

    # 新しいスクリプトテンプレート（再試行処理とAPI制限対策を含む）
    script_template = f"""
    from pathlib import Path
    import yfinance as yf
    import requests
    import time

    SYMBOLS = [{symbol_list}]
    PERIOD = "{period}"
    OUTPUT_DIR = Path(r"{output_path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def download(symbol: str) -> None:
        ticker = symbol  # _format_symbol によって .T が既に追加されている
        print(f"Fetching {{ticker}} for period {{PERIOD}}...")
        
        # 再試行回数
        max_retries = 3
        retry_delay = 5  # 秒
        
        for attempt in range(max_retries):
            try:
                data = yf.Ticker(ticker).history(period=PERIOD)
                if data.empty:
                    print(f"No data returned for {{ticker}}")
                    return
                file_path = OUTPUT_DIR / f"{{symbol}}_{{PERIOD}}.csv"
                data.to_csv(file_path)
                print(f"Saved {{file_path}}")
                return  # 成功したらループを抜ける
            except requests.exceptions.RequestException as e:
                # ネットワークエラー（API制限を含む）
                if "429" in str(e) or "Too Many Requests" in str(e):
                    print(f"API rate limit exceeded for {{ticker}}. Waiting {{retry_delay}} seconds before retry...")
                    if attempt < max_retries - 1:  # 最後の試行でなければ待機
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 次回は2倍の待機時間
                        continue
                print(f"Network error occurred while fetching data for {{ticker}} (attempt {{attempt + 1}}/{{max_retries}}): {{e}}")
                if attempt < max_retries - 1:  # 最後の試行でなければ待機
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 次回は2倍の待機時間
                    continue
            except Exception as e:
                # その他のエラー
                print(f"An unexpected error occurred while fetching data for {{ticker}} (attempt {{attempt + 1}}/{{max_retries}}): {{e}}")
                if attempt < max_retries - 1:  # 最後の試行でなければ待機
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 次回は2倍の待機時間
                    continue
        
        # 全ての再試行が失敗した場合
        print(f"Failed to fetch data for {{ticker}} after {{max_retries}} attempts. Skipping...")

    for sym in SYMBOLS:
        download(sym)
    """

    return dedent(script_template).strip() + "\n"
