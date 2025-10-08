"""完全自動投資システムのCLIエントリーポイント。"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import List, Optional

from .system import run_full_auto

__all__ = ["build_cli_parser", "main"]


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full auto investment pipeline",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Limit the number of tickers processed (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--prefer-local-data",
        action="store_true",
        help="Prioritize local CSV data before calling yfinance.",
    )
    parser.add_argument(
        "--skip-local-data",
        action="store_true",
        help="Force yfinance downloads even if local CSV data exists.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    if args.prefer_local_data and args.skip_local_data:
        parser.error(
            "--prefer-local-data and --skip-local-data cannot be used together.",
        )

    if args.max_tickers is not None and args.max_tickers <= 0:
        parser.error("--max-tickers must be a positive integer.")

    if args.prefer_local_data:
        os.environ["CLSTOCK_PREFER_LOCAL_DATA"] = "1"
    elif args.skip_local_data:
        os.environ["CLSTOCK_PREFER_LOCAL_DATA"] = "0"

    try:
        asyncio.run(run_full_auto(max_symbols=args.max_tickers))
    except KeyboardInterrupt:
        print("[INFO] Full auto run interrupted by user.")
        return 130

    return 0
