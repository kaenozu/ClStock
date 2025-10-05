#!/usr/bin/env python3
"""エントリーポイント: 厳選システムのクイック実行スクリプト."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from ClStock.utils.logger_config import get_logger
from ClStock.systems.resource_monitor import ResourceMonitor


logger = get_logger(__name__)


def _load_watchlist() -> List[Dict[str, str]]:
    """サンプルの厳選銘柄リストを読み込む."""
    sample_file = Path("data/watchlists/selective_watchlist.json")
    if sample_file.exists():
        try:
            return json.loads(sample_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
            logger.warning("ウォッチリストのJSON読み込みに失敗しました: %s", exc)
    # フォールバック: 静的に定義した推奨リスト
    return [
        {"symbol": "6758.T", "name": "ソニーG", "reason": "AI事業の収益拡大"},
        {"symbol": "7203.T", "name": "トヨタ", "reason": "EV戦略の加速"},
        {"symbol": "8306.T", "name": "三菱UFJ", "reason": "配当利回り安定"},
    ]


def main() -> None:
    """厳選システムのサマリを出力する."""
    logger.info("Selective system を起動します")

    monitor = ResourceMonitor()
    usage = monitor.get_system_usage()

    print("\n=== Selective System ===")
    print(f"起動時刻: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"CPU使用率: {usage.cpu_percent:.1f}% / メモリ使用率: {usage.memory_percent:.1f}%")

    watchlist = _load_watchlist()
    print("\n推奨ウォッチリスト (上位3件)")
    for idx, info in enumerate(watchlist, start=1):
        print(f" {idx}. {info['symbol']} - {info['name']} ({info['reason']})")

    print("\nシステム診断: 正常終了")


if __name__ == "__main__":
    main()
