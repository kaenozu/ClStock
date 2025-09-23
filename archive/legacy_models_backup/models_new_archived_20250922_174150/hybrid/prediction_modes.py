#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
予測モード定義
循環インポート回避のため分離
"""

from enum import Enum

class PredictionMode(Enum):
    """予測モード"""
    # 既存モード
    SPEED_PRIORITY = "speed"      # 速度優先（拡張システム）
    ACCURACY_PRIORITY = "accuracy"  # 精度優先（87%システム）
    BALANCED = "balanced"         # バランス（ハイブリッド）
    AUTO = "auto"                # 自動選択

    # 次世代モード（Phase 1追加）
    ULTRA_SPEED = "ultra_speed"      # 0.001秒応答（HFT向け）
    RESEARCH_MODE = "research"       # 95%精度目標（精密分析）
    SWING_TRADE = "swing"           # 中期トレード最適化
    SCALPING = "scalping"           # スキャルピング特化
    PORTFOLIO_ANALYSIS = "portfolio" # ポートフォリオ全体最適化
    RISK_MANAGEMENT = "risk"        # リスク管理特化