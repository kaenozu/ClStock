#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
システム整理後の動作確認テスト
"""

import sys
import os


def test_imports():
    """主要モジュールのインポートテスト"""
    print("=== インポートテスト ===")

    try:
        # 87%精度システム
        from models.precision.precision_87_system import (
            Precision87BreakthroughSystem,
        )

        print("[OK] 87%精度システム")

        # データプロバイダー
        from data.stock_data import StockDataProvider

        print("[OK] データプロバイダー")

        # デモ取引システム
        from trading.demo_trader import DemoTrader

        print("[OK] デモ取引システム")

        return True

    except Exception as e:
        print(f"[ERROR] インポートエラー: {e}")
        return False


def test_file_structure():
    """ファイル構造テスト"""
    print("\n=== ファイル構造テスト ===")

    required_files = [
        "demo_start.py",
        "menu.py",
        "clstock_cli.py",
        "investment_advisor_cui.py",
    ]

    archived_files = [
        "archive/old_systems/optimal_30_prediction_system.py",
        "archive/tests/test_precision_87_system.py",
    ]

    all_good = True

    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file}: 存在")
        else:
            print(f"[ERROR] {file}: 存在しない")
            all_good = False

    for file in archived_files:
        if os.path.exists(file):
            print(f"[OK] {file}: アーカイブ済み")
        else:
            print(f"[ERROR] {file}: アーカイブされていない")
            all_good = False

    return all_good


def test_demo_functionality():
    """デモ機能の簡易テスト"""
    print("\n=== デモ機能テスト ===")

    try:
        from models.precision.precision_87_system import (
            Precision87BreakthroughSystem,
        )

        # システム初期化
        system = Precision87BreakthroughSystem()
        print("[OK] 87%精度システム初期化")

        # 予測テスト（1銘柄のみ）
        result = system.predict_with_87_precision("7203")

        if "final_prediction" in result:
            print(f"[OK] 予測実行 (予測値: {result['final_prediction']:.1f})")
            print(f"   信頼度: {result['final_confidence']:.1%}")
            print(f"   精度: {result['final_accuracy']:.1f}%")
            return True
        else:
            print("[ERROR] 予測結果の形式が不正")
            return False

    except Exception as e:
        print(f"[ERROR] デモ機能エラー: {e}")
        return False


def main():
    """メインテスト実行"""
    print("ClStock システム整理後動作確認")
    print("=" * 50)

    tests = [
        ("インポート", test_imports),
        ("ファイル構造", test_file_structure),
        ("デモ機能", test_demo_functionality),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"[ERROR] {name}テストでエラー: {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    print("テスト結果サマリー")
    print("=" * 50)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{name}: {status}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n[SUCCESS] 全テスト成功！システムは正常に動作しています。")
    else:
        print("\n[WARNING] 一部テストが失敗しました。")

    return all_passed


if __name__ == "__main__":
    main()
