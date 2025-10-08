"""
ポートフォリオ価値計算ロジックの簡易テスト
資産の二重計上問題が修正されたことを確認する
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.risk_management import PortfolioRiskManager


def simple_test():
    print("簡易テスト開始")

    # 初期資本100万円でリスクマネージャーを初期化
    risk_manager = PortfolioRiskManager(initial_capital=1000000)
    print(
        f"初期: 現金={risk_manager.current_capital:,}, ポジション数={len(risk_manager.positions)}, ポトフォリオ価値={risk_manager.calculate_portfolio_value():,}"
    )

    # 購入: 3000円の株を100株購入 (30万円)
    risk_manager.update_position("7203", 100, 3000, cash_flow=-300000)
    print(
        f"購入後: 現金={risk_manager.current_capital:,}, ポジション数={len(risk_manager.positions)}, ポトフォリオ価値={risk_manager.calculate_portfolio_value():,}"
    )

    # 売却: 50株を3200円で売却 (16万円)
    risk_manager.update_position("7203", 50, 3200, cash_flow=160000)
    print(
        f"売却後: 現金={risk_manager.current_capital:,}, ポジション数={len(risk_manager.positions)}, ポトフォリオ価値={risk_manager.calculate_portfolio_value():,}"
    )

    # 検証
    expected_value = risk_manager.current_capital + (
        50 * 3200
    )  # 現金 + 残りのポジション価値
    actual_value = risk_manager.calculate_portfolio_value()

    print(
        f"検証: 計算値({expected_value:,}) == 実際({actual_value:,}) -> {expected_value == actual_value}"
    )

    print("簡易テスト完了")


def test_with_debug_info():
    print("\nデバッグ情報付きテスト開始")

    # 初期資本100万円でリスクマネージャーを初期化
    risk_manager = PortfolioRiskManager(initial_capital=1000000)

    print("=== 初期状態 ===")
    print(f"初期資本: {risk_manager.initial_capital:,}")
    print(f"現金: {risk_manager.current_capital:,}")
    print(f"ポジション: {risk_manager.positions}")
    print(f"ポートフォリオ価値: {risk_manager.calculate_portfolio_value():,}")
    print()

    # 購入: 3000円の株を100株購入 (30万円)
    print("=== 100株購入 (30万円) ===")
    risk_manager.update_position("7203", 100, 3000, cash_flow=-300000)

    print(f"現金: {risk_manager.current_capital:,}")
    print(f"ポジション: {risk_manager.positions}")
    print(f"ポートフォリオ価値: {risk_manager.calculate_portfolio_value():,}")

    # 各ポジションの詳細を表示
    total_pos_value = 0
    for symbol, pos_data in risk_manager.positions.items():
        if isinstance(pos_data, dict):
            value = pos_data.get("value", 0)
            print(
                f"  {symbol}: {pos_data.get('quantity')}株 × {pos_data.get('price'):,}円 = {value:,}円"
            )
            total_pos_value += value
        else:
            # tupleの可能性も考慮
            print(f"  {symbol}: pos_data is {type(pos_data)} = {pos_data}")

    expected_total = risk_manager.current_capital + total_pos_value
    print(
        f"検証: 現金({risk_manager.current_capital:,}) + ポジション({total_pos_value:,}) = {expected_total:,}"
    )
    print(f"整合性: {risk_manager.calculate_portfolio_value() == expected_total}")
    print()

    # 売却: 50株を3200円で売却 (16万円)
    print("=== 50株売却 (16万円) ===")
    risk_manager.update_position("7203", 50, 3200, cash_flow=160000)

    print(f"現金: {risk_manager.current_capital:,}")
    print(f"ポジション: {risk_manager.positions}")
    print(f"ポートフォリオ価値: {risk_manager.calculate_portfolio_value():,}")

    # 各ポジションの詳細を表示
    total_pos_value = 0
    for symbol, pos_data in risk_manager.positions.items():
        if isinstance(pos_data, dict):
            value = pos_data.get("value", 0)
            print(
                f"  {symbol}: {pos_data.get('quantity')}株 × {pos_data.get('price'):,}円 = {value:,}円"
            )
            total_pos_value += value
        else:
            print(f"  {symbol}: pos_data is {type(pos_data)} = {pos_data}")

    expected_total = risk_manager.current_capital + total_pos_value
    print(
        f"検証: 現金({risk_manager.current_capital:,}) + ポジション({total_pos_value:,}) = {expected_total:,}"
    )
    print(f"整合性: {risk_manager.calculate_portfolio_value() == expected_total}")


if __name__ == "__main__":
    simple_test()
    test_with_debug_info()
    print("\nテスト完了！")
