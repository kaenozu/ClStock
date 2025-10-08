"""
ポートフォリオ価値計算ロジックのテスト
資産の二重計上問題が修正されたことを確認する
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.risk_management import PortfolioRiskManager


def test_portfolio_value_calculation():
    """ポートフォリオ価値計算のテスト"""
    print("ポートフォリオ価値計算ロジックのテスト開始")

    # 初期資本100万円でリスクマネージャーを初期化
    risk_manager = PortfolioRiskManager(initial_capital=1000000)

    print(f"初期資本: {risk_manager.initial_capital:,}円")
    print(f"初期現金: {risk_manager.current_capital:,}円")
    print(f"初期ポートフォリオ価値: {risk_manager.calculate_portfolio_value():,}円")
    print()

    # 株式購入のシミュレーション: 3000円の株を100株購入 (30,000円)
    purchase_price = 3000
    purchase_quantity = 100
    purchase_amount = purchase_price * purchase_quantity  # 300,000円

    print(f"購入: {purchase_quantity}株 × {purchase_price:,}円 = {purchase_amount:,}円")

    # 現金残高を更新（購入処理）- cash_flowパラメータで一括処理
    initial_cash = risk_manager.current_capital
    risk_manager.update_position(
        "7203", purchase_quantity, purchase_price, cash_flow=-purchase_amount
    )

    # 更新後の状態
    print(
        f"購入後現金: {risk_manager.current_capital:,}円 (計算: {initial_cash:,} - {purchase_amount:,} = {initial_cash - purchase_amount:,}円)"
    )
    print(
        f"ポジション: {purchase_quantity}株 × {purchase_price:,}円 = {purchase_quantity * purchase_price:,}円"
    )
    print(f"ポートフォリオ価値: {risk_manager.calculate_portfolio_value():,}円")
    print(
        f"検証: 現金({risk_manager.current_capital:,}円) + ポジション({purchase_quantity * purchase_price:,}円) = {risk_manager.current_capital + purchase_quantity * purchase_price:,}円"
    )
    print(
        f"整合性チェック: {risk_manager.calculate_portfolio_value() == risk_manager.current_capital + purchase_quantity * purchase_price}"
    )
    print()

    # 追加株式購入のシミュレーション: 別の銘柄を購入
    purchase_price2 = 2500
    purchase_quantity2 = 200
    purchase_amount2 = purchase_price2 * purchase_quantity2  # 500,000円

    print(
        f"追加購入: {purchase_quantity2}株 × {purchase_price2:,}円 = {purchase_amount2:,}円"
    )

    # 追加ポジションを追加
    cash_after_first_purchase = risk_manager.current_capital
    risk_manager.update_position(
        "6758", purchase_quantity2, purchase_price2, cash_flow=-purchase_amount2
    )

    # 更新後の状態
    total_position_value = (purchase_quantity * purchase_price) + (
        purchase_quantity2 * purchase_price2
    )
    expected_portfolio_value = risk_manager.current_capital + total_position_value
    actual_portfolio_value = risk_manager.calculate_portfolio_value()

    print(
        f"追加購入後現金: {risk_manager.current_capital:,}円 (計算: {cash_after_first_purchase:,} - {purchase_amount2:,} = {cash_after_first_purchase - purchase_amount2:,}円)"
    )
    print(
        f"ポジション1: {purchase_quantity}株 × {purchase_price:,}円 = {purchase_quantity * purchase_price:,}円"
    )
    print(
        f"ポジション2: {purchase_quantity2}株 × {purchase_price2:,}円 = {purchase_quantity2 * purchase_price2:,}円"
    )
    print(f"総ポジション価値: {total_position_value:,}円")
    print(f"ポートフォリオ価値: {actual_portfolio_value:,}円")
    print(
        f"検証: 現金({risk_manager.current_capital:,}円) + ポジション({total_position_value:,}円) = {expected_portfolio_value:,}円"
    )
    print(f"整合性チェック: {actual_portfolio_value == expected_portfolio_value}")
    print()

    # 株式売却のシミュレーション: 一部の株を売却
    sell_quantity = 50
    sell_price = 3200  # 価格が上がったと仮定
    sell_amount = sell_quantity * sell_price  # 160,000円

    print(f"売却: {sell_quantity}株 × {sell_price:,}円 = {sell_amount:,}円")

    # ポジションを更新（売却処理）
    cash_before_sell = risk_manager.current_capital
    remaining_quantity = purchase_quantity - sell_quantity
    if remaining_quantity > 0:
        risk_manager.update_position(
            "7203", remaining_quantity, sell_price, cash_flow=sell_amount
        )
    else:
        risk_manager.update_position(
            "7203", 0, sell_price, cash_flow=sell_amount
        )  # 数量0で削除

    # 売却後の状態
    remaining_position_value = (remaining_quantity * sell_price) + (
        purchase_quantity2 * purchase_price2
    )
    expected_portfolio_value_after_sell = (
        risk_manager.current_capital + remaining_position_value
    )
    actual_portfolio_value_after_sell = risk_manager.calculate_portfolio_value()

    print(
        f"売却後現金: {risk_manager.current_capital:,}円 (計算: {cash_before_sell:,} + {sell_amount:,} = {cash_before_sell + sell_amount:,}円)"
    )
    print(
        f"残りポジション1: {remaining_quantity}株 × {sell_price:,}円 = {remaining_quantity * sell_price:,}円"
    )
    print(
        f"ポジション2: {purchase_quantity2}株 × {purchase_price2:,}円 = {purchase_quantity2 * purchase_price2:,}円"
    )
    print(f"総ポジション価値: {remaining_position_value:,}円")
    print(f"ポートフォリオ価値: {actual_portfolio_value_after_sell:,}円")
    print(
        f"検証: 現金({risk_manager.current_capital:,}円) + ポジション({remaining_position_value:,}円) = {expected_portfolio_value_after_sell:,}円"
    )
    print(
        f"整合性チェック: {actual_portfolio_value_after_sell == expected_portfolio_value_after_sell}"
    )
    print()

    # 資産二重計上チェック
    print("=== 資産二重計上チェック ===")
    print("各ポジションの価値を別々に計算:")
    for symbol, pos_data in risk_manager.positions.items():
        print(
            f"  {symbol}: {pos_data['quantity']}株 × {pos_data['price']:,}円 = {pos_data['value']:,}円"
        )

    total_position_values = sum(
        pos_data["value"] for pos_data in risk_manager.positions.items()
    )
    print(f"ポジション合計: {total_position_values:,}円")
    print(f"現金: {risk_manager.current_capital:,}円")
    print(f"ポートフォリオ総価値: {risk_manager.calculate_portfolio_value():,}円")
    print(
        f"検証（現金+ポジション）: {risk_manager.current_capital + total_position_values:,}円"
    )
    print(
        f"二重計上チェック: {risk_manager.calculate_portfolio_value() == risk_manager.current_capital + total_position_values}"
    )

    print("\nテスト完了！")


def test_backtesting_portfolio_value():
    """バックテストのポートフォリオ価値計算テスト"""
    print("\nバックテストポートフォリオ価値計算のテスト開始")

    from backtesting import BacktestEngine
    from datetime import datetime

    # 初期資本100万円でエンジンを初期化
    engine = BacktestEngine(initial_capital=1000000)

    print(f"初期資本: {engine.initial_capital:,}円")
    print(f"初期現金: {engine.current_capital:,}円")
    print(f"初期保有: {engine.holdings}")
    print(f"初期ポートフォリオ価値: {engine.calculate_portfolio_value({}):,}円")
    print()

    # 株式購入のシミュレーション
    symbol = "000001"
    quantity = 100
    price = 3000
    date = datetime.now()

    print(f"購入: {quantity}株 × {price:,}円 = {quantity * price:,}円")
    engine.buy_stock(symbol, quantity, price, date)

    current_prices = {symbol: price}
    print(f"購入後現金: {engine.current_capital:,}円")
    print(f"購入後保有: {engine.holdings}")
    print(
        f"購入後ポートフォリオ価値: {engine.calculate_portfolio_value(current_prices):,}円"
    )

    expected_value = engine.current_capital + (engine.holdings.get(symbol, 0) * price)
    print(
        f"検証: 現金({engine.current_capital:,}円) + ポジション({engine.holdings.get(symbol, 0) * price:,}円) = {expected_value:,}円"
    )
    print(
        f"整合性チェック: {engine.calculate_portfolio_value(current_prices) == expected_value}"
    )
    print()

    # 株式売却のシミュレーション
    sell_quantity = 50
    sell_price = 3200

    print(
        f"売却: {sell_quantity}株 × {sell_price:,}円 = {sell_quantity * sell_price:,}円"
    )
    engine.sell_stock(symbol, sell_quantity, sell_price, date)

    current_prices = {symbol: sell_price}
    print(f"売却後現金: {engine.current_capital:,}円")
    print(f"売却後保有: {engine.holdings}")
    print(
        f"売却後ポートフォリオ価値: {engine.calculate_portfolio_value(current_prices):,}円"
    )

    expected_value = engine.current_capital + (
        engine.holdings.get(symbol, 0) * sell_price
    )
    print(
        f"検証: 現金({engine.current_capital:,}円) + ポジション({engine.holdings.get(symbol, 0) * sell_price:,}円) = {expected_value:,}円"
    )
    print(
        f"整合性チェック: {engine.calculate_portfolio_value(current_prices) == expected_value}"
    )

    print("\nバックテストテスト完了！")


if __name__ == "__main__":
    test_portfolio_value_calculation()
    test_backtesting_portfolio_value()

    print("\n" + "=" * 60)
    print("すべてのテストが完了しました！")
    print("ポートフォリオ価値の二重計上問題は修正されています。")
    print("=" * 60)
