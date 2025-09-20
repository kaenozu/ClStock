#!/usr/bin/env python3
"""
ClStock デモ運用 簡単スタート
1週間のデモ取引を開始するための簡単なスクリプト
"""

import sys
import os
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def start_demo_trading():
    """デモ取引を開始する"""
    print("=" * 60)
    print("🚀 ClStock デモ運用システム")
    print("87%精度システムによる1週間のデモ取引を開始します")
    print("=" * 60)

    # 設定
    initial_money = 1000000  # 100万円でスタート
    target_stocks = ["7203", "6758", "8306", "6861", "9984"]  # 主要日本株

    print(f"💰 初期資金: {initial_money:,}円")
    print(f"📈 対象銘柄: {', '.join(target_stocks)}")
    print(f"🎯 使用システム: 87%精度突破システム")
    print(f"📅 期間: 1週間")
    print()

    # デモ取引システムの初期化（簡単版）
    try:
        print("📊 システム初期化中...")

        # 87%精度システム初期化
        from models_new.precision.precision_87_system import Precision87BreakthroughSystem
        precision_system = Precision87BreakthroughSystem()
        print("✅ 87%精度システム初期化完了")

        # データプロバイダー初期化
        from data.stock_data import StockDataProvider
        data_provider = StockDataProvider()
        print("✅ データプロバイダー初期化完了")

        # 簡単なデモ取引実行
        print("\n🔄 デモ取引シミュレーション開始...")

        portfolio = {
            "cash": initial_money,
            "positions": {},
            "trades": [],
            "daily_pnl": []
        }

        # 各銘柄で予測実行
        for symbol in target_stocks:
            print(f"\n📊 {symbol} 分析中...")

            try:
                # 87%精度予測実行
                result = precision_system.predict_with_87_precision(symbol)

                prediction = result['final_prediction']
                confidence = result['final_confidence']
                accuracy = result['final_accuracy']
                achieved_87 = result['precision_87_achieved']

                print(f"  💡 予測結果:")
                print(f"    価格予測: {prediction:.1f}")
                print(f"    信頼度: {confidence:.1%}")
                print(f"    推定精度: {accuracy:.1f}%")
                print(f"    87%達成: {'✅ YES' if achieved_87 else '❌ NO'}")

                # 取引判断（簡単版）
                if achieved_87 and confidence > 0.7:
                    # 高精度・高信頼度の場合は取引実行
                    position_size = min(100000, portfolio["cash"] * 0.1)  # 最大10万円または資金の10%

                    if position_size > 10000 and portfolio["cash"] >= position_size:
                        # 実際の現在価格を取得
                        try:
                            current_data = data_provider.get_stock_data(symbol, "1d")
                            if not current_data.empty:
                                current_price = float(current_data['Close'].iloc[-1])
                            else:
                                # データが取得できない場合のみ予測価格を使用
                                current_price = prediction * 0.98
                                print(f"  ⚠️  {symbol} の現在価格が取得できないため、予測価格を使用")
                        except Exception as e:
                            # エラーの場合は予測価格を使用
                            current_price = prediction * 0.98
                            print(f"  ⚠️  {symbol} の価格取得エラー: {str(e)} - 予測価格を使用")

                        shares = int(position_size / current_price)

                        portfolio["positions"][symbol] = {
                            "shares": shares,
                            "buy_price": current_price,
                            "current_value": shares * current_price
                        }
                        portfolio["cash"] -= shares * current_price

                        trade_record = {
                            "symbol": symbol,
                            "action": "BUY",
                            "shares": shares,
                            "price": current_price,
                            "amount": shares * current_price,
                            "confidence": confidence,
                            "accuracy": accuracy,
                            "timestamp": datetime.now()
                        }
                        portfolio["trades"].append(trade_record)

                        print(f"  🔥 取引実行: {shares}株 買い注文 (単価: {current_price:.0f}円)")
                    else:
                        print(f"  ⏸️ 資金不足のため取引見送り")
                else:
                    print(f"  ⏸️ 基準未達のため取引見送り")

            except Exception as e:
                print(f"  ❌ エラー: {symbol} の分析に失敗 - {str(e)}")

        # 結果表示
        print("\n" + "=" * 60)
        print("📈 デモ取引結果サマリー")
        print("=" * 60)

        total_investment = sum(pos["current_value"] for pos in portfolio["positions"].values())
        remaining_cash = portfolio["cash"]
        total_portfolio_value = total_investment + remaining_cash

        print(f"💰 残り現金: {remaining_cash:,.0f}円")
        print(f"📊 投資金額: {total_investment:,.0f}円")
        print(f"💼 ポートフォリオ総額: {total_portfolio_value:,.0f}円")
        print(f"📈 投資比率: {(total_investment/initial_money)*100:.1f}%")

        print(f"\n🎯 実行した取引:")
        if portfolio["trades"]:
            for i, trade in enumerate(portfolio["trades"], 1):
                print(f"  {i}. {trade['symbol']}: {trade['shares']}株 @ {trade['price']:.0f}円")
                print(f"     信頼度: {trade['confidence']:.1%}, 精度: {trade['accuracy']:.1f}%")
        else:
            print("  取引実行なし（基準を満たす予測がありませんでした）")

        print(f"\n📊 ポジション:")
        if portfolio["positions"]:
            for symbol, pos in portfolio["positions"].items():
                print(f"  {symbol}: {pos['shares']}株 (買値: {pos['buy_price']:.0f}円)")
        else:
            print("  ポジションなし")

        print("\n" + "=" * 60)
        print("ℹ️  これは1日分のデモシミュレーションです")
        print("ℹ️  実際の1週間デモ運用では毎日このような分析・取引を行います")
        print("ℹ️  利益・損失は実際の株価変動に基づいて計算されます")
        print("=" * 60)

        return portfolio

    except Exception as e:
        print(f"❌ システムエラー: {str(e)}")
        print("📝 まず以下を確認してください:")
        print("  1. 必要なライブラリがインストールされているか")
        print("  2. models_new/モジュールが正しく配置されているか")
        print("  3. インターネット接続でデータが取得できるか")
        return None

def show_help():
    """使い方の説明"""
    print("=" * 60)
    print("📖 ClStock デモ運用システム 使い方")
    print("=" * 60)
    print()
    print("🚀 簡単スタート:")
    print("  python demo_start.py")
    print()
    print("💡 このスクリプトは:")
    print("  1. 100万円の仮想資金でスタート")
    print("  2. 主要日本株5銘柄を分析")
    print("  3. 87%精度システムで予測")
    print("  4. 高精度・高信頼度の場合のみ取引実行")
    print("  5. 結果をわかりやすく表示")
    print()
    print("📊 分析される銘柄:")
    print("  7203: トヨタ自動車")
    print("  6758: ソニーグループ")
    print("  8306: 三菱UFJフィナンシャル・グループ")
    print("  6861: キーエンス")
    print("  9984: ソフトバンクグループ")
    print()
    print("🎯 取引基準:")
    print("  - 87%精度達成")
    print("  - 信頼度70%以上")
    print("  - 1銘柄あたり最大10万円")
    print()
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["help", "-h", "--help"]:
        show_help()
    else:
        portfolio = start_demo_trading()

        if portfolio:
            print("\n🎉 デモ運用シミュレーション完了！")
            print("💡 実際の1週間運用を開始したい場合は、")
            print("   trading/demo_trader.py を使用してください。")
        else:
            print("\n🔧 システムの設定を確認してから再実行してください。")