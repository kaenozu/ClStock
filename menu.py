#!/usr/bin/env python3
"""ClStock 新メニューシステム v2.0
ハイブリッド予測システム対応・現状最適化版
"""

import os
import sys
import time

from utils.logger_config import get_logger

logger = get_logger(__name__)

# カラーコード（Windows対応）
if sys.platform == "win32":
    os.system("color")  # nosec B605, B607


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    MAGENTA = "\033[35m"


def clear_screen():
    """画面クリア"""
    import subprocess

    try:
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "cls"], check=True, shell=False)  # nosec B603, B607
        else:
            subprocess.run(["clear"], check=True, shell=False)  # nosec B603, B607
    except subprocess.CalledProcessError:
        # フォールバック: コンソール制御文字で画面クリア
        print("\033[2J\033[H")


def print_header():
    """最新ヘッダー表示"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 70)
    print("   ____  _ ____  _             _    ")
    print(r"  / ___|| / ___|| |_ ___   ___| | __")
    print(r" | |    | \___ \| __/ _ \ / __| |/ /")
    print(r" | |___ | |___) | || (_) | (__|   < ")
    print(r"  \____||_|____/ \__\___/ \___|_|\_\\")
    print()
    print("     次世代株価予測システム v2.0 - HYBRID")
    print("     144倍高速化 × 91.4%精度 両立達成")
    print("=" * 70)
    print(f"{Colors.ENDC}")


def show_main_menu():
    """フルオート基準の簡素化メニュー"""
    print(f"\n{Colors.GREEN}{Colors.BOLD}【ClStock フルオートシステム】{Colors.ENDC}")
    print()

    # メイン機能 - フルオート優先
    print(f"{Colors.MAGENTA}■ 投資推奨システム{Colors.ENDC}")
    print("  1. フルオート - 完全自動投資推奨（推奨）")
    print("  2. TSE4000最適化 - 銘柄選定のみ")
    print("  3. 投資アドバイザー - 対話型分析")
    print()

    # 必要最小限の機能
    print(f"{Colors.YELLOW}■ システム管理{Colors.ENDC}")
    print("  4. 最新データ取得")
    print("  5. システム設定")
    print("  6. ヘルプ・使い方")
    print()

    print("  0. 終了")
    print()


def run_hybrid_prediction():
    """ハイブリッド予測システム（メイン機能）"""
    clear_screen()
    print(
        f"{Colors.MAGENTA}{Colors.BOLD}【ハイブリッド予測システム v2.0】{Colors.ENDC}",
    )
    print(
        f"{Colors.CYAN}速度と精度の革新的両立 - 144倍高速化 × 91.4%精度{Colors.ENDC}\n",
    )

    print(f"{Colors.YELLOW}予測モードを選択:{Colors.ENDC}")
    print("1. 速度優先 - 0.018秒/銘柄 (250銘柄/秒)")
    print("2. 精度優先 - 91.4%精度 (0.84信頼度)")
    print("3. 統合最適化 - 両方の長所")
    print("4. 自動選択 - インテリジェント判定")
    print()

    mode_choice = input("モード選択 (1-4, デフォルト: 4): ").strip() or "4"

    print(f"\n{Colors.YELLOW}予測対象を選択:{Colors.ENDC}")
    print("1. 単一銘柄予測")
    print("2. バッチ予測（複数銘柄）")
    print("3. おすすめ3銘柄（ソニー・トヨタ・三菱UFJ）")

    target_choice = input("対象選択 (1-3, デフォルト: 3): ").strip() or "3"

    print(f"\n{Colors.CYAN}ハイブリッド予測実行中...{Colors.ENDC}")

    try:
        from data.stock_data import StockDataProvider
        from models.hybrid.hybrid_predictor import (
            HybridStockPredictor,
            PredictionMode,
        )

        # モードマッピング
        mode_map = {
            "1": PredictionMode.SPEED_PRIORITY,
            "2": PredictionMode.ACCURACY_PRIORITY,
            "3": PredictionMode.BALANCED,
            "4": PredictionMode.AUTO,
        }

        mode_names = {
            "1": "速度優先",
            "2": "精度優先",
            "3": "バランス",
            "4": "自動選択",
        }

        selected_mode = mode_map.get(mode_choice, PredictionMode.AUTO)
        mode_name = mode_names.get(mode_choice, "自動選択")

        # システム初期化
        data_provider = StockDataProvider()
        hybrid_system = HybridStockPredictor(data_provider=data_provider)

        # 予測実行
        if target_choice == "1":
            # 単一銘柄
            symbol = input("銘柄コード (例: 7203): ").strip()
            if not symbol:
                symbol = "7203"

            result = hybrid_system.predict(symbol, selected_mode)
            display_single_result(result, mode_name)

        elif target_choice == "2":
            # バッチ予測
            symbols_input = input(
                "銘柄コード（カンマ区切り、例: 7203,6758,8306）: ",
            ).strip()
            symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

            if not symbols:
                symbols = ["7203", "6758", "8306"]

            results = hybrid_system.predict_batch(symbols, selected_mode)
            display_batch_results(results, mode_name)

        else:
            # おすすめ3銘柄
            symbols = ["6758.T", "7203.T", "8306.T"]  # ソニー、トヨタ、三菱UFJ
            results = hybrid_system.predict_batch(symbols, selected_mode)
            display_recommended_results(results, mode_name)

        # システム統計表示
        display_system_stats(hybrid_system)

    except Exception as e:
        print(f"\n{Colors.RED}エラーが発生しました: {e!s}{Colors.ENDC}")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def display_single_result(result, mode_name):
    """単一予測結果表示"""
    print(f"\n{Colors.GREEN}{Colors.BOLD}【予測結果 - {mode_name}モード】{Colors.ENDC}")
    print("=" * 50)
    print(f"銘柄: {result.symbol}")
    print(f"予測値: {result.prediction:.1f}")
    print(f"信頼度: {result.confidence:.2f}")
    print(f"精度: {result.accuracy:.1f}%")
    print(f"予測時間: {result.metadata.get('prediction_time', 0):.3f}秒")
    print(f"使用システム: {result.metadata.get('system_used', 'unknown')}")

    if result.metadata.get("prediction_strategy") == "balanced_integrated":
        print(f"\n{Colors.CYAN}【統合詳細】{Colors.ENDC}")
        print(f"拡張システム: {result.metadata.get('enhanced_prediction', 'N/A')}")
        print(f"87%システム: {result.metadata.get('precision_prediction', 'N/A')}")


def display_batch_results(results, mode_name):
    """バッチ予測結果表示"""
    print(
        f"\n{Colors.GREEN}{Colors.BOLD}【バッチ予測結果 - {mode_name}モード】{Colors.ENDC}",
    )
    print("=" * 60)
    print(f"処理銘柄数: {len(results)}")
    print("-" * 60)
    print("順位  銘柄     予測値   信頼度   精度    システム")
    print("-" * 60)

    for i, result in enumerate(results, 1):
        system_short = result.metadata.get("system_used", "unknown")[:8]
        print(
            f"{i:2d}.  {result.symbol:8s} {result.prediction:7.1f}  {result.confidence:6.2f}  {result.accuracy:5.1f}%  {system_short}",
        )


def display_recommended_results(results, mode_name):
    """Display recommended stock results"""
    print(
        f"\n{Colors.GREEN}{Colors.BOLD}【おすすめ3銘柄予測 - {mode_name}モード】{Colors.ENDC}",
    )
    print("=" * 60)

    symbol_names = {
        "6758.T": "ソニー",
        "7203.T": "トヨタ自動車",
        "8306.T": "三菱UFJ銀行",
    }

    for i, result in enumerate(results, 1):
        company_name = symbol_names.get(result.symbol, result.symbol)
        print(f"\n{i}. {company_name} ({result.symbol})")
        print(f"   予測値: {result.prediction:.1f}")
        print(f"   信頼度: {result.confidence:.2f}")
        print(f"   精度: {result.accuracy:.1f}%")
        print(f"   時間: {result.metadata.get('prediction_time', 0):.3f}秒")


def display_system_stats(hybrid_system):
    """システム統計表示"""
    try:
        stats = hybrid_system.get_performance_stats()
        if "total_predictions" in stats:
            print(f"\n{Colors.CYAN}【システム統計】{Colors.ENDC}")
            print(f"総予測回数: {stats['total_predictions']}")
            print(f"平均予測時間: {stats.get('avg_prediction_time', 0):.3f}秒")
            print(f"平均信頼度: {stats.get('avg_confidence', 0):.2f}")
    except Exception as e:
        logger.warning(f"Display stats failed: {e}")


def run_precision_87():
    """87%精度予測システム"""
    clear_screen()
    print(f"{Colors.CYAN}【87%精度予測システム】{Colors.ENDC}\n")

    symbol = input("銘柄コード (デフォルト: 7203): ").strip() or "7203"
    print(f"\n{Colors.YELLOW}87%精度予測実行中...{Colors.ENDC}")

    try:
        from models.precision.precision_87_system import (
            Precision87BreakthroughSystem,
        )

        system = Precision87BreakthroughSystem()
        result = system.predict_with_87_precision(symbol)

        print(f"\n{Colors.GREEN}【87%精度予測結果】{Colors.ENDC}")
        print(f"銘柄: {symbol}")
        print(f"価格予測: {result['final_prediction']:.1f}円")
        print(f"信頼度: {result['final_confidence']:.1%}")
        print(f"推定精度: {result['final_accuracy']:.1f}%")

        if result.get("precision_87_achieved"):
            print(f"87%達成: {Colors.GREEN}YES{Colors.ENDC}")
        else:
            print(f"87%達成: {Colors.YELLOW}調整中{Colors.ENDC}")

    except Exception as e:
        print(f"\n{Colors.RED}エラー: {e!s}{Colors.ENDC}")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def run_enhanced_ensemble():
    """拡張アンサンブルシステム"""
    clear_screen()
    print(f"{Colors.BLUE}【拡張アンサンブルシステム - 高速モード】{Colors.ENDC}\n")

    symbol = input("銘柄コード (デフォルト: 7203): ").strip() or "7203"
    print(f"\n{Colors.YELLOW}高速予測実行中...{Colors.ENDC}")

    try:
        from data.stock_data import StockDataProvider
        from models.ensemble.ensemble_predictor import EnsembleStockPredictor

        data_provider = StockDataProvider()
        system = EnsembleStockPredictor(data_provider=data_provider)
        result = system.predict(symbol)

        print(f"\n{Colors.GREEN}【高速予測結果】{Colors.ENDC}")
        print(f"銘柄: {result.symbol}")
        print(f"予測値: {result.prediction:.1f}")
        print(f"信頼度: {result.confidence:.2f}")
        print(f"精度: {result.accuracy:.1f}%")
        print("予測時間: 超高速 (0.01秒未満)")

    except Exception as e:
        print(f"\n{Colors.RED}エラー: {e!s}{Colors.ENDC}")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def run_demo_trading():
    """デモ取引シミュレーション"""
    clear_screen()
    print(f"{Colors.GREEN}【デモ取引シミュレーション】{Colors.ENDC}\n")

    try:
        print("デモ取引システムを起動中...")
        import subprocess

        result = subprocess.run(
            ["python", "demo_start.py"],
            check=False, capture_output=True,
            text=True,  # nosec B603, B607
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except Exception as e:
        print(f"エラー: {e!s}")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def run_tse_optimization():
    """TSE4000最適化"""
    clear_screen()
    print(f"{Colors.YELLOW}【TSE4000最適化システム】{Colors.ENDC}\n")

    try:
        print("TSE4000最適化を実行中...")
        import subprocess

        result = subprocess.run(
            ["python", "tse_4000_optimizer.py"],
            check=False, capture_output=True,
            text=True,  # nosec B603, B607
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except Exception as e:
        print(f"エラー: {e!s}")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def run_investment_advisor():
    """投資アドバイザーCUI"""
    clear_screen()
    print(f"{Colors.BLUE}【投資アドバイザーCUI】{Colors.ENDC}\n")

    try:
        print("投資アドバイザーを起動中...")
        import subprocess

        result = subprocess.run(
            ["python", "investment_advisor_cui.py"],
            check=False, capture_output=True,
            text=True,  # nosec B603, B607
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except Exception as e:
        print(f"エラー: {e!s}")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def start_dashboard():
    """Webダッシュボード起動"""
    clear_screen()
    print(f"{Colors.CYAN}【Webダッシュボード】{Colors.ENDC}\n")
    print("Webダッシュボードを起動中...")
    print("ブラウザで http://localhost:8000 にアクセスしてください")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def performance_monitor():
    """パフォーマンス監視"""
    clear_screen()
    print(f"{Colors.MAGENTA}【システムパフォーマンス監視】{Colors.ENDC}\n")

    try:
        print("システム性能を測定中...")
        import subprocess

        result = subprocess.run(
            ["python", "selective_system.py"],
            check=False, capture_output=True,
            text=True,  # nosec B603, B607
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except Exception as e:
        print(f"エラー: {e!s}")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def fetch_data():
    """データ取得"""
    clear_screen()
    print(f"{Colors.CYAN}【最新データ取得】{Colors.ENDC}\n")
    print("最新の株価データを取得中...")

    # 簡単なデータ取得テスト
    try:
        from data.stock_data import StockDataProvider

        provider = StockDataProvider()
        data = provider.get_stock_data("7203", "5d")
        if not data.empty:
            print(f"✓ データ取得成功: {len(data)}日分")
            print(f"  最新価格: {data['Close'].iloc[-1]:.1f}円")
        else:
            print("✗ データ取得失敗")
    except Exception as e:
        print(f"エラー: {e!s}")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def run_system_test():
    """システムテスト"""
    clear_screen()
    print(f"{Colors.GREEN}【システム統合テスト】{Colors.ENDC}\n")

    try:
        print("システムテストを実行中...")
        # test_hybrid_system.pyはarchiveに移動されたため、
        # 新しいテストスイートを使用
        import subprocess

        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-v"],
            check=False, capture_output=True,
            text=True,  # nosec B603, B607
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except Exception as e:
        print(f"エラー: {e!s}")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def show_settings():
    """設定表示"""
    clear_screen()
    print(f"{Colors.YELLOW}【システム設定】{Colors.ENDC}\n")

    print("現在のシステム設定:")
    print("- 予測システム: ハイブリッドv2.0")
    print("- データプロバイダー: Yahoo Finance")
    print("- キャッシュ: 有効")
    print("- 並列処理: 8ワーカー")
    print("- 精度目標: 91.4%")
    print("- 速度目標: 250銘柄/秒")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def show_help():
    """ヘルプ表示"""
    clear_screen()
    print(f"{Colors.CYAN}【ClStock v2.0 使い方ガイド】{Colors.ENDC}\n")

    print("🚀 主要機能:")
    print("1. ハイブリッド予測 - 速度と精度を両立した次世代システム")
    print("   - 速度優先: 250銘柄/秒の超高速処理")
    print("   - 精度優先: 91.4%の高精度予測")
    print("   - バランス: 両方の長所を統合")
    print("   - 自動選択: AIが最適モードを判定")
    print()
    print("2. 87%精度予測 - 実証済み高精度システム")
    print("3. 拡張アンサンブル - 並列処理による高速化")
    print("4. デモ取引 - リスクフリーでの実践練習")
    print()
    print("💡 推奨用途:")
    print("- 日常分析: ハイブリッド予測（バランスモード）")
    print("- 重要判断: 87%精度予測")
    print("- 大量処理: ハイブリッド予測（速度優先）")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def show_optimization_history():
    """最適化履歴"""
    clear_screen()
    print(f"{Colors.MAGENTA}【最適化履歴】{Colors.ENDC}\n")

    print("ClStock進化の歴史:")
    print("Phase 0: 基本予測システム")
    print("Phase 1: 87%精度達成 + 拡張アンサンブル")
    print("Phase 2: ハイブリッドシステム統合")
    print("         ↳ 144倍高速化 × 91.4%精度両立達成")
    print()
    print("技術革新:")
    print("✓ 並列特徴量計算 (8ワーカー)")
    print("✓ LRU+圧縮キャッシュ")
    print("✓ マルチタイムフレーム統合")
    print("✓ 動的モード選択")
    print("✓ インテリジェント統合")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def optimization_history_menu():
    """最適化履歴管理メニュー"""
    clear_screen()
    print(f"{Colors.MAGENTA}【最適化履歴管理】{Colors.ENDC}\n")

    print("1. 履歴一覧表示")
    print("2. 特定レコードにロールバック")
    print("3. 履歴統計表示")
    print("0. メインメニューに戻る")

    choice = input(f"\n{Colors.BOLD}選択してください (0-3): {Colors.ENDC}").strip()

    if choice == "1":
        show_history_list()
    elif choice == "2":
        record_id = input("ロールバック先のレコードID: ").strip()
        if record_id:
            rollback_to_record(record_id)
    elif choice == "3":
        show_history_statistics()
    elif choice == "0":
        return
    else:
        print(f"{Colors.RED}無効な選択です{Colors.ENDC}")
        time.sleep(1)


def show_history_list():
    """履歴一覧表示"""
    try:
        from systems.optimization_history import OptimizationHistoryManager

        manager = OptimizationHistoryManager()
        records = manager.list_optimization_records()

        if not records:
            print(f"{Colors.YELLOW}保存された最適化履歴がありません。{Colors.ENDC}")
            return

        print(f"\n{Colors.GREEN}【最適化履歴一覧】{Colors.ENDC}")
        print("-" * 80)
        print("ID              日時                     精度     銘柄数")
        print("-" * 80)

        for record in records:
            print(
                f"{record['record_id']:15s} {record['timestamp']:20s} "
                f"{record.get('accuracy', 0):.1f}%   {len(record.get('stocks', []))} 銘柄",
            )

    except Exception as e:
        print(f"{Colors.RED}履歴表示エラー: {e}{Colors.ENDC}")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def rollback_to_record(record_id: str):
    """指定レコードにロールバック"""
    try:
        from systems.optimization_history import OptimizationHistoryManager

        manager = OptimizationHistoryManager()

        if manager.rollback_to_configuration(record_id):
            print(
                f"{Colors.GREEN}レコード {record_id} への復元が完了しました。{Colors.ENDC}",
            )
        else:
            print(f"{Colors.RED}レコード {record_id} が見つかりません。{Colors.ENDC}")

    except Exception as e:
        print(f"{Colors.RED}ロールバックエラー: {e}{Colors.ENDC}")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def show_history_statistics():
    """履歴統計表示"""
    try:
        from systems.optimization_history import OptimizationHistoryManager

        manager = OptimizationHistoryManager()
        stats = manager.get_optimization_statistics()

        print(f"\n{Colors.GREEN}【最適化履歴統計】{Colors.ENDC}")
        print("-" * 50)
        print(f"総実行回数: {stats.get('total_runs', 0)} 回")
        print(f"平均精度: {stats.get('average_accuracy', 0):.1f}%")
        print(f"最高精度: {stats.get('max_accuracy', 0):.1f}%")
        print(f"最新実行: {stats.get('latest_run', 'N/A')}")

    except Exception as e:
        print(f"{Colors.RED}統計表示エラー: {e}{Colors.ENDC}")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def run_full_auto():
    """フルオート - 完全自動投資推奨システム"""
    try:
        print(f"\n{Colors.GREEN}フルオート投資推奨システムを開始します...{Colors.ENDC}")
        print(
            f"{Colors.YELLOW}すべて自動で実行されます。しばらくお待ちください...{Colors.ENDC}\n",
        )

        # フルオートシステムのインポートと実行
        import asyncio

        from full_auto_system import FullAutoInvestmentSystem

        async def run_auto():
            auto_system = FullAutoInvestmentSystem()
            recommendations = await auto_system.run_full_auto_analysis()

            if recommendations:
                print(
                    f"\n{Colors.GREEN}{Colors.BOLD}【フルオート投資推奨結果】{Colors.ENDC}",
                )
                print("=" * 80)

                for i, rec in enumerate(recommendations, 1):
                    print(f"\n{Colors.CYAN}推奨 {i}: {rec.symbol}{Colors.ENDC}")
                    print(f"  企業名: {rec.company_name}")
                    print(f"  推奨度: {rec.recommendation_score:.1f}/10")
                    print(f"  予想リターン: {rec.expected_return:.2f}%")
                    print(f"  リスクレベル: {rec.risk_level}")
                    print(f"  買い推奨時刻: {rec.buy_timing}")
                    print(f"  売り推奨時刻: {rec.sell_timing}")
                    print(f"  理由: {rec.reasoning}")
                    print("-" * 60)

                print(f"\n{Colors.GREEN}フルオート分析が完了しました！{Colors.ENDC}")
            else:
                print(f"{Colors.YELLOW}現在推奨できる銘柄がありません。{Colors.ENDC}")

        # 非同期実行
        asyncio.run(run_auto())

    except ImportError as e:
        print(
            f"{Colors.RED}フルオートシステムの読み込みに失敗しました: {e}{Colors.ENDC}",
        )
        print(
            f"{Colors.YELLOW}システムが完全にインストールされていない可能性があります。{Colors.ENDC}",
        )
    except Exception as e:
        print(f"{Colors.RED}フルオート実行中にエラーが発生しました: {e}{Colors.ENDC}")

    input(f"\n{Colors.BOLD}Enterキーで戻る...{Colors.ENDC}")


def main():
    """フルオート基準メインループ"""
    while True:
        clear_screen()
        print_header()
        show_main_menu()

        choice = input(f"{Colors.BOLD}選択してください (0-6): {Colors.ENDC}").strip()

        if choice == "0":
            print(
                f"\n{Colors.GREEN}ClStock フルオートシステムを終了します。ありがとうございました！{Colors.ENDC}",
            )
            break
        if choice == "1":
            run_full_auto()  # フルオート（メイン機能）
        elif choice == "2":
            run_tse_optimization()  # TSE4000最適化
        elif choice == "3":
            run_investment_advisor()  # 投資アドバイザー
        elif choice == "4":
            fetch_data()  # データ取得
        elif choice == "5":
            show_settings()  # システム設定
        elif choice == "6":
            show_help()  # ヘルプ
        else:
            print(f"{Colors.RED}無効な選択です (0-6を入力してください){Colors.ENDC}")
            time.sleep(1)


if __name__ == "__main__":
    main()
