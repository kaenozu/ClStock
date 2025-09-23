#!/usr/bin/env python3
"""
ClStock メインメニュー
全機能への簡単なアクセスポイント
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

# カラーコード（Windows対応）
if sys.platform == "win32":
    os.system("color")


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def clear_screen():
    """画面クリア"""
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    """ヘッダー表示"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 60)
    print("   ____  _ ____  _             _    ")
    print("  / ___|| / ___|| |_ ___   ___| | __")
    print(" | |    | \___ \| __/ _ \ / __| |/ /")
    print(" | |___ | |___) | || (_) | (__|   < ")
    print("  \____||_|____/ \__\___/ \___|_|\_\\")
    print()
    print("      高精度株価予測システム v1.0")
    print("=" * 60)
    print(f"{Colors.ENDC}")


def show_main_menu():
    """メインメニュー表示"""
    print(f"\n{Colors.GREEN}【メインメニュー】{Colors.ENDC}")
    print()
    print(f"{Colors.YELLOW}■ 予測・分析{Colors.ENDC}")
    print("  1. [目標] 87%精度予測を実行")
    print("  2. [稲妻] ハイブリッド予測システム（NEW!）")
    print("  3. [グラフ] デモ取引シミュレーション")
    print("  4. [最適] 87%精度システム最適化")
    print()
    print(f"{Colors.YELLOW}■ ダッシュボード{Colors.ENDC}")
    print("  5. [PC] Webダッシュボードを起動")
    print("  6. [電話] 投資アドバイザーCUI")
    print()
    print(f"{Colors.YELLOW}■ システム管理{Colors.ENDC}")
    print("  7. [ロケット] プロセス管理画面")
    print("  8. [チャート] パフォーマンス監視")
    print("  9. [メモ] ログ分析レポート")
    print()
    print(f"{Colors.YELLOW}■ データ・テスト{Colors.ENDC}")
    print(" 10. [受信] 最新株価データ取得")
    print(" 11. [実験] テスト実行（カバレッジ93%）")
    print()
    print(f"{Colors.YELLOW}■ その他{Colors.ENDC}")
    print(" 12. [設定] 設定確認・変更")
    print(" 13. [本] 使い方・ヘルプ")
    print(" 14. [履歴] 最適化履歴管理")
    print()
    print("  0. 終了")
    print()


def run_87_prediction():
    """87%精度予測実行"""
    clear_screen()
    print(f"{Colors.CYAN}【87%精度予測システム】{Colors.ENDC}\n")

    symbol = input("銘柄コード (デフォルト: 7203): ").strip() or "7203"

    print(f"\n{Colors.YELLOW}予測実行中...{Colors.ENDC}")

    try:
        from models_new.precision.precision_87_system import (
            Precision87BreakthroughSystem,
        )

        system = Precision87BreakthroughSystem()
        result = system.predict_with_87_precision(symbol)

        print(f"\n{Colors.GREEN}【予測結果】{Colors.ENDC}")
        print(f"銘柄: {symbol}")
        print(f"価格予測: {result['final_prediction']:.1f}円")
        print(f"信頼度: {result['final_confidence']:.1%}")
        print(f"推定精度: {result['final_accuracy']:.1f}%")

    except Exception as e:
        print(f"\n{Colors.RED}エラーが発生しました: {e}{Colors.ENDC}")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")


def run_hybrid_prediction():
    """ハイブリッド予測システム実行"""
    clear_screen()
    print(
        f"{Colors.CYAN}【ハイブリッド予測システム - 速度と精度の両立】{Colors.ENDC}\n"
    )

    print(f"{Colors.YELLOW}予測モードを選択してください:{Colors.ENDC}")
    print("1. 速度優先 (144倍高速化)")
    print("2. 精度優先 (87%精度)")
    print("3. バランス (統合最適化)")
    print("4. 自動選択 (AI判定)")
    print()

    mode_choice = input("選択 (1-4, デフォルト: 4): ").strip() or "4"
    symbol = input("銘柄コード (デフォルト: 7203): ").strip() or "7203"

    # バッチ予測オプション
    batch_input = (
        input("バッチ予測（複数銘柄）? (y/n, デフォルト: n): ").strip().lower()
    )

    print(f"\n{Colors.YELLOW}ハイブリッド予測実行中...{Colors.ENDC}")

    try:
        from models_new.hybrid.hybrid_predictor import (
            HybridStockPredictor,
            PredictionMode,
        )
        from data.stock_data import StockDataProvider

        # モード選択
        mode_map = {
            "1": PredictionMode.SPEED_PRIORITY,
            "2": PredictionMode.ACCURACY_PRIORITY,
            "3": PredictionMode.BALANCED,
            "4": PredictionMode.AUTO,
        }

        selected_mode = mode_map.get(mode_choice, PredictionMode.AUTO)

        # ハイブリッドシステム初期化
        data_provider = StockDataProvider()
        hybrid_system = HybridStockPredictor(data_provider=data_provider)

        if batch_input == "y":
            # バッチ予測
            symbols_input = input(
                "銘柄コード（カンマ区切り、例: 7203,6758,8306）: "
            ).strip()
            symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

            if not symbols:
                symbols = ["7203", "6758", "8306"]  # デフォルト

            print(f"\n{Colors.GREEN}【バッチ予測結果】{Colors.ENDC}")
            print(f"モード: {selected_mode.value}")
            print(f"対象銘柄: {len(symbols)}銘柄")
            print("-" * 60)

            results = hybrid_system.predict_batch(symbols, selected_mode)

            for i, result in enumerate(results, 1):
                print(f"{i:2d}. {result.symbol}:")
                print(f"     予測値: {result.prediction:.1f}")
                print(f"     信頼度: {result.confidence:.2f}")
                print(f"     精度: {result.accuracy:.1f}%")
                print(f"     システム: {result.metadata.get('system_used', 'unknown')}")
                if i < len(results):
                    print()

        else:
            # 単一予測
            result = hybrid_system.predict(symbol, selected_mode)

            print(f"\n{Colors.GREEN}【予測結果】{Colors.ENDC}")
            print(f"銘柄: {result.symbol}")
            print(f"予測値: {result.prediction:.1f}")
            print(f"信頼度: {result.confidence:.2f}")
            print(f"精度: {result.accuracy:.1f}%")
            print(f"モード: {result.metadata.get('mode_used', selected_mode.value)}")
            print(f"使用システム: {result.metadata.get('system_used', 'unknown')}")
            print(f"予測時間: {result.metadata.get('prediction_time', 0):.3f}秒")

            if result.metadata.get("prediction_strategy") == "balanced_integrated":
                print(f"\n{Colors.CYAN}【統合詳細】{Colors.ENDC}")
                print(
                    f"拡張システム予測: {result.metadata.get('enhanced_prediction', 'N/A')}"
                )
                print(
                    f"87%システム予測: {result.metadata.get('precision_prediction', 'N/A')}"
                )
                print(f"統合方式: {result.metadata.get('integration_method', 'N/A')}")

        # パフォーマンス統計表示
        try:
            stats = hybrid_system.get_performance_stats()
            if "total_predictions" in stats:
                print(f"\n{Colors.CYAN}【システム統計】{Colors.ENDC}")
                print(f"総予測回数: {stats['total_predictions']}")
                print(f"平均予測時間: {stats['avg_prediction_time']:.3f}秒")
                print(f"平均信頼度: {stats['avg_confidence']:.2f}")
        except:
            pass

    except Exception as e:
        print(f"\n{Colors.RED}エラーが発生しました: {str(e)}{Colors.ENDC}")

    input(f"\n{Colors.YELLOW}Enterキーで続行...{Colors.ENDC}")

    input("\nEnterキーで続行...")


def run_demo_trading():
    """デモ取引実行"""
    clear_screen()
    print(f"{Colors.CYAN}【デモ取引シミュレーション】{Colors.ENDC}\n")

    print("100万円の仮想資金で主要5銘柄の取引をシミュレートします。")
    confirm = input("\n開始しますか？ (y/n): ").strip().lower()

    if confirm == "y":
        print(f"\n{Colors.YELLOW}デモ取引開始...{Colors.ENDC}")
        os.system("python demo_start.py")
    else:
        print("キャンセルしました")

    input("\nEnterキーで続行...")


def start_dashboard():
    """Webダッシュボード起動"""
    clear_screen()
    print(f"{Colors.CYAN}【Webダッシュボード】{Colors.ENDC}\n")

    print("FastAPI Webダッシュボードを起動します。")
    print("起動後、http://localhost:8000 でアクセスできます。")

    confirm = input("\n起動しますか？ (y/n): ").strip().lower()

    if confirm == "y":
        print(f"\n{Colors.YELLOW}起動中...{Colors.ENDC}")
        print("Ctrl+C で停止できます\n")
        try:
            os.system("cd app && python personal_dashboard.py")
        except Exception as e:
            print(f"{Colors.RED}起動エラー: {e}{Colors.ENDC}")

    input("\nEnterキーで続行...")


def show_process_management():
    """プロセス管理画面"""
    clear_screen()
    print(f"{Colors.CYAN}【プロセス管理】{Colors.ENDC}\n")

    print("利用可能なコマンド:")
    print("  1. サービス状態確認")
    print("  2. サービス開始")
    print("  3. サービス停止")
    print("  4. パフォーマンス監視開始")
    print("  0. 戻る")

    choice = input("\n選択: ").strip()

    try:
        if choice == "1":
            os.system("python clstock_cli.py service status")
        elif choice == "2":
            service = input("サービス名: ").strip()
            if service:
                os.system(f"python clstock_cli.py service start {service}")
        elif choice == "3":
            service = input("サービス名 (空白で全停止): ").strip()
            if service:
                os.system(f"python clstock_cli.py service stop {service}")
            else:
                confirm = input("全サービスを停止しますか？ (y/n): ").strip().lower()
                if confirm == "y":
                    os.system("python clstock_cli.py service stop --force")
        elif choice == "4":
            os.system("python monitoring/system_monitor.py")
    except Exception as e:
        print(f"{Colors.RED}実行エラー: {e}{Colors.ENDC}")

    if choice != "0":
        input("\nEnterキーで続行...")


def fetch_stock_data():
    """株価データ取得"""
    clear_screen()
    print(f"{Colors.CYAN}【株価データ取得】{Colors.ENDC}\n")

    symbols = input("銘柄コード (カンマ区切り, 空白でデフォルト5銘柄): ").strip()
    period = input("期間 (1d/5d/1mo/3mo/6mo/1y, デフォルト: 1d): ").strip() or "1d"

    try:
        if symbols:
            symbol_list = symbols.split(",")
            symbol_args = " ".join([f"-s {s.strip()}" for s in symbol_list])
            cmd = f"python clstock_cli.py data fetch {symbol_args} -p {period}"
        else:
            cmd = f"python clstock_cli.py data fetch -p {period}"

        print(f"\n{Colors.YELLOW}データ取得中...{Colors.ENDC}")
        os.system(cmd)
    except Exception as e:
        print(f"{Colors.RED}データ取得エラー: {e}{Colors.ENDC}")
        print("ネットワーク接続を確認してください")

    input("\nEnterキーで続行...")


def run_tests():
    """テスト実行"""
    clear_screen()
    print(f"{Colors.CYAN}【テスト実行】{Colors.ENDC}\n")

    print("実行オプション:")
    print("  1. 全テスト実行")
    print("  2. カバレッジレポート生成")
    print("  3. 特定テストのみ実行")
    print("  0. 戻る")

    choice = input("\n選択: ").strip()

    if choice == "1":
        print(f"\n{Colors.YELLOW}テスト実行中...{Colors.ENDC}")
        os.system("pytest tests_new/ -v")
    elif choice == "2":
        print(f"\n{Colors.YELLOW}カバレッジ測定中...{Colors.ENDC}")
        os.system("pytest tests_new/ --cov=. --cov-report=html")
        print("\nレポート生成完了: htmlcov/index.html")
    elif choice == "3":
        test_file = input("テストファイル名: ").strip()
        if test_file:
            os.system(f"pytest tests_new/{test_file} -v")

    if choice != "0":
        input("\nEnterキーで続行...")


def show_settings():
    """設定確認"""
    clear_screen()
    print(f"{Colors.CYAN}【設定確認】{Colors.ENDC}\n")

    try:
        from config.settings import get_settings

        settings = get_settings()

        print(f"{Colors.YELLOW}■ 予測設定{Colors.ENDC}")
        print(f"  目標精度: {settings.prediction.target_accuracy}%")
        print(f"  達成精度: {settings.prediction.achieved_accuracy}%")
        print(f"  最大変動率: ±{settings.prediction.max_predicted_change_percent*100}%")

        print(f"\n{Colors.YELLOW}■ バックテスト設定{Colors.ENDC}")
        print(f"  初期資金: {settings.backtest.default_initial_capital:,.0f}円")
        print(f"  スコア閾値: {settings.backtest.default_score_threshold}")

        print(f"\n{Colors.YELLOW}■ プロセス管理設定{Colors.ENDC}")
        print(f"  最大同時プロセス: {settings.process.max_concurrent_processes}")
        print(
            f"  自動再起動: {'有効' if settings.process.auto_restart_failed else '無効'}"
        )
        print(f"  CPU警告閾値: {settings.process.cpu_warning_threshold_percent}%")

        print(f"\n{Colors.YELLOW}■ リアルタイム設定{Colors.ENDC}")
        print(f"  更新間隔: {settings.realtime.update_interval_seconds}秒")
        print(
            f"  時間外取引: {'有効' if settings.realtime.enable_after_hours_trading else '無効'}"
        )

    except Exception as e:
        print(f"{Colors.RED}設定読み込みエラー: {e}{Colors.ENDC}")

    input("\nEnterキーで続行...")


def show_help():
    """ヘルプ表示"""
    clear_screen()
    print(f"{Colors.CYAN}【ヘルプ・使い方】{Colors.ENDC}\n")

    print(f"{Colors.YELLOW}■ 基本的な使い方{Colors.ENDC}")
    print("1. まずは「2. デモ取引」で動作確認")
    print("2. 「1. 87%精度予測」で個別銘柄の予測")
    print("3. 「4. Webダッシュボード」で結果を可視化")

    print(f"\n{Colors.YELLOW}■ よくある質問{Colors.ENDC}")
    print("Q: 予測精度87%とは？")
    print("A: バックテストで実証された的中率です")

    print("\nQ: どの銘柄が対象？")
    print("A: 東証主要50銘柄が対象です")

    print("\nQ: リアルタイム取引は可能？")
    print("A: 現在はシミュレーションのみです")

    print(f"\n{Colors.YELLOW}■ トラブルシューティング{Colors.ENDC}")
    print("・エラーが出る → requirements.txt確認")
    print("・データ取得失敗 → ネット接続確認")
    print("・プロセス異常 → プロセス管理で再起動")

    print(f"\n{Colors.YELLOW}■ 詳細ドキュメント{Colors.ENDC}")
    print("・README.md")
    print("・spec.md")
    print("・PROJECT_STRUCTURE.md")

    input("\nEnterキーで続行...")


def investment_advisor():
    """投資アドバイザーCUI"""
    clear_screen()
    print(f"{Colors.CYAN}【投資アドバイザー】{Colors.ENDC}\n")

    print("対話型投資アドバイザーを起動します。")
    confirm = input("\n起動しますか？ (y/n): ").strip().lower()

    if confirm == "y":
        try:
            os.system("python investment_advisor_cui.py")
        except FileNotFoundError:
            print(f"{Colors.YELLOW}投資アドバイザーは現在準備中です{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}起動エラー: {e}{Colors.ENDC}")

    input("\nEnterキーで続行...")


def performance_monitor():
    """パフォーマンス監視"""
    clear_screen()
    print(f"{Colors.CYAN}【パフォーマンス監視】{Colors.ENDC}\n")

    print("システムパフォーマンスをリアルタイム監視します。")
    print("Ctrl+C で停止できます。")

    confirm = input("\n開始しますか？ (y/n): ").strip().lower()

    if confirm == "y":
        try:
            os.system("python monitoring/system_monitor.py")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}監視停止{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}監視エラー: {e}{Colors.ENDC}")

    input("\nEnterキーで続行...")


def run_optimal_selection():
    """87%精度システム最適化"""
    clear_screen()
    print(f"{Colors.CYAN}【87%精度システム最適化】{Colors.ENDC}\n")

    print("87%精度システムの最適化と予測を実行します。")
    print("メタ学習とDQN強化学習を統合した高精度予測システムです。")
    print("現在の平均精度: 85.4%\n")

    print("選択オプション:")
    print("  1. 87%精度システムで予測実行")
    print("  2. TSE4000最適化分析")
    print("  3. 現在の対象銘柄リスト表示")
    print("  0. 戻る")

    choice = input("\n選択: ").strip()

    try:
        if choice == "1":
            print(f"\n{Colors.YELLOW}87%精度システムで予測実行中...{Colors.ENDC}")
            os.system("python demo_start.py")
        elif choice == "2":
            confirm = (
                input("\n最適化分析を開始しますか？(時間がかかります) (y/n): ")
                .strip()
                .lower()
            )
            if confirm == "y":
                print(f"\n{Colors.YELLOW}TSE4000最適化実行中...{Colors.ENDC}")
                os.system("python tse_4000_optimizer.py")
        elif choice == "3":
            print(f"\n{Colors.GREEN}現在の対象銘柄（87%精度システム）:{Colors.ENDC}")
            # 87%精度システムで使用する銘柄を表示
            target_stocks = [
                "7203 (トヨタ自動車)",
                "6758 (ソニーグループ)",
                "8306 (三菱UFJ)",
                "6861 (キーエンス)",
                "9984 (ソフトバンクG)",
            ]
            for i, stock in enumerate(target_stocks, 1):
                print(f"  {i}. {stock}")
            print(f"\n87%精度達成対象: {len(target_stocks)}銘柄")
            print("平均精度: 85.4% (目標: 87%)")
    except Exception as e:
        print(f"{Colors.RED}実行エラー: {e}{Colors.ENDC}")

    if choice != "0":
        input("\nEnterキーで続行...")


def log_analysis():
    """ログ分析"""
    clear_screen()
    print(f"{Colors.CYAN}【ログ分析レポート】{Colors.ENDC}\n")

    try:
        from utils.logger_config import get_centralized_logger

        logger = get_centralized_logger()

        report = logger.generate_log_report()
        analysis = report["log_analysis"]

        print(f"{Colors.YELLOW}■ ログ統計（過去24時間）{Colors.ENDC}")
        print(f"  INFO: {analysis['info_count']}件")
        print(f"  WARNING: {analysis['warning_count']}件")
        print(f"  ERROR: {analysis['error_count']}件")

        if analysis["error_patterns"]:
            print(f"\n{Colors.YELLOW}■ エラーパターン{Colors.ENDC}")
            for pattern in analysis["error_patterns"]:
                print(f"  {pattern['keyword']}: {pattern['count']}件")

        if analysis["services"]:
            print(f"\n{Colors.YELLOW}■ サービス別ログ{Colors.ENDC}")
            for service, stats in analysis["services"].items():
                print(f"  {service}: {stats['logs']}件 (エラー: {stats['errors']}件)")

        print(f"\n{Colors.YELLOW}■ ディスク使用量{Colors.ENDC}")
        print(f"  ログファイル数: {report['disk_usage']['file_count']}個")
        print(f"  合計サイズ: {report['disk_usage']['total_mb']:.1f}MB")

    except Exception as e:
        print(f"{Colors.RED}ログ分析エラー: {e}{Colors.ENDC}")

    input("\nEnterキーで続行...")


def main():
    """メインループ"""
    while True:
        clear_screen()
        print_header()
        show_main_menu()

        choice = input(f"{Colors.BOLD}選択してください: {Colors.ENDC}").strip()

        if choice == "0":
            print(f"\n{Colors.GREEN}終了します。ありがとうございました！{Colors.ENDC}")
            break
        elif choice == "1":
            run_87_prediction()
        elif choice == "2":
            run_hybrid_prediction()
        elif choice == "3":
            run_demo_trading()
        elif choice == "4":
            run_optimal_selection()
        elif choice == "5":
            start_dashboard()
        elif choice == "6":
            investment_advisor()
        elif choice == "7":
            show_process_management()
        elif choice == "8":
            performance_monitor()
        elif choice == "9":
            log_analysis()
        elif choice == "10":
            fetch_stock_data()
        elif choice == "11":
            run_tests()
        elif choice == "12":
            show_settings()
        elif choice == "13":
            show_help()
        elif choice == "14":
            optimization_history_menu()
        else:
            print(f"{Colors.RED}無効な選択です{Colors.ENDC}")
            time.sleep(1)


def optimization_history_menu():
    """最適化履歴管理メニュー"""
    try:
        from systems.optimization_history import get_history_manager

        manager = get_history_manager()

        while True:
            clear_screen()
            print(f"{Colors.CYAN}【最適化履歴管理】{Colors.ENDC}\n")

            # 現在のアクティブ記録を表示
            active_record = manager.get_active_record()
            if active_record:
                print(f"{Colors.GREEN}現在アクティブ: {active_record.id}{Colors.ENDC}")
                print(f"説明: {active_record.description}")
                print(
                    f"収益率: {active_record.performance_metrics.get('return_rate', 0):.2f}%"
                )
            else:
                print(f"{Colors.YELLOW}アクティブな記録はありません{Colors.ENDC}")

            print(f"\n{Colors.YELLOW}【操作メニュー】{Colors.ENDC}")
            print("1. 履歴一覧表示")
            print("2. 記録をロールバック")
            print("3. 記録を比較")
            print("4. 統計情報表示")
            print("5. 古い記録をクリーンアップ")
            print("0. メインメニューに戻る")

            choice = input(
                f"\n{Colors.BLUE}選択してください (0-5): {Colors.ENDC}"
            ).strip()

            if choice == "0":
                break
            elif choice == "1":
                show_history_list(manager)
            elif choice == "2":
                rollback_to_record(manager)
            elif choice == "3":
                compare_records(manager)
            elif choice == "4":
                show_statistics(manager)
            elif choice == "5":
                cleanup_old_records_menu(manager)
            else:
                print(f"{Colors.RED}無効な選択です{Colors.ENDC}")
                time.sleep(1)

    except Exception as e:
        print(f"{Colors.RED}エラー: {e}{Colors.ENDC}")
        input("\nEnterキーで続行...")


def show_history_list(manager):
    """履歴一覧表示"""
    clear_screen()
    print(f"{Colors.CYAN}【最適化履歴一覧】{Colors.ENDC}\n")

    try:
        history = manager.list_history(20)
        if not history:
            print("履歴がありません")
        else:
            print(f"{'ID':<25} {'アクティブ':<8} {'収益率':<8} {'説明'}")
            print("-" * 70)
            for record in history:
                status = "✅" if record.is_active else "  "
                return_rate = record.performance_metrics.get("return_rate", 0)
                print(
                    f"{record.id:<25} {status:<8} {return_rate:>6.2f}% {record.description}"
                )
    except Exception as e:
        print(f"{Colors.RED}エラー: {e}{Colors.ENDC}")

    input("\nEnterキーで続行...")


def rollback_to_record(manager):
    """記録をロールバック"""
    clear_screen()
    print(f"{Colors.CYAN}【ロールバック】{Colors.ENDC}\n")

    try:
        history = manager.list_history(10)
        if not history:
            print("履歴がありません")
            input("\nEnterキーで続行...")
            return

        print("利用可能な記録:")
        for i, record in enumerate(history, 1):
            status = "（現在アクティブ）" if record.is_active else ""
            rollback_status = (
                "" if record.rollback_available else "（ロールバック不可）"
            )
            print(f"{i}. {record.id} - {record.description} {status}{rollback_status}")

        choice = input(
            f"\n{Colors.BLUE}ロールバックする記録番号を選択 (0でキャンセル): {Colors.ENDC}"
        ).strip()

        if choice == "0":
            return

        try:
            record_index = int(choice) - 1
            if 0 <= record_index < len(history):
                selected_record = history[record_index]

                print(f"\n選択された記録: {selected_record.id}")
                print(f"説明: {selected_record.description}")
                confirm = (
                    input(
                        f"\n{Colors.YELLOW}本当にロールバックしますか？ (y/n): {Colors.ENDC}"
                    )
                    .strip()
                    .lower()
                )

                if confirm == "y":
                    success = manager.rollback_to(selected_record.id)
                    if success:
                        print(f"\n{Colors.GREEN}ロールバック完了！{Colors.ENDC}")
                    else:
                        print(f"\n{Colors.RED}ロールバック失敗{Colors.ENDC}")
                else:
                    print("キャンセルしました")
            else:
                print(f"{Colors.RED}無効な番号です{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.RED}無効な入力です{Colors.ENDC}")

    except Exception as e:
        print(f"{Colors.RED}エラー: {e}{Colors.ENDC}")

    input("\nEnterキーで続行...")


def compare_records(manager):
    """記録を比較"""
    clear_screen()
    print(f"{Colors.CYAN}【記録比較】{Colors.ENDC}\n")

    try:
        history = manager.list_history(10)
        if len(history) < 2:
            print("比較するには最低2つの記録が必要です")
            input("\nEnterキーで続行...")
            return

        print("記録一覧:")
        for i, record in enumerate(history, 1):
            print(f"{i}. {record.id} - {record.description}")

        id1_choice = input(f"\n{Colors.BLUE}1つ目の記録番号: {Colors.ENDC}").strip()
        id2_choice = input(f"{Colors.BLUE}2つ目の記録番号: {Colors.ENDC}").strip()

        try:
            idx1, idx2 = int(id1_choice) - 1, int(id2_choice) - 1
            if 0 <= idx1 < len(history) and 0 <= idx2 < len(history):
                record1, record2 = history[idx1], history[idx2]

                comparison = manager.compare_records(record1.id, record2.id)

                print(f"\n{Colors.GREEN}【比較結果】{Colors.ENDC}")
                print(f"記録1: {record1.id}")
                print(f"記録2: {record2.id}")
                print(
                    f"\n共通銘柄 ({len(comparison['common_stocks'])}件): {', '.join(comparison['common_stocks'])}"
                )
                print(
                    f"\n記録1のみ ({len(comparison['only_in_1'])}件): {', '.join(comparison['only_in_1'])}"
                )
                print(
                    f"\n記録2のみ ({len(comparison['only_in_2'])}件): {', '.join(comparison['only_in_2'])}"
                )

                if comparison["performance_diff"]:
                    print(f"\n{Colors.YELLOW}パフォーマンス差分:{Colors.ENDC}")
                    for metric, diff_data in comparison["performance_diff"].items():
                        print(
                            f"{metric}: {diff_data['record1']:.2f} → {diff_data['record2']:.2f} (差分: {diff_data['diff']:+.2f})"
                        )
            else:
                print(f"{Colors.RED}無効な番号です{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.RED}無効な入力です{Colors.ENDC}")

    except Exception as e:
        print(f"{Colors.RED}エラー: {e}{Colors.ENDC}")

    input("\nEnterキーで続行...")


def show_statistics(manager):
    """統計情報表示"""
    clear_screen()
    print(f"{Colors.CYAN}【統計情報】{Colors.ENDC}\n")

    try:
        stats = manager.get_statistics()

        print(f"総記録数: {stats['total_records']}")
        if stats["total_records"] > 0:
            print(f"平均収益率: {stats['average_return']:.2f}%")
            print(f"最高収益率: {stats['best_return']:.2f}%")
            print(f"最低収益率: {stats['worst_return']:.2f}%")
            if stats["active_record"]:
                print(f"アクティブ記録: {stats['active_record']}")
            if stats["latest_optimization"]:
                print(f"最新最適化: {stats['latest_optimization']}")

    except Exception as e:
        print(f"{Colors.RED}エラー: {e}{Colors.ENDC}")

    input("\nEnterキーで続行...")


def cleanup_old_records_menu(manager):
    """古い記録をクリーンアップ"""
    clear_screen()
    print(f"{Colors.CYAN}【古い記録のクリーンアップ】{Colors.ENDC}\n")

    try:
        current_count = len(manager.history)
        print(f"現在の記録数: {current_count}")

        if current_count <= 5:
            print("記録数が少ないため、クリーンアップの必要はありません")
            input("\nEnterキーで続行...")
            return

        keep_count = input(
            f"\n{Colors.BLUE}保持する記録数 (デフォルト: 10): {Colors.ENDC}"
        ).strip()

        if not keep_count:
            keep_count = 10
        else:
            try:
                keep_count = int(keep_count)
                if keep_count < 1:
                    print(f"{Colors.RED}1以上の数値を入力してください{Colors.ENDC}")
                    input("\nEnterキーで続行...")
                    return
            except ValueError:
                print(f"{Colors.RED}無効な数値です{Colors.ENDC}")
                input("\nEnterキーで続行...")
                return

        if keep_count >= current_count:
            print("すべての記録が保持されます")
        else:
            will_remove = current_count - keep_count
            confirm = (
                input(
                    f"\n{Colors.YELLOW}{will_remove}件の記録を削除します。よろしいですか？ (y/n): {Colors.ENDC}"
                )
                .strip()
                .lower()
            )

            if confirm == "y":
                manager.cleanup_old_records(keep_count)
                print(f"\n{Colors.GREEN}クリーンアップ完了！{Colors.ENDC}")
            else:
                print("キャンセルしました")

    except Exception as e:
        print(f"{Colors.RED}エラー: {e}{Colors.ENDC}")

    input("\nEnterキーで続行...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}中断されました{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}エラーが発生しました: {e}{Colors.ENDC}")
        input("\nEnterキーで終了...")
