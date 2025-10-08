#!/usr/bin/env python3
"""ClStock 統合CLI
全機能へのエントリーポイント
"""

import logging
import time
from pathlib import Path
from typing import Optional

import click

from ClStock.config.settings import get_settings
from ClStock.systems.process_manager import ProcessStatus, get_process_manager
from ClStock.utils.logger_config import get_logger
from data.stock_data import StockDataProvider
from investment_advisor_cui import InvestmentAdvisorCUI

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent

logger = get_logger(__name__)
settings = get_settings()


def _raise_cli_error(message: str) -> None:
    """Log and raise a ClickException with the provided message."""
    logger.error(message)
    raise click.ClickException(message)


def _bad_parameter(message: str, param_name: Optional[str] = None) -> None:
    """Raise a BadParameter error while preserving logging."""
    logger.error(
        (
            f"Bad parameter {param_name}: {message}"
            if param_name
            else f"Bad parameter: {message}"
        ),
    )
    if param_name:
        raise click.BadParameter(message, param_hint=param_name)
    raise click.BadParameter(message)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="詳細ログ出力")
def cli(verbose):
    """ClStock 統合管理CLI"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("詳細モード有効")


@cli.group()
def service():
    """サービス管理コマンド"""


@service.command()
@click.argument("name", required=False)
def start(name: Optional[str]):
    """サービスの開始"""
    manager = get_process_manager()

    if name:
        # 指定サービスの開始
        if manager.start_service(name):
            return click.echo(f"[成功] サービス開始: {name}")

        message = f"[失敗] サービス開始失敗: {name}"
        logger.error(message)
        raise click.ClickException(message)
    # 利用可能なサービス表示
    click.echo("利用可能なサービス:")
    for service_info in manager.list_services():
        status_emoji = "🟢" if service_info.status == ProcessStatus.RUNNING else "🔴"
        click.echo(f"  {status_emoji} {service_info.name}: {service_info.command}")


@service.command()
@click.argument("name", required=False)
@click.option("--force", "-f", is_flag=True, help="強制停止")
def stop(name: Optional[str], force: bool):
    """サービスの停止"""
    manager = get_process_manager()

    if name:
        if manager.stop_service(name, force=force):
            return click.echo(f"[成功] サービス停止: {name}")

        message = f"[失敗] サービス停止失敗: {name}"
        logger.error(message)
        raise click.ClickException(message)
    # 全サービス停止確認
    if click.confirm("全サービスを停止しますか？"):
        manager.stop_all_services(force=force)
        click.echo("[成功] 全サービス停止完了")


@service.command()
@click.argument("name")
def restart(name: str):
    """サービスの再起動"""
    manager = get_process_manager()

    if manager.restart_service(name):
        return click.echo(f"[成功] サービス再起動: {name}")

    message = f"[失敗] サービス再起動失敗: {name}"
    logger.error(message)
    raise click.ClickException(message)


@service.command()
@click.option("--watch", "-w", is_flag=True, help="リアルタイム監視")
def status(watch: bool):
    """サービス状態の表示"""
    manager = get_process_manager()

    def show_status():
        system_status = manager.get_system_status()

        click.clear()
        click.echo("=" * 60)
        click.echo("[システム] ClStock システム状態")
        click.echo("=" * 60)
        click.echo(f"[統計] サービス数: {system_status['total_services']}")
        click.echo(f"[実行中] 実行中: {system_status['running']}")
        click.echo(f"[失敗] 失敗: {system_status['failed']}")
        click.echo(
            f"[監視] 監視: {'有効' if system_status['monitoring_active'] else '無効'}",
        )
        click.echo(
            f"[時刻] 時刻: {system_status['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}",
        )
        click.echo()

        click.echo("[詳細] サービス詳細:")
        for service_info in manager.list_services():
            status_emoji = {
                ProcessStatus.RUNNING: "[実行]",
                ProcessStatus.STOPPED: "[停止]",
                ProcessStatus.STARTING: "[開始中]",
                ProcessStatus.STOPPING: "[停止中]",
                ProcessStatus.FAILED: "[失敗]",
                ProcessStatus.UNKNOWN: "[不明]",
            }.get(service_info.status, "[不明]")

            click.echo(
                f"  {status_emoji} {service_info.name:<20} {service_info.status.value}",
            )

            if service_info.pid:
                click.echo(f"      PID: {service_info.pid}")
            if service_info.start_time:
                uptime = (
                    system_status["timestamp"] - service_info.start_time
                ).total_seconds()
                click.echo(f"      稼働時間: {uptime / 60:.1f}分")
            if service_info.last_error:
                click.echo(f"      エラー: {service_info.last_error}")
            if service_info.restart_count > 0:
                click.echo(f"      再起動回数: {service_info.restart_count}")

    if watch:
        try:
            while True:
                show_status()
                time.sleep(5)
        except KeyboardInterrupt:
            click.echo("\n監視終了")
    else:
        show_status()


@service.command()
def monitor():
    """監視の開始/停止"""
    manager = get_process_manager()

    if manager.monitoring_active:
        manager.stop_monitoring()
        click.echo("📴 監視停止")
    else:
        manager.start_monitoring()
        click.echo("👀 監視開始")


@cli.group()
def system():
    """システム管理コマンド"""


@system.command()
def dashboard():
    """ダッシュボードの起動"""
    manager = get_process_manager()

    click.echo("[起動] ダッシュボード起動中...")
    if manager.start_service("dashboard"):
        click.echo("[成功] ダッシュボード起動完了")
        return click.echo("📱 http://localhost:8000 でアクセスできます")

    message = "[失敗] ダッシュボード起動失敗"
    logger.error(message)
    raise click.ClickException(message)


@system.command()
def demo():
    """デモ取引の開始"""
    manager = get_process_manager()

    click.echo("[開始] デモ取引開始...")
    if manager.start_service("demo_trading"):
        return click.echo("[成功] デモ取引開始完了")

    message = "[失敗] デモ取引開始失敗"
    logger.error(message)
    raise click.ClickException(message)


@system.command()
@click.option("--symbol", "-s", default="7203", help="銘柄コード (デフォルト: 7203)")
def predict(symbol: str):
    """予測システムの実行 (CUI表示改善版)"""
    # 銘柄コードの形式チェック（数値のみ or 数値+.T）
    is_numeric = symbol.isdigit()
    is_numeric_with_t = symbol.endswith(".T") and symbol[:-2].isdigit()
    if not (is_numeric or is_numeric_with_t):
        message = "[失敗] 銘柄コードは数値のみ、または数値+.T形式で有効です"
        logger.error(message)
        raise click.BadParameter(message, param_hint="symbol")

    # 銘柄コードを正規化（数値のみの場合は.Tを付与）
    if is_numeric:
        symbol = symbol + ".T"

    click.echo(f"[予測] システム実行: {symbol}")

    try:
        advisor = InvestmentAdvisorCUI()
        click.echo("[結果] 投資診断:")
        analysis = advisor.get_comprehensive_analysis(symbol)  # 分析実行
        # analysisの内容を整形して出力 (display_recommendationsの一部を流用)
        integrated = analysis["integrated_recommendation"]
        short = analysis["short_term"]
        name = analysis["name"]

        click.echo("[提案] 投資判断:")
        click.echo(f"  銘柄: {name} ({symbol})")
        click.echo(f"  推奨: {integrated['action']}")
        click.echo(f"  タイミング: {integrated['timing']}")
        click.echo(f"  現在価格: {short['current_price']:,.0f}円")
        click.echo(f"  短期見通し: {integrated['short_term_outlook']} (1日)")
        click.echo(f"  中期見通し: {integrated['medium_term_outlook']} (1ヶ月)")
        click.echo(f"  信頼度: {integrated['confidence']:.1%}")
        click.echo(f"  リスク: {integrated['risk_level']}")
        evaluation = short.get("evaluation", {})
        sample_size = evaluation.get("sample_size", 0)
        if sample_size:
            avg_up = evaluation.get("avg_positive_return", 0.0)
            avg_down = evaluation.get("avg_negative_return", 0.0)
            click.echo(
                f"  シグナル実績: 過去{sample_size}件で命中率 {short.get('accuracy_estimate', 0.0):.1f}%"
            )
            click.echo(
                f"    平均上昇リターン: {avg_up:+.2%} / 平均下落: {avg_down:+.2%}",
            )
        if integrated["action"] in ["強い買い", "買い"]:
            click.echo(f"  目標価格: {integrated['target_price']:,.0f}円")
            click.echo(f"  損切価格: {integrated['stop_loss']:,.0f}円")
        # 必要に応じて medium_signals や reasoning も表示

    except Exception as e:
        message = f"[失敗] 予測実行エラー: {e}"
        logger.error(message)
        raise click.ClickException(message)


@system.command()
def optimize():
    """最適化システムの実行"""
    manager = get_process_manager()

    click.echo("[最適化] ウルトラ最適化システム起動...")
    if manager.start_service("optimized_system"):
        return click.echo("[成功] ウルトラ最適化システム起動完了")

    message = "[失敗] 最適化システム起動失敗"
    logger.error(message)
    raise click.ClickException(message)


@system.command()
def integration():
    """統合テストサービスの実行"""
    manager = get_process_manager()

    click.echo("[開始] 統合テストサービス起動...")
    if manager.start_service("integration_test"):
        return click.echo("[成功] 統合テストサービス起動完了")

    message = "[失敗] 統合テストサービス起動失敗"
    logger.error(message)
    raise click.ClickException(message)


@cli.group()
def data():
    """データ管理コマンド"""


@data.command()
@click.option("--symbol", "-s", multiple=True, help="銘柄コード（複数指定可能）")
@click.option("--period", "-p", default="1d", help="期間 (1d, 5d, 1mo, 3mo, 6mo, 1y)")
def fetch(symbol, period):
    """株価データの取得"""
    # 入力バリデーション
    # yfinance がサポートする期間 (https://pypi.org/project/yfinance/ 参照)
    # 1d: 1日, 5d: 5日, 1mo: 1ヶ月, 3mo: 3ヶ月, 6mo: 6ヶ月
    # 1y: 1年, 2y: 2年, 5y: 5年, 10y: 10年
    # ytd: 年初来 (Year to Date), max: 利用可能な最も長い期間
    valid_periods = [
        "1d",  # 1日
        "5d",  # 5日
        "1mo",  # 1ヶ月
        "3mo",  # 3ヶ月
        "6mo",  # 6ヶ月
        "1y",  # 1年
        "2y",  # 2年
        "5y",  # 5年
        "10y",  # 10年
        "ytd",  # 年初来 (Year to Date)
        "max",  # 利用可能な最も長い期間
    ]
    if period not in valid_periods:
        message = f"[失敗] 無効な期間: {period}. 有効な期間: {', '.join(valid_periods)}"
        logger.error(message)
        raise click.BadParameter(message, param_hint="period")

    if not symbol:
        symbol = ["7203", "6758", "8306", "6861", "9984"]  # デフォルト銘柄

    # 銘柄コードのバリデーション
    for sym in symbol:
        if not isinstance(sym, str) or not sym.isdigit():
            message = f"[失敗] 無効な銘柄コード: {sym}"
            logger.error(message)
            raise click.BadParameter(message, param_hint="symbol")

    click.echo(f"📊 データ取得: {list(symbol)} (期間: {period})")

    try:
        provider = StockDataProvider()

        for sym in symbol:
            click.echo(f"  取得中: {sym}")
            data = provider.get_stock_data(sym, period)

            if not data.empty:
                latest_price = data["Close"].iloc[-1]
                click.echo(f"    最新価格: {latest_price:.1f}円")
            else:
                click.echo("    [失敗] データ取得失敗")

        click.echo("[成功] データ取得完了")

    except Exception as e:
        message = f"[失敗] データ取得エラー: {e}"
        logger.error(message)
        raise click.ClickException(message)


@cli.command()
def setup():
    """初期セットアップ"""
    click.echo("🔧 ClStock セットアップ")

    # ディレクトリ作成
    dirs_to_create = [
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "cache",
    ]

    for dir_path in dirs_to_create:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            click.echo(f"📁 ディレクトリ作成: {dir_path}")

    # 依存関係の確認は requirements.txt で管理されているため、ここでのチェックは不要
    click.echo("📦 依存関係は requirements.txt に基づいてインストールしてください。")

    click.echo("[成功] セットアップ完了")


@cli.command()
def version():
    """バージョン情報"""
    click.echo("ClStock v1.0.0")
    click.echo("高精度株価予測システム")


if __name__ == "__main__":
    cli()
