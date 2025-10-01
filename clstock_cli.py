#!/usr/bin/env python3
"""
ClStock 統合CLI
全機能へのエントリーポイント
"""

import sys
import os
import time
import click
from pathlib import Path
from typing import Optional

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from systems.process_manager import get_process_manager, ProcessStatus
from utils.logger_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="詳細ログ出力")
def cli(verbose):
    """ClStock 統合管理CLI"""
    if verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("詳細モード有効")


@cli.group()
def service():
    """サービス管理コマンド"""
    pass


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
    else:
        # 利用可能なサービス表示
        click.echo("利用可能なサービス:")
        for service_info in manager.list_services():
            status_emoji = (
                "🟢" if service_info.status == ProcessStatus.RUNNING else "🔴"
            )
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
    else:
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
            f"[監視] 監視: {'有効' if system_status['monitoring_active'] else '無効'}"
        )
        click.echo(
            f"[時刻] 時刻: {system_status['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
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
                f"  {status_emoji} {service_info.name:<20} {service_info.status.value}"
            )

            if service_info.pid:
                click.echo(f"      PID: {service_info.pid}")
            if service_info.start_time:
                uptime = (
                    system_status["timestamp"] - service_info.start_time
                ).total_seconds()
                click.echo(f"      稼働時間: {uptime/60:.1f}分")
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
    pass


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

    click.echo("🎯 デモ取引開始...")
    if manager.start_service("demo_trading"):
        return click.echo("[成功] デモ取引開始完了")

    message = "[失敗] デモ取引開始失敗"
    logger.error(message)
    raise click.ClickException(message)


@system.command()
@click.option("--symbol", "-s", default="7203", help="銘柄コード (デフォルト: 7203)")
def predict(symbol: str):
    """予測システムの実行"""
    # 入力バリデーション
    if not symbol or not isinstance(symbol, str):
        message = "[失敗] 無効な銘柄コード"
        logger.error(message)
        raise click.BadParameter(message, param_hint="symbol")

    # 銘柄コードの形式チェック（数値のみ）
    if not symbol.isdigit():
        message = "[失敗] 銘柄コードは数値のみ有効です"
        logger.error(message)
        raise click.BadParameter(message, param_hint="symbol")

    click.echo(f"🔮 予測システム実行: {symbol}")

    try:
        # 直接予測システムを実行
        from models_new.precision.precision_87_system import (
            Precision87BreakthroughSystem,
        )

        system = Precision87BreakthroughSystem()
        result = system.predict_with_87_precision(symbol)

        click.echo(f"💡 予測結果:")
        click.echo(f"  価格予測: {result['final_prediction']:.1f}")
        click.echo(f"  信頼度: {result['final_confidence']:.1%}")
        click.echo(f"  推定精度: {result['final_accuracy']:.1f}%")
        click.echo(
            f"  87%達成: {'[成功] YES' if result['precision_87_achieved'] else '[失敗] NO'}"
        )

    except Exception as e:
        message = f"[失敗] 予測実行エラー: {e}"
        logger.exception(message)
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

    click.echo("🔬 統合テストサービス起動...")
    if manager.start_service("integration_test"):
        return click.echo("[成功] 統合テストサービス起動完了")

    message = "[失敗] 統合テストサービス起動失敗"
    logger.error(message)
    raise click.ClickException(message)


@cli.group()
def data():
    """データ管理コマンド"""
    pass


@data.command()
@click.option("--symbol", "-s", multiple=True, help="銘柄コード（複数指定可能）")
@click.option("--period", "-p", default="1d", help="期間 (1d, 5d, 1mo, 3mo, 6mo, 1y)")
def fetch(symbol, period):
    """株価データの取得"""
    # 入力バリデーション
    valid_periods = [
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    ]
    if period not in valid_periods:
        message = (
            f"[失敗] 無効な期間: {period}. 有効な期間: {', '.join(valid_periods)}"
        )
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
        from data.stock_data import StockDataProvider

        provider = StockDataProvider()

        for sym in symbol:
            click.echo(f"  取得中: {sym}")
            data = provider.get_stock_data(sym, period)

            if not data.empty:
                latest_price = data["Close"].iloc[-1]
                click.echo(f"    最新価格: {latest_price:.1f}円")
            else:
                click.echo(f"    [失敗] データ取得失敗")

        click.echo("[成功] データ取得完了")

    except Exception as e:
        message = f"[失敗] データ取得エラー: {e}"
        logger.exception(message)
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

    # 依存関係チェック
    click.echo("📦 依存関係チェック...")
    try:
        import pandas
        import numpy
        import yfinance


        click.echo("[成功] 必要なライブラリがインストール済み")
    except ImportError as e:
        message = f"[失敗] 不足ライブラリ: {e}"
        logger.error(message)
        raise click.ClickException(
            f"{message}\npip install -r requirements.txt を実行してください"
        )

    click.echo("[成功] セットアップ完了")


@cli.command()
def version():
    """バージョン情報"""
    click.echo("ClStock v1.0.0")
    click.echo("高精度株価予測システム")


if __name__ == "__main__":
    cli()
