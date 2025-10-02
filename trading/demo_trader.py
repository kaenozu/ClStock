"""
ClStock デモトレーダー

87%精度システムとリアルタイムデータを使用した
仮想資金による高精度取引シミュレーションシステム
"""

import logging
import os
from pathlib import Path
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import sqlite3
import json

# 内部モジュール
from .trading_strategy import TradingStrategy, TradingSignal, SignalType
from .portfolio_manager import DemoPortfolioManager
from .risk_manager import DemoRiskManager
from .trade_recorder import TradeRecorder

# 既存システム
from data.stock_data import StockDataProvider
from models.precision.precision_87_system import Precision87BreakthroughSystem


class TradeStatus(Enum):
    """取引ステータス"""

    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


@dataclass
class DemoTrade:
    """デモ取引記録"""

    trade_id: str
    symbol: str
    signal_type: SignalType
    quantity: int
    entry_price: float
    current_price: float
    target_price: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    entry_time: datetime
    status: TradeStatus
    confidence: float
    precision_87_achieved: bool
    expected_return: float
    actual_return: float
    profit_loss: float
    trading_costs: Dict[str, float]
    reasoning: str


@dataclass
class DemoSession:
    """デモセッション情報"""

    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    initial_capital: float
    current_capital: float
    total_trades: int
    winning_trades: int
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    precision_87_count: int
    active_positions: int


class DemoTrader:
    """
    87%精度システム統合デモトレーダー

    リアルタイムデータと仮想資金による
    実際の取引と同等の精度でのシミュレーション
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        target_symbols: List[str] = None,
        precision_threshold: float = 85.0,
        confidence_threshold: float = 0.7,
        update_interval: int = 300,
    ):  # 5分間隔
        """
        Args:
            initial_capital: 初期仮想資金
            target_symbols: 対象銘柄リスト
            precision_threshold: 取引実行精度閾値
            confidence_threshold: 取引実行信頼度閾値
            update_interval: データ更新間隔（秒）
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.target_symbols = target_symbols or self._get_default_symbols()
        self.update_interval = update_interval

        # システムコンポーネント初期化
        self.trading_strategy = TradingStrategy(
            initial_capital=initial_capital,
            precision_threshold=precision_threshold,
            confidence_threshold=confidence_threshold,
        )
        self.portfolio_manager = DemoPortfolioManager(initial_capital)
        self.risk_manager = DemoRiskManager(initial_capital)
        self.trade_recorder = TradeRecorder()

        # データプロバイダー
        self.data_provider = StockDataProvider()
        self.precision_system = Precision87BreakthroughSystem()

        # デモ取引管理
        self.active_trades: Dict[str, DemoTrade] = {}
        self.completed_trades: List[DemoTrade] = []
        self.current_session: Optional[DemoSession] = None

        # 実行制御
        self.is_running = False
        self.trading_thread: Optional[threading.Thread] = None

        # 統計情報
        self.total_signals_generated = 0
        self.total_trades_executed = 0
        self.precision_87_trades = 0
        self.winning_trades = 0

        self.logger = logging.getLogger(__name__)

    def start_demo_trading(self, session_duration_days: int = 7) -> str:
        """
        デモ取引開始

        Args:
            session_duration_days: セッション期間（日数）

        Returns:
            セッションID
        """
        if self.is_running:
            raise RuntimeError("デモ取引は既に実行中です")

        # 新しいセッション開始
        session_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = DemoSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(days=session_duration_days),
            initial_capital=self.initial_capital,
            current_capital=self.current_capital,
            total_trades=0,
            winning_trades=0,
            total_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            precision_87_count=0,
            active_positions=0,
        )

        # システム初期化
        self.portfolio_manager.reset_portfolio()
        self.risk_manager.reset_risk_metrics()
        self.active_trades.clear()
        self.completed_trades.clear()

        # 取引スレッド開始
        self.is_running = True
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.start()

        self.logger.info(f"デモ取引開始: {session_id}")
        self.logger.info(f"初期資金: {self.initial_capital:,.0f}円")
        self.logger.info(f"対象銘柄数: {len(self.target_symbols)}")
        self.logger.info(f"期間: {session_duration_days}日間")

        return session_id

    def stop_demo_trading(self) -> DemoSession:
        """
        デモ取引停止

        Returns:
            最終セッション情報
        """
        if not self.is_running:
            raise RuntimeError("デモ取引は実行されていません")

        self.is_running = False

        # スレッド終了待機
        if self.trading_thread:
            self.trading_thread.join()

        # 最終セッション情報更新
        if self.current_session:
            self.current_session.end_time = datetime.now()
            self.current_session.current_capital = self.current_capital
            self.current_session.total_trades = len(self.completed_trades)
            self.current_session.winning_trades = self.winning_trades
            self.current_session.total_return = (
                (self.current_capital - self.initial_capital)
                / self.initial_capital
                * 100
            )
            self.current_session.precision_87_count = self.precision_87_trades

            # 全ポジションクローズ
            self._close_all_positions()

            # セッション保存
            self._save_session_results()

        self.logger.info("デモ取引停止")
        return self.current_session

    def _trading_loop(self):
        """メイン取引ループ"""
        while self.is_running and self.current_session:
            try:
                # セッション期間チェック
                if datetime.now() > self.current_session.end_time:
                    self.logger.info("セッション期間終了")
                    break

                # 市場時間チェック
                if not self._is_market_hours():
                    time.sleep(300)  # 5分待機
                    continue

                # 既存ポジションの監視・管理
                self._monitor_existing_positions()

                # 新しい取引機会の探索
                self._scan_trading_opportunities()

                # リスク管理チェック
                self._perform_risk_management()

                # ポートフォリオ状況更新
                self._update_portfolio_status()

                # 待機
                time.sleep(self.update_interval)

            except KeyboardInterrupt:
                self.logger.info("ユーザーによる停止")
                break
            except Exception as e:
                self.logger.error(f"取引ループエラー: {e}")
                time.sleep(60)  # エラー時は1分待機

    def _scan_trading_opportunities(self):
        """取引機会スキャン"""
        for symbol in self.target_symbols:
            try:
                # 既にポジションを持っている場合はスキップ
                if self._has_position(symbol):
                    continue

                # 87%精度システムでシグナル生成
                signal = self.trading_strategy.generate_trading_signal(
                    symbol, self.current_capital
                )

                if signal is None:
                    continue

                self.total_signals_generated += 1

                # リスク管理チェック
                if not self.risk_manager.can_open_position(
                    symbol, signal.position_size
                ):
                    self.logger.info(f"リスク管理により取引見送り: {symbol}")
                    continue

                # デモ取引実行
                trade = self._execute_demo_trade(signal)
                if trade:
                    self.active_trades[trade.trade_id] = trade
                    self.total_trades_executed += 1

                    if trade.precision_87_achieved:
                        self.precision_87_trades += 1

                    self.logger.info(
                        f"デモ取引実行: {symbol} {signal.signal_type.value} "
                        f"数量:{trade.quantity} 価格:{trade.entry_price:.2f}"
                    )

            except Exception as e:
                self.logger.error(f"取引機会スキャンエラー {symbol}: {e}")

    def _execute_demo_trade(self, signal: TradingSignal) -> Optional[DemoTrade]:
        """デモ取引実行"""
        try:
            # 現在価格取得（リアルタイムデータ）
            current_data = self.data_provider.get_stock_data(signal.symbol, period="1d")
            if current_data is None or len(current_data) == 0:
                return None

            current_price = current_data["Close"].iloc[-1]

            # スリッページ計算
            slippage_rate = np.random.normal(0, 0.0002)  # 平均0、標準偏差0.02%
            execution_price = current_price * (1 + slippage_rate)

            # 数量計算
            position_value = signal.position_size
            quantity = int(position_value / execution_price)

            if quantity <= 0:
                return None

            # 実際のポジション価値
            actual_position_value = quantity * execution_price

            # 取引コスト計算
            trading_costs = self.trading_strategy.calculate_trading_costs(
                actual_position_value, signal.signal_type
            )

            # 総コスト
            total_cost = actual_position_value + trading_costs["total_cost"]

            # 資金チェック
            if total_cost > self.current_capital:
                self.logger.warning(f"資金不足: {signal.symbol}")
                return None

            # 取引ID生成
            trade_id = f"{signal.symbol}_{signal.signal_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # デモ取引作成
            trade = DemoTrade(
                trade_id=trade_id,
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                quantity=quantity,
                entry_price=execution_price,
                current_price=execution_price,
                target_price=signal.predicted_price,
                stop_loss_price=signal.stop_loss_price,
                take_profit_price=signal.take_profit_price,
                entry_time=datetime.now(),
                status=TradeStatus.EXECUTED,
                confidence=signal.confidence,
                precision_87_achieved=signal.precision_87_achieved,
                expected_return=signal.expected_return,
                actual_return=0.0,
                profit_loss=0.0,
                trading_costs=trading_costs,
                reasoning=signal.reasoning,
            )

            # 資金更新
            self.current_capital -= total_cost

            # ポートフォリオ更新
            self.portfolio_manager.add_position(
                signal.symbol, quantity, execution_price, signal.signal_type
            )

            # 取引記録
            self.trade_recorder.record_trade(
                {
                    "trade_id": trade_id,
                    "symbol": signal.symbol,
                    "action": "OPEN",
                    "quantity": quantity,
                    "price": execution_price,
                    "timestamp": datetime.now().isoformat(),
                    "signal_data": asdict(signal),
                    "trading_costs": trading_costs,
                }
            )

            return trade

        except Exception as e:
            self.logger.error(f"デモ取引実行エラー {signal.symbol}: {e}")
            return None

    def _monitor_existing_positions(self):
        """既存ポジション監視"""
        trades_to_close = []

        for trade_id, trade in self.active_trades.items():
            try:
                # 現在価格取得
                current_data = self.data_provider.get_stock_data(
                    trade.symbol, period="1d"
                )
                if current_data is None or len(current_data) == 0:
                    continue

                current_price = current_data["Close"].iloc[-1]
                trade.current_price = current_price

                # 損益計算
                if trade.signal_type == SignalType.BUY:
                    profit_loss = (current_price - trade.entry_price) * trade.quantity
                    actual_return = (
                        current_price - trade.entry_price
                    ) / trade.entry_price
                else:  # SELL
                    profit_loss = (trade.entry_price - current_price) * trade.quantity
                    actual_return = (
                        trade.entry_price - current_price
                    ) / trade.entry_price

                trade.profit_loss = profit_loss - trade.trading_costs["total_cost"]
                trade.actual_return = actual_return

                # エグジット条件チェック
                should_close = False
                close_reason = ""

                # ストップロス
                if trade.stop_loss_price and self._check_stop_loss(
                    trade, current_price
                ):
                    should_close = True
                    close_reason = "ストップロス"

                # 利確
                elif trade.take_profit_price and self._check_take_profit(
                    trade, current_price
                ):
                    should_close = True
                    close_reason = "利確"

                # 時間ベース（7日経過）
                elif (datetime.now() - trade.entry_time).days >= 7:
                    should_close = True
                    close_reason = "期間満了"

                if should_close:
                    trades_to_close.append((trade_id, close_reason))

            except Exception as e:
                self.logger.error(f"ポジション監視エラー {trade.symbol}: {e}")

        # ポジションクローズ
        for trade_id, reason in trades_to_close:
            self._close_position(trade_id, reason)

    def _close_position(self, trade_id: str, reason: str):
        """ポジションクローズ"""
        if trade_id not in self.active_trades:
            return

        trade = self.active_trades[trade_id]

        try:
            # クローズ価格（現在価格 + スリッページ）
            slippage_rate = np.random.normal(0, 0.0002)
            close_price = trade.current_price * (1 + slippage_rate)

            # 最終損益計算
            if trade.signal_type == SignalType.BUY:
                final_profit_loss = (close_price - trade.entry_price) * trade.quantity
            else:
                final_profit_loss = (trade.entry_price - close_price) * trade.quantity

            # 取引コスト差し引き
            final_profit_loss -= trade.trading_costs["total_cost"]

            # クローズ時の追加コスト
            close_costs = self.trading_strategy.calculate_trading_costs(
                trade.quantity * close_price, trade.signal_type
            )
            final_profit_loss -= close_costs["total_cost"]

            trade.profit_loss = final_profit_loss

            # 資金に反映
            position_value = trade.quantity * close_price
            self.current_capital += position_value - close_costs["total_cost"]

            # 勝敗判定
            if final_profit_loss > 0:
                self.winning_trades += 1

            # 完了取引に移動
            self.completed_trades.append(trade)
            del self.active_trades[trade_id]

            # ポートフォリオ更新
            self.portfolio_manager.remove_position(trade.symbol)

            # 取引記録
            self.trade_recorder.record_trade(
                {
                    "trade_id": trade_id,
                    "symbol": trade.symbol,
                    "action": "CLOSE",
                    "quantity": trade.quantity,
                    "price": close_price,
                    "timestamp": datetime.now().isoformat(),
                    "profit_loss": final_profit_loss,
                    "close_reason": reason,
                    "close_costs": close_costs,
                }
            )

            self.logger.info(
                f"ポジションクローズ: {trade.symbol} "
                f"損益:{final_profit_loss:,.0f}円 理由:{reason}"
            )

        except Exception as e:
            self.logger.error(f"ポジションクローズエラー {trade_id}: {e}")

    def _check_stop_loss(self, trade: DemoTrade, current_price: float) -> bool:
        """ストップロス判定"""
        if not trade.stop_loss_price:
            return False

        if trade.signal_type == SignalType.BUY:
            return current_price <= trade.stop_loss_price
        else:
            return current_price >= trade.stop_loss_price

    def _check_take_profit(self, trade: DemoTrade, current_price: float) -> bool:
        """利確判定"""
        if not trade.take_profit_price:
            return False

        if trade.signal_type == SignalType.BUY:
            return current_price >= trade.take_profit_price
        else:
            return current_price <= trade.take_profit_price

    def _perform_risk_management(self):
        """リスク管理実行"""
        # 最大ドローダウンチェック
        current_equity = self._calculate_total_equity()
        max_equity = max(self.initial_capital, current_equity)
        drawdown = (max_equity - current_equity) / max_equity

        if drawdown > 0.2:  # 20%ドローダウンで緊急停止
            self.logger.warning(f"最大ドローダウン到達: {drawdown:.1%}")
            self._close_all_positions()

        # 1日の最大取引回数制限
        today_trades = len(
            [
                t
                for t in self.completed_trades
                if t.entry_time.date() == datetime.now().date()
            ]
        )
        if today_trades >= 10:
            self.logger.info("1日の最大取引回数に到達")
            return

    def _update_portfolio_status(self):
        """ポートフォリオ状況更新"""
        total_equity = self._calculate_total_equity()
        total_return = (
            (total_equity - self.initial_capital) / self.initial_capital * 100
        )

        if self.current_session:
            self.current_session.current_capital = self.current_capital
            self.current_session.total_return = total_return
            self.current_session.active_positions = len(self.active_trades)

    def _calculate_total_equity(self) -> float:
        """総資産計算"""
        equity = self.current_capital

        # アクティブポジションの現在価値
        for trade in self.active_trades.values():
            position_value = trade.quantity * trade.current_price
            equity += position_value

        return equity

    def _close_all_positions(self):
        """全ポジションクローズ"""
        for trade_id in list(self.active_trades.keys()):
            self._close_position(trade_id, "強制クローズ")

    def _has_position(self, symbol: str) -> bool:
        """ポジション保有確認"""
        return any(trade.symbol == symbol for trade in self.active_trades.values())

    def _is_market_hours(self) -> bool:
        """市場時間判定"""
        now = datetime.now()
        # 平日 9:00-15:00（簡略化）
        if now.weekday() >= 5:  # 土日
            return False
        return 9 <= now.hour < 15

    def _get_default_symbols(self) -> List[str]:
        """デフォルト対象銘柄取得"""
        return [
            "6758.T",
            "7203.T",
            "8306.T",
            "9984.T",
            "6861.T",  # 主要株
            "4502.T",
            "6503.T",
            "7201.T",
            "8001.T",
            "9022.T",  # 追加株
        ]

    def _save_session_results(self):
        """セッション結果保存"""
        if not self.current_session:
            return

        session_data = {
            "session": asdict(self.current_session),
            "completed_trades": [asdict(trade) for trade in self.completed_trades],
            "final_statistics": self.get_trading_statistics(),
        }

        # JSONファイルに保存
        filename = f"demo_session_{self.current_session.session_id}.json"
        output_dir_env = os.getenv("CLSTOCK_DEMO_RESULTS_DIR")
        output_dir = (
            Path(output_dir_env)
            if output_dir_env
            else Path(__file__).resolve().parents[1] / "data"
        )

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / filename
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"セッション結果保存: {filepath}")
        except Exception as e:
            self.logger.error(f"セッション保存エラー: {e}")

    def get_trading_statistics(self) -> Dict[str, Any]:
        """取引統計取得"""
        if not self.completed_trades:
            return {}

        profits = [trade.profit_loss for trade in self.completed_trades]
        returns = [trade.actual_return for trade in self.completed_trades]

        win_rate = self.winning_trades / len(self.completed_trades) * 100
        avg_profit = np.mean(profits)
        avg_return = np.mean(returns)
        total_profit = sum(profits)

        # シャープレシオ計算
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0

        return {
            "total_trades": len(self.completed_trades),
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "average_profit": avg_profit,
            "average_return": avg_return,
            "sharpe_ratio": sharpe_ratio,
            "precision_87_trades": self.precision_87_trades,
            "precision_87_rate": self.precision_87_trades
            / len(self.completed_trades)
            * 100,
            "total_signals_generated": self.total_signals_generated,
            "signal_execution_rate": self.total_trades_executed
            / self.total_signals_generated
            * 100,
        }

    def get_current_status(self) -> Dict[str, Any]:
        """現在の状況取得"""
        total_equity = self._calculate_total_equity()
        unrealized_pnl = sum(trade.profit_loss for trade in self.active_trades.values())

        return {
            "session_id": (
                self.current_session.session_id if self.current_session else None
            ),
            "is_running": self.is_running,
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "total_equity": total_equity,
            "total_return": (total_equity - self.initial_capital)
            / self.initial_capital
            * 100,
            "active_positions": len(self.active_trades),
            "completed_trades": len(self.completed_trades),
            "unrealized_pnl": unrealized_pnl,
            "precision_87_trades": self.precision_87_trades,
            "trading_statistics": self.get_trading_statistics(),
        }
