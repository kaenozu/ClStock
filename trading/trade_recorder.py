"""
ClStock 取引記録・レポートシステム

取引履歴の詳細記録とパフォーマンスレポート生成
87%精度システムの成果追跡とCSV/JSONエクスポート機能
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# 内部モジュール
from .trading_strategy import SignalType


@dataclass
class TradeRecord:
    """取引記録"""
    trade_id: str
    session_id: str
    symbol: str
    action: str  # OPEN, CLOSE, PARTIAL_CLOSE
    quantity: int
    price: float
    timestamp: datetime
    signal_type: SignalType
    confidence: float
    precision: float
    precision_87_achieved: bool
    expected_return: float
    actual_return: Optional[float]
    profit_loss: Optional[float]
    trading_costs: Dict[str, float]
    position_size: float
    market_value: float
    reasoning: str
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    execution_details: Dict[str, Any]


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    total_return_pct: float
    average_return: float
    average_win: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    precision_87_trades: int
    precision_87_success_rate: float
    best_trade: float
    worst_trade: float
    average_holding_period: float


@dataclass
class TaxCalculation:
    """税務計算"""
    total_realized_gains: float
    total_realized_losses: float
    net_capital_gains: float
    short_term_gains: float
    long_term_gains: float
    wash_sales: float
    estimated_tax_liability: float
    deductible_expenses: float


class TradeRecorder:
    """
    取引記録・レポートシステム

    全ての取引の詳細記録とパフォーマンス分析
    """

    def __init__(self, db_path: str = "C:\\gemini-desktop\\ClStock\\data\\trading_records.db"):
        """
        Args:
            db_path: データベースファイルパス
        """
        self.db_path = db_path
        self.trade_records: List[TradeRecord] = []
        self.session_id: Optional[str] = None

        # データベース初期化
        self._init_database()

        # チャート設定
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        self.logger = logging.getLogger(__name__)

    def record_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        取引記録

        Args:
            trade_data: 取引データ

        Returns:
            記録成功フラグ
        """
        try:
            # TradeRecord作成
            trade_record = TradeRecord(
                trade_id=trade_data.get('trade_id', ''),
                session_id=trade_data.get('session_id', self.session_id or ''),
                symbol=trade_data.get('symbol', ''),
                action=trade_data.get('action', ''),
                quantity=trade_data.get('quantity', 0),
                price=trade_data.get('price', 0.0),
                timestamp=datetime.fromisoformat(trade_data.get('timestamp', datetime.now().isoformat())),
                signal_type=SignalType(trade_data.get('signal_type', SignalType.HOLD.value)),
                confidence=trade_data.get('confidence', 0.0),
                precision=trade_data.get('precision', 0.0),
                precision_87_achieved=trade_data.get('precision_87_achieved', False),
                expected_return=trade_data.get('expected_return', 0.0),
                actual_return=trade_data.get('actual_return'),
                profit_loss=trade_data.get('profit_loss'),
                trading_costs=trade_data.get('trading_costs', {}),
                position_size=trade_data.get('position_size', 0.0),
                market_value=trade_data.get('market_value', 0.0),
                reasoning=trade_data.get('reasoning', ''),
                stop_loss_price=trade_data.get('stop_loss_price'),
                take_profit_price=trade_data.get('take_profit_price'),
                execution_details=trade_data.get('execution_details', {})
            )

            # メモリ内記録
            self.trade_records.append(trade_record)

            # データベース保存
            self._save_to_database(trade_record)

            self.logger.info(f"取引記録: {trade_record.symbol} {trade_record.action}")
            return True

        except Exception as e:
            self.logger.error(f"取引記録エラー: {e}")
            return False

    def generate_performance_report(self,
                                  session_id: Optional[str] = None,
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> PerformanceMetrics:
        """
        パフォーマンスレポート生成

        Args:
            session_id: セッションID（指定時はそのセッションのみ）
            start_date: 開始日
            end_date: 終了日

        Returns:
            パフォーマンス指標
        """
        try:
            # 対象取引抽出
            trades = self._filter_trades(session_id, start_date, end_date)

            if not trades:
                return PerformanceMetrics(
                    total_trades=0, winning_trades=0, losing_trades=0,
                    win_rate=0.0, total_return=0.0, total_return_pct=0.0,
                    average_return=0.0, average_win=0.0, average_loss=0.0,
                    profit_factor=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                    max_drawdown=0.0, max_consecutive_wins=0, max_consecutive_losses=0,
                    precision_87_trades=0, precision_87_success_rate=0.0,
                    best_trade=0.0, worst_trade=0.0, average_holding_period=0.0
                )

            # 完了取引のみを対象
            completed_trades = [t for t in trades if t.action == 'CLOSE' and t.profit_loss is not None]

            if not completed_trades:
                return self._empty_metrics()

            # 基本統計
            total_trades = len(completed_trades)
            profits = [t.profit_loss for t in completed_trades]
            returns = [t.actual_return for t in completed_trades if t.actual_return is not None]

            winning_trades = len([p for p in profits if p > 0])
            losing_trades = len([p for p in profits if p < 0])
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0.0

            # リターン統計
            total_return = sum(profits)
            average_return = np.mean(returns) if returns else 0.0

            # 勝ち負け統計
            winning_profits = [p for p in profits if p > 0]
            losing_profits = [p for p in profits if p < 0]

            average_win = np.mean(winning_profits) if winning_profits else 0.0
            average_loss = abs(np.mean(losing_profits)) if losing_profits else 0.0

            # プロフィットファクター
            total_wins = sum(winning_profits) if winning_profits else 0.0
            total_losses = abs(sum(losing_profits)) if losing_profits else 0.0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

            # シャープレシオ
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if returns and np.std(returns) > 0 else 0.0

            # ソルティーノレシオ
            negative_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(negative_returns) if negative_returns else 0.0
            sortino_ratio = (np.mean(returns) / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0.0

            # 最大ドローダウン
            max_drawdown = self._calculate_max_drawdown(completed_trades)

            # 連続勝敗
            max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_streaks(profits)

            # 87%精度統計
            precision_87_trades = len([t for t in completed_trades if t.precision_87_achieved])
            precision_87_success_rate = 0.0
            if precision_87_trades > 0:
                precision_87_wins = len([t for t in completed_trades
                                       if t.precision_87_achieved and t.profit_loss > 0])
                precision_87_success_rate = precision_87_wins / precision_87_trades * 100

            # ベスト・ワースト取引
            best_trade = max(profits) if profits else 0.0
            worst_trade = min(profits) if profits else 0.0

            # 平均保有期間計算
            average_holding_period = self._calculate_average_holding_period(completed_trades)

            # 総リターン率計算（初期資本基準）
            initial_capital = 1000000  # デフォルト値
            total_return_pct = total_return / initial_capital * 100

            return PerformanceMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_return=total_return,
                total_return_pct=total_return_pct,
                average_return=average_return,
                average_win=average_win,
                average_loss=average_loss,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                precision_87_trades=precision_87_trades,
                precision_87_success_rate=precision_87_success_rate,
                best_trade=best_trade,
                worst_trade=worst_trade,
                average_holding_period=average_holding_period
            )

        except Exception as e:
            self.logger.error(f"パフォーマンスレポート生成エラー: {e}")
            return self._empty_metrics()

    def generate_trading_charts(self,
                              session_id: Optional[str] = None,
                              chart_types: List[str] = None) -> Dict[str, str]:
        """
        取引チャート生成

        Args:
            session_id: セッションID
            chart_types: チャートタイプリスト

        Returns:
            チャート画像（Base64エンコード）辞書
        """
        if chart_types is None:
            chart_types = ['equity_curve', 'monthly_returns', 'trade_distribution', 'precision_analysis']

        charts = {}

        try:
            trades = self._filter_trades(session_id)
            completed_trades = [t for t in trades if t.action == 'CLOSE' and t.profit_loss is not None]

            if not completed_trades:
                return charts

            # 1. 資産曲線
            if 'equity_curve' in chart_types:
                charts['equity_curve'] = self._create_equity_curve(completed_trades)

            # 2. 月次リターン
            if 'monthly_returns' in chart_types:
                charts['monthly_returns'] = self._create_monthly_returns_chart(completed_trades)

            # 3. 取引分布
            if 'trade_distribution' in chart_types:
                charts['trade_distribution'] = self._create_trade_distribution_chart(completed_trades)

            # 4. 87%精度分析
            if 'precision_analysis' in chart_types:
                charts['precision_analysis'] = self._create_precision_analysis_chart(completed_trades)

        except Exception as e:
            self.logger.error(f"チャート生成エラー: {e}")

        return charts

    def export_to_csv(self,
                     filepath: str,
                     session_id: Optional[str] = None,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> bool:
        """
        CSV エクスポート

        Args:
            filepath: 出力ファイルパス
            session_id: セッションID
            start_date: 開始日
            end_date: 終了日

        Returns:
            エクスポート成功フラグ
        """
        try:
            trades = self._filter_trades(session_id, start_date, end_date)

            if not trades:
                return False

            # CSV用データ作成
            csv_data = []
            for trade in trades:
                csv_data.append({
                    'trade_id': trade.trade_id,
                    'session_id': trade.session_id,
                    'timestamp': trade.timestamp.isoformat(),
                    'symbol': trade.symbol,
                    'action': trade.action,
                    'signal_type': trade.signal_type.value,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'confidence': trade.confidence,
                    'precision': trade.precision,
                    'precision_87_achieved': trade.precision_87_achieved,
                    'expected_return': trade.expected_return,
                    'actual_return': trade.actual_return,
                    'profit_loss': trade.profit_loss,
                    'position_size': trade.position_size,
                    'market_value': trade.market_value,
                    'reasoning': trade.reasoning,
                    'commission': trade.trading_costs.get('commission', 0),
                    'spread': trade.trading_costs.get('spread', 0),
                    'total_cost': trade.trading_costs.get('total_cost', 0)
                })

            # CSV書き出し
            df = pd.DataFrame(csv_data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

            self.logger.info(f"CSV エクスポート完了: {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"CSV エクスポートエラー: {e}")
            return False

    def export_to_json(self,
                      filepath: str,
                      session_id: Optional[str] = None,
                      include_performance: bool = True) -> bool:
        """
        JSON エクスポート

        Args:
            filepath: 出力ファイルパス
            session_id: セッションID
            include_performance: パフォーマンス指標含む

        Returns:
            エクスポート成功フラグ
        """
        try:
            trades = self._filter_trades(session_id)

            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'trade_count': len(trades),
                'trades': [asdict(trade) for trade in trades]
            }

            # パフォーマンス指標追加
            if include_performance:
                metrics = self.generate_performance_report(session_id)
                export_data['performance_metrics'] = asdict(metrics)

            # JSON書き出し
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"JSON エクスポート完了: {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"JSON エクスポートエラー: {e}")
            return False

    def calculate_tax_implications(self,
                                 session_id: Optional[str] = None,
                                 tax_year: int = None) -> TaxCalculation:
        """
        税務計算

        Args:
            session_id: セッションID
            tax_year: 課税年度

        Returns:
            税務計算結果
        """
        try:
            if tax_year is None:
                tax_year = datetime.now().year

            # 対象年度の取引抽出
            start_date = datetime(tax_year, 1, 1)
            end_date = datetime(tax_year, 12, 31)
            trades = self._filter_trades(session_id, start_date, end_date)

            completed_trades = [t for t in trades if t.action == 'CLOSE' and t.profit_loss is not None]

            if not completed_trades:
                return TaxCalculation(
                    total_realized_gains=0.0, total_realized_losses=0.0,
                    net_capital_gains=0.0, short_term_gains=0.0, long_term_gains=0.0,
                    wash_sales=0.0, estimated_tax_liability=0.0, deductible_expenses=0.0
                )

            # 実現損益計算
            gains = [t.profit_loss for t in completed_trades if t.profit_loss > 0]
            losses = [abs(t.profit_loss) for t in completed_trades if t.profit_loss < 0]

            total_realized_gains = sum(gains) if gains else 0.0
            total_realized_losses = sum(losses) if losses else 0.0
            net_capital_gains = total_realized_gains - total_realized_losses

            # 短期・長期分類（簡略化: 1年未満=短期）
            short_term_gains = 0.0
            long_term_gains = 0.0

            for trade in completed_trades:
                if trade.profit_loss > 0:
                    holding_days = self._calculate_holding_days(trade)
                    if holding_days < 365:
                        short_term_gains += trade.profit_loss
                    else:
                        long_term_gains += trade.profit_loss

            # 取引コスト合計（控除可能経費）
            deductible_expenses = sum(
                sum(t.trading_costs.values()) for t in completed_trades
            )

            # 簡易税額計算（20.315%の申告分離課税）
            taxable_gains = max(net_capital_gains - deductible_expenses, 0)
            estimated_tax_liability = taxable_gains * 0.20315

            return TaxCalculation(
                total_realized_gains=total_realized_gains,
                total_realized_losses=total_realized_losses,
                net_capital_gains=net_capital_gains,
                short_term_gains=short_term_gains,
                long_term_gains=long_term_gains,
                wash_sales=0.0,  # 簡略化
                estimated_tax_liability=estimated_tax_liability,
                deductible_expenses=deductible_expenses
            )

        except Exception as e:
            self.logger.error(f"税務計算エラー: {e}")
            return TaxCalculation(
                total_realized_gains=0.0, total_realized_losses=0.0,
                net_capital_gains=0.0, short_term_gains=0.0, long_term_gains=0.0,
                wash_sales=0.0, estimated_tax_liability=0.0, deductible_expenses=0.0
            )

    def get_precision_87_analysis(self) -> Dict[str, Any]:
        """87%精度システム分析"""
        try:
            all_trades = self.trade_records
            completed_trades = [t for t in all_trades if t.action == 'CLOSE' and t.profit_loss is not None]

            if not completed_trades:
                return {}

            # 87%精度取引
            precision_87_trades = [t for t in completed_trades if t.precision_87_achieved]
            regular_trades = [t for t in completed_trades if not t.precision_87_achieved]

            # 統計計算
            precision_87_stats = self._calculate_trade_stats(precision_87_trades)
            regular_stats = self._calculate_trade_stats(regular_trades)

            return {
                'total_trades': len(completed_trades),
                'precision_87_trades': len(precision_87_trades),
                'precision_87_ratio': len(precision_87_trades) / len(completed_trades) * 100,
                'precision_87_performance': precision_87_stats,
                'regular_performance': regular_stats,
                'performance_comparison': {
                    'win_rate_difference': precision_87_stats['win_rate'] - regular_stats['win_rate'],
                    'avg_return_difference': precision_87_stats['avg_return'] - regular_stats['avg_return'],
                    'profit_factor_ratio': precision_87_stats['profit_factor'] / regular_stats['profit_factor'] if regular_stats['profit_factor'] > 0 else float('inf')
                }
            }

        except Exception as e:
            self.logger.error(f"87%精度分析エラー: {e}")
            return {}

    def set_session_id(self, session_id: str):
        """セッションID設定"""
        self.session_id = session_id

    def clear_records(self):
        """記録クリア"""
        self.trade_records.clear()

    # --- プライベートメソッド ---

    def _init_database(self):
        """データベース初期化"""
        try:
            # データディレクトリ作成
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    session_id TEXT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER,
                    price REAL,
                    timestamp TEXT,
                    signal_type TEXT,
                    confidence REAL,
                    precision_val REAL,
                    precision_87_achieved BOOLEAN,
                    expected_return REAL,
                    actual_return REAL,
                    profit_loss REAL,
                    trading_costs_json TEXT,
                    position_size REAL,
                    market_value REAL,
                    reasoning TEXT,
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    execution_details_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")

    def _save_to_database(self, trade: TradeRecord):
        """データベース保存"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO trade_records (
                    trade_id, session_id, symbol, action, quantity, price, timestamp,
                    signal_type, confidence, precision_val, precision_87_achieved,
                    expected_return, actual_return, profit_loss, trading_costs_json,
                    position_size, market_value, reasoning, stop_loss_price,
                    take_profit_price, execution_details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.trade_id, trade.session_id, trade.symbol, trade.action,
                trade.quantity, trade.price, trade.timestamp.isoformat(),
                trade.signal_type.value, trade.confidence, trade.precision,
                trade.precision_87_achieved, trade.expected_return,
                trade.actual_return, trade.profit_loss,
                json.dumps(trade.trading_costs), trade.position_size,
                trade.market_value, trade.reasoning, trade.stop_loss_price,
                trade.take_profit_price, json.dumps(trade.execution_details)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"データベース保存エラー: {e}")

    def _filter_trades(self,
                      session_id: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> List[TradeRecord]:
        """取引フィルタリング"""
        trades = self.trade_records

        if session_id:
            trades = [t for t in trades if t.session_id == session_id]

        if start_date:
            trades = [t for t in trades if t.timestamp >= start_date]

        if end_date:
            trades = [t for t in trades if t.timestamp <= end_date]

        return trades

    def _empty_metrics(self) -> PerformanceMetrics:
        """空のメトリクス"""
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, total_return=0.0, total_return_pct=0.0,
            average_return=0.0, average_win=0.0, average_loss=0.0,
            profit_factor=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
            max_drawdown=0.0, max_consecutive_wins=0, max_consecutive_losses=0,
            precision_87_trades=0, precision_87_success_rate=0.0,
            best_trade=0.0, worst_trade=0.0, average_holding_period=0.0
        )

    def _calculate_max_drawdown(self, trades: List[TradeRecord]) -> float:
        """最大ドローダウン計算"""
        if not trades:
            return 0.0

        cumulative_returns = []
        cumulative = 0.0

        for trade in sorted(trades, key=lambda x: x.timestamp):
            cumulative += trade.profit_loss
            cumulative_returns.append(cumulative)

        if not cumulative_returns:
            return 0.0

        peak = cumulative_returns[0]
        max_drawdown = 0.0

        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / abs(peak) if peak != 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown * 100  # パーセント表示

    def _calculate_consecutive_streaks(self, profits: List[float]) -> Tuple[int, int]:
        """連続勝敗計算"""
        if not profits:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for profit in profits:
            if profit > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def _calculate_average_holding_period(self, trades: List[TradeRecord]) -> float:
        """平均保有期間計算"""
        holding_periods = []

        for trade in trades:
            if trade.action == 'CLOSE':
                # 簡略化: OPENとCLOSEの時間差を計算
                # 実際には対応するOPEN取引を見つける必要がある
                holding_days = 1.0  # デフォルト値
                holding_periods.append(holding_days)

        return np.mean(holding_periods) if holding_periods else 0.0

    def _calculate_holding_days(self, trade: TradeRecord) -> int:
        """保有日数計算（簡略化）"""
        # 実際には対応するOPEN取引との差を計算
        return 1  # デフォルト値

    def _calculate_trade_stats(self, trades: List[TradeRecord]) -> Dict[str, float]:
        """取引統計計算"""
        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0.0, 'avg_return': 0.0,
                'profit_factor': 0.0, 'best_trade': 0.0, 'worst_trade': 0.0
            }

        profits = [t.profit_loss for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        return {
            'total_trades': len(trades),
            'win_rate': len(wins) / len(profits) * 100,
            'avg_return': np.mean(profits),
            'profit_factor': sum(wins) / abs(sum(losses)) if losses else float('inf'),
            'best_trade': max(profits),
            'worst_trade': min(profits)
        }

    def _create_equity_curve(self, trades: List[TradeRecord]) -> str:
        """資産曲線チャート作成"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # 累積損益計算
        sorted_trades = sorted(trades, key=lambda x: x.timestamp)
        cumulative_pnl = []
        cumulative = 0.0

        for trade in sorted_trades:
            cumulative += trade.profit_loss
            cumulative_pnl.append(cumulative)

        dates = [t.timestamp for t in sorted_trades]
        ax.plot(dates, cumulative_pnl, linewidth=2, label='累積損益')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('資産曲線', fontsize=14, fontweight='bold')
        ax.set_xlabel('日付')
        ax.set_ylabel('累積損益 (円)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _create_monthly_returns_chart(self, trades: List[TradeRecord]) -> str:
        """月次リターンチャート作成"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # 月次集計
        monthly_returns = {}
        for trade in trades:
            month_key = trade.timestamp.strftime('%Y-%m')
            if month_key not in monthly_returns:
                monthly_returns[month_key] = 0.0
            monthly_returns[month_key] += trade.profit_loss

        months = sorted(monthly_returns.keys())
        returns = [monthly_returns[month] for month in months]

        colors = ['green' if r >= 0 else 'red' for r in returns]
        ax.bar(months, returns, color=colors, alpha=0.7)
        ax.set_title('月次リターン', fontsize=14, fontweight='bold')
        ax.set_xlabel('月')
        ax.set_ylabel('リターン (円)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _create_trade_distribution_chart(self, trades: List[TradeRecord]) -> str:
        """取引分布チャート作成"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        profits = [t.profit_loss for t in trades]

        # ヒストグラム
        ax1.hist(profits, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('取引損益分布', fontsize=12, fontweight='bold')
        ax1.set_xlabel('損益 (円)')
        ax1.set_ylabel('取引数')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)

        # ボックスプロット
        ax2.boxplot(profits)
        ax2.set_title('取引損益ボックスプロット', fontsize=12, fontweight='bold')
        ax2.set_ylabel('損益 (円)')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _create_precision_analysis_chart(self, trades: List[TradeRecord]) -> str:
        """87%精度分析チャート作成"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        precision_87_trades = [t for t in trades if t.precision_87_achieved]
        regular_trades = [t for t in trades if not t.precision_87_achieved]

        # 勝率比較
        if precision_87_trades and regular_trades:
            precision_87_wins = len([t for t in precision_87_trades if t.profit_loss > 0])
            regular_wins = len([t for t in regular_trades if t.profit_loss > 0])

            precision_87_win_rate = precision_87_wins / len(precision_87_trades) * 100
            regular_win_rate = regular_wins / len(regular_trades) * 100

            categories = ['87%精度取引', '通常取引']
            win_rates = [precision_87_win_rate, regular_win_rate]

            ax1.bar(categories, win_rates, color=['gold', 'skyblue'], alpha=0.8)
            ax1.set_title('勝率比較', fontsize=12, fontweight='bold')
            ax1.set_ylabel('勝率 (%)')
            ax1.set_ylim(0, 100)

        # 平均リターン比較
        if precision_87_trades and regular_trades:
            precision_87_avg = np.mean([t.profit_loss for t in precision_87_trades])
            regular_avg = np.mean([t.profit_loss for t in regular_trades])

            categories = ['87%精度取引', '通常取引']
            avg_returns = [precision_87_avg, regular_avg]

            colors = ['green' if r >= 0 else 'red' for r in avg_returns]
            ax2.bar(categories, avg_returns, color=colors, alpha=0.8)
            ax2.set_title('平均リターン比較', fontsize=12, fontweight='bold')
            ax2.set_ylabel('平均リターン (円)')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        """図をBase64エンコード"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return image_base64