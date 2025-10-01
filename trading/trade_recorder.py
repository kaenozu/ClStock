"""
ClStock 取引記録・レポートシステム

取引履歴の詳細記録とパフォーマンスレポート生成
87%精度システムの成果追跡とCSV/JSONエクスポート機能
"""

from __future__ import annotations

import logging
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from trading.models import PerformanceMetrics, TaxCalculation, TradeRecord
from trading.persistence.trade_repository import TradeRepository
from trading.reporting.charts import ChartGenerator
from trading.reporting.exporters import ExportService
from trading.reporting.performance_metrics import PerformanceMetricsService
from trading.tax.tax_calculator import TaxCalculator

from .trading_strategy import SignalType


_DEFAULT_DB_SUBPATH = Path("gemini-desktop") / "ClStock" / "data" / "trading_records.db"


def _resolve_db_path(db_path: Optional[Union[str, PathLike]]) -> str:
    """Resolve the database path ensuring the parent directory exists."""

    if db_path is None:
        resolved_path = Path.home() / _DEFAULT_DB_SUBPATH
    else:
        resolved_path = Path(db_path)

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return str(resolved_path)


class TradeRecorder:
    """取引記録・レポートシステム"""

    def __init__(
        self,
        db_path: Optional[Union[str, PathLike]] = None,
        *,
        repository: Optional[TradeRepository] = None,
        metrics_service: Optional[PerformanceMetricsService] = None,
        chart_generator: Optional[ChartGenerator] = None,
        export_service: Optional[ExportService] = None,
        tax_calculator: Optional[TaxCalculator] = None,
    ):
        """初期化"""

        resolved_db_path = _resolve_db_path(db_path)
        self.db_path = resolved_db_path
        self.repository = repository or TradeRepository(resolved_db_path)
        self.metrics_service = metrics_service or PerformanceMetricsService()
        self.chart_generator = chart_generator or ChartGenerator()
        self.export_service = export_service or ExportService()
        self.tax_calculator = tax_calculator or TaxCalculator()

        self.repository.init_database()

        self.trade_records: List[TradeRecord] = []
        self.session_id: Optional[str] = None
        self.logger = logging.getLogger(__name__)

    def record_trade(self, trade_data: Dict[str, Any]) -> bool:
        """取引記録"""

        try:
            trade_record = TradeRecord(
                trade_id=trade_data.get("trade_id", ""),
                session_id=trade_data.get("session_id", self.session_id or ""),
                symbol=trade_data.get("symbol", ""),
                action=trade_data.get("action", ""),
                quantity=trade_data.get("quantity", 0),
                price=trade_data.get("price", 0.0),
                timestamp=datetime.fromisoformat(
                    trade_data.get("timestamp", datetime.now().isoformat())
                ),
                signal_type=SignalType(trade_data.get("signal_type", SignalType.HOLD.value)),
                confidence=trade_data.get("confidence", 0.0),
                precision=trade_data.get("precision", 0.0),
                precision_87_achieved=trade_data.get("precision_87_achieved", False),
                expected_return=trade_data.get("expected_return", 0.0),
                actual_return=trade_data.get("actual_return"),
                profit_loss=trade_data.get("profit_loss"),
                trading_costs=trade_data.get("trading_costs", {}),
                position_size=trade_data.get("position_size", 0.0),
                market_value=trade_data.get("market_value", 0.0),
                reasoning=trade_data.get("reasoning", ""),
                stop_loss_price=trade_data.get("stop_loss_price"),
                take_profit_price=trade_data.get("take_profit_price"),
                execution_details=trade_data.get("execution_details", {}),
            )

            self.trade_records.append(trade_record)
            self.repository.save_trade(trade_record)

            self.logger.info("取引記録: %s %s", trade_record.symbol, trade_record.action)
            return True

        except Exception as exc:
            self.logger.error("取引記録エラー: %s", exc)
            return False

    def generate_performance_report(
        self,
        session_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> PerformanceMetrics:
        """パフォーマンスレポート生成"""

        trades = self._filter_trades(session_id, start_date, end_date)
        return self.metrics_service.generate_report(trades)

    def generate_trading_charts(
        self,
        session_id: Optional[str] = None,
        chart_types: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """取引チャート生成"""

        if chart_types is None:
            chart_types = [
                "equity_curve",
                "monthly_returns",
                "trade_distribution",
                "precision_analysis",
            ]

        charts: Dict[str, str] = {}
        trades = self._filter_trades(session_id)
        completed_trades = [
            trade for trade in trades if trade.action == "CLOSE" and trade.profit_loss is not None
        ]

        if not completed_trades:
            return charts

        chart_builders = {
            "equity_curve": self.chart_generator.create_equity_curve,
            "monthly_returns": self.chart_generator.create_monthly_returns_chart,
            "trade_distribution": self.chart_generator.create_trade_distribution_chart,
            "precision_analysis": self.chart_generator.create_precision_analysis_chart,
        }

        for chart_type in chart_types:
            builder = chart_builders.get(chart_type)
            if builder:
                try:
                    charts[chart_type] = builder(completed_trades)
                except Exception as exc:
                    self.logger.error("チャート生成エラー(%s): %s", chart_type, exc)

        return charts

    def export_to_csv(
        self,
        filepath: str,
        session_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> bool:
        """CSV エクスポート"""

        trades = self._filter_trades(session_id, start_date, end_date)
        try:
            return self.export_service.export_to_csv(trades, filepath)
        except Exception as exc:
            self.logger.error("CSV エクスポートエラー: %s", exc)
            return False

    def export_to_json(
        self,
        filepath: str,
        session_id: Optional[str] = None,
        include_performance: bool = True,
    ) -> bool:
        """JSON エクスポート"""

        trades = self._filter_trades(session_id)
        metrics: Optional[PerformanceMetrics] = None
        if include_performance:
            metrics = self.metrics_service.generate_report(trades)

        try:
            return self.export_service.export_to_json(
                trades,
                filepath,
                session_id=session_id,
                include_performance=include_performance,
                performance_metrics=metrics,
            )
        except Exception as exc:
            self.logger.error("JSON エクスポートエラー: %s", exc)
            return False

    def calculate_tax_implications(
        self,
        session_id: Optional[str] = None,
        tax_year: Optional[int] = None,
    ) -> TaxCalculation:
        """税務計算"""

        trades = self._filter_trades(session_id)
        try:
            return self.tax_calculator.calculate(trades, tax_year)
        except Exception as exc:
            self.logger.error("税務計算エラー: %s", exc)
            return TaxCalculation(
                total_realized_gains=0.0,
                total_realized_losses=0.0,
                net_capital_gains=0.0,
                short_term_gains=0.0,
                long_term_gains=0.0,
                wash_sales=0.0,
                estimated_tax_liability=0.0,
                deductible_expenses=0.0,
            )

    def get_precision_87_analysis(self) -> Dict[str, Any]:
        """87%精度システム分析"""

        try:
            return self.metrics_service.get_precision_87_analysis(self.trade_records)
        except Exception as exc:
            self.logger.error("87%精度分析エラー: %s", exc)
            return {}

    def set_session_id(self, session_id: str):
        """セッションID設定"""

        self.session_id = session_id

    def clear_records(self):
        """記録クリア"""

        self.trade_records.clear()

    def _filter_trades(
        self,
        session_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[TradeRecord]:
        """取引フィルタリング"""

        trades = self.trade_records

        if session_id:
            trades = [trade for trade in trades if trade.session_id == session_id]
        if start_date:
            trades = [trade for trade in trades if trade.timestamp >= start_date]
        if end_date:
            trades = [trade for trade in trades if trade.timestamp <= end_date]

        return trades
