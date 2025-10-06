from __future__ import annotations

"""取引データ永続化リポジトリ"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from trading.models import TradeRecord
from trading.trading_strategy import SignalType


class TradeRepository:
    """SQLite ベースの取引リポジトリ"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_database()

    def init_database(self) -> None:
        """データベース初期化"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
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
                    """,
                )
                conn.commit()
        except Exception as exc:
            self.logger.error("データベース初期化エラー: %s", exc)

    def save_trade(self, trade: TradeRecord) -> None:
        """取引記録を保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO trade_records (
                        trade_id, session_id, symbol, action, quantity, price, timestamp,
                        signal_type, confidence, precision_val, precision_87_achieved,
                        expected_return, actual_return, profit_loss, trading_costs_json,
                        position_size, market_value, reasoning, stop_loss_price,
                        take_profit_price, execution_details_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade.trade_id,
                        trade.session_id,
                        trade.symbol,
                        trade.action,
                        trade.quantity,
                        trade.price,
                        trade.timestamp.isoformat(),
                        trade.signal_type.value,
                        trade.confidence,
                        trade.precision,
                        trade.precision_87_achieved,
                        trade.expected_return,
                        trade.actual_return,
                        trade.profit_loss,
                        json.dumps(trade.trading_costs),
                        trade.position_size,
                        trade.market_value,
                        trade.reasoning,
                        trade.stop_loss_price,
                        trade.take_profit_price,
                        json.dumps(trade.execution_details),
                    ),
                )
                conn.commit()
        except Exception as exc:
            self.logger.error("データベース保存エラー: %s", exc)

    def fetch_trades(
        self,
        session_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[TradeRecord]:
        """保存済み取引を取得"""
        query = [
            "SELECT trade_id, session_id, symbol, action, quantity, price, timestamp,",
            "signal_type, confidence, precision_val, precision_87_achieved,",
            "expected_return, actual_return, profit_loss, trading_costs_json,",
            "position_size, market_value, reasoning, stop_loss_price,",
            "take_profit_price, execution_details_json FROM trade_records WHERE 1=1",
        ]
        params: List[object] = []

        if session_id:
            query.append("AND session_id = ?")
            params.append(session_id)
        if start_date:
            query.append("AND timestamp >= ?")
            params.append(start_date.isoformat())
        if end_date:
            query.append("AND timestamp <= ?")
            params.append(end_date.isoformat())

        query.append("ORDER BY timestamp ASC")
        sql = " ".join(query)

        records: List[TradeRecord] = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for row in cursor.execute(sql, params):
                    (
                        trade_id,
                        sess_id,
                        symbol,
                        action,
                        quantity,
                        price,
                        timestamp,
                        signal_type,
                        confidence,
                        precision_val,
                        precision_87_achieved,
                        expected_return,
                        actual_return,
                        profit_loss,
                        trading_costs_json,
                        position_size,
                        market_value,
                        reasoning,
                        stop_loss_price,
                        take_profit_price,
                        execution_details_json,
                    ) = row

                    trading_costs = json.loads(trading_costs_json or "{}")
                    execution_details = json.loads(execution_details_json or "{}")

                    parsed_timestamp = (
                        datetime.fromisoformat(timestamp)
                        if timestamp
                        else datetime.now()
                    )

                    records.append(
                        TradeRecord(
                            trade_id=trade_id,
                            session_id=sess_id or "",
                            symbol=symbol,
                            action=action,
                            quantity=quantity or 0,
                            price=price or 0.0,
                            timestamp=parsed_timestamp,
                            signal_type=SignalType(signal_type),
                            confidence=confidence or 0.0,
                            precision=precision_val or 0.0,
                            precision_87_achieved=bool(precision_87_achieved),
                            expected_return=expected_return or 0.0,
                            actual_return=actual_return,
                            profit_loss=profit_loss,
                            trading_costs=trading_costs,
                            position_size=position_size or 0.0,
                            market_value=market_value or 0.0,
                            reasoning=reasoning or "",
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                            execution_details=execution_details,
                        ),
                    )
        except Exception as exc:
            self.logger.error("取引取得エラー: %s", exc)

        return records

    def save_all(self, trades: Iterable[TradeRecord]) -> None:
        """複数取引の保存"""
        for trade in trades:
            self.save_trade(trade)
