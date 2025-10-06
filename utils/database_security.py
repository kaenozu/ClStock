"""データベースセキュリティユーティリティ
SQLインジェクション対策とセキュアなクエリ実行
"""

import logging
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SecureDatabase:
    """セキュアなデータベース接続クラス"""

    def __init__(self, db_path: str = "clstock.db"):
        self.db_path = db_path
        self.connection = None

    @contextmanager
    def get_connection(self):
        """セキュアなデータベース接続を取得"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 辞書形式でアクセス可能
            # セキュリティ設定
            conn.execute("PRAGMA foreign_keys = ON")  # 外部キー制約を有効化
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e!s}")
            raise
        finally:
            if conn:
                conn.close()

    def execute_safe_query(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
        fetch_all: bool = True,
    ) -> List[Dict[str, Any]]:
        """パラメータ化クエリの安全な実行

        Args:
            query: SQL クエリ（パラメータプレースホルダー付き）
            params: クエリパラメータ
            fetch_all: 全結果を取得するか

        Returns:
            クエリ結果のリスト

        Raises:
            ValueError: 危険なクエリが検出された場合
            sqlite3.Error: データベースエラー

        """
        # 危険なクエリパターンをチェック
        self._validate_query(query)

        with self.get_connection() as conn:
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if fetch_all:
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            row = cursor.fetchone()
            return [dict(row)] if row else []

    def execute_safe_update(
        self, query: str, params: Optional[Union[tuple, dict]] = None,
    ) -> int:
        """安全な更新クエリの実行

        Args:
            query: 更新SQL クエリ
            params: クエリパラメータ

        Returns:
            影響を受けた行数

        """
        self._validate_query(query)

        # 更新系クエリかチェック
        query_upper = query.strip().upper()
        if not any(
            query_upper.startswith(cmd) for cmd in ["INSERT", "UPDATE", "DELETE"]
        ):
            raise ValueError("This method only supports INSERT, UPDATE, DELETE queries")

        with self.get_connection() as conn:
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            conn.commit()
            return cursor.rowcount

    def _validate_query(self, query: str):
        """クエリの安全性を検証

        Args:
            query: 検証するSQL クエリ

        Raises:
            ValueError: 危険なクエリパターンが検出された場合

        """
        query_upper = query.upper()

        # 危険なパターンリスト
        dangerous_patterns = [
            "DROP ",
            "TRUNCATE ",
            "ALTER ",
            "CREATE ",
            "EXEC ",
            "EXECUTE ",
            "UNION ",
            "INFORMATION_SCHEMA",
            "SYS.",
            "MASTER.",
            "--",
            "/*",
            "*/",
            ";",
            "SCRIPT",
            "JAVASCRIPT",
            "VBSCRIPT",
        ]

        for pattern in dangerous_patterns:
            if pattern in query_upper:
                logger.warning(
                    f"Potentially dangerous query pattern detected: {pattern}",
                )
                raise ValueError(f"Query contains dangerous pattern: {pattern}")

        # パラメータプレースホルダーの使用を確認
        if "'" in query or '"' in query:
            # 単純な文字列リテラルの存在をチェック
            # 実際の実装では、より精密な解析が必要
            logger.warning("Query may contain string literals - use parameters instead")

    def get_stock_data_safe(
        self, symbol: str, limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """安全な株価データ取得

        Args:
            symbol: 銘柄コード
            limit: 取得件数制限

        Returns:
            株価データのリスト

        """
        query = """
        SELECT date, open, high, low, close, volume
        FROM stock_prices
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT ?
        """

        return self.execute_safe_query(query, (symbol, limit))

    def insert_stock_data_safe(
        self,
        symbol: str,
        date: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ) -> int:
        """安全な株価データ挿入

        Args:
            symbol: 銘柄コード
            date: 日付
            open_price: 始値
            high: 高値
            low: 安値
            close: 終値
            volume: 出来高

        Returns:
            影響を受けた行数

        """
        query = """
        INSERT OR REPLACE INTO stock_prices
        (symbol, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        params = (symbol, date, open_price, high, low, close, volume)
        return self.execute_safe_update(query, params)

    def search_stocks_safe(self, search_term: str) -> List[Dict[str, Any]]:
        """安全な銘柄検索

        Args:
            search_term: 検索語句

        Returns:
            検索結果のリスト

        """
        # 検索語句のサニタイズ
        search_term = str(search_term).strip()[:50]  # 長さ制限

        query = """
        SELECT symbol, name, sector
        FROM stocks
        WHERE symbol LIKE ? OR name LIKE ?
        LIMIT 50
        """

        like_pattern = f"%{search_term}%"
        return self.execute_safe_query(query, (like_pattern, like_pattern))

    def create_tables_if_not_exists(self):
        """必要なテーブルを作成"""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS stocks (
                symbol TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                sector TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS stock_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (symbol) REFERENCES stocks (symbol),
                UNIQUE(symbol, date)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date
            ON stock_prices (symbol, date)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_stock_prices_date
            ON stock_prices (date)
            """,
        ]

        with self.get_connection() as conn:
            for table_sql in tables:
                conn.execute(table_sql)
            conn.commit()

        logger.info("Database tables created/verified successfully")


# グローバルなデータベースインスタンス
_db_instance = None


def get_secure_db() -> SecureDatabase:
    """セキュアなデータベースインスタンスを取得"""
    global _db_instance
    if _db_instance is None:
        _db_instance = SecureDatabase()
        _db_instance.create_tables_if_not_exists()
    return _db_instance
