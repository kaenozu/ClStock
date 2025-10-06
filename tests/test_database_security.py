"""データベースセキュリティのテスト
"""

import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from utils.database_security import SecureDatabase, get_secure_db


class TestSecureDatabase:
    """SecureDatabase のテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        # 一時的なデータベースファイルを作成
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.db = SecureDatabase(self.temp_db.name)

    def teardown_method(self):
        """各テストメソッドの後に実行"""
        # 一時ファイルを削除
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_create_tables(self):
        """テーブル作成のテスト"""
        self.db.create_tables_if_not_exists()

        # テーブルが作成されたことを確認
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

        assert "stocks" in tables
        assert "stock_prices" in tables

    def test_safe_query_execution(self):
        """安全なクエリ実行のテスト"""
        self.db.create_tables_if_not_exists()

        # データ挿入
        query = "INSERT INTO stocks (symbol, name, sector) VALUES (?, ?, ?)"
        result = self.db.execute_safe_update(
            query, ("TEST", "Test Company", "Technology"),
        )
        assert result == 1

        # データ取得
        query = "SELECT * FROM stocks WHERE symbol = ?"
        results = self.db.execute_safe_query(query, ("TEST",))
        assert len(results) == 1
        assert results[0]["symbol"] == "TEST"
        assert results[0]["name"] == "Test Company"

    def test_dangerous_query_detection(self):
        """危険なクエリの検出"""
        dangerous_queries = [
            "DROP TABLE stocks",
            "DELETE FROM stocks; DROP TABLE stocks;",
            "SELECT * FROM stocks UNION SELECT * FROM sqlite_master",
            "INSERT INTO stocks VALUES ('test', 'name', 'sector'); DROP TABLE stocks;",
            "SELECT * FROM stocks WHERE symbol = 'test'--'",
        ]

        for query in dangerous_queries:
            with pytest.raises(ValueError):
                self.db.execute_safe_query(query)

    def test_stock_data_operations(self):
        """株価データ操作のテスト"""
        self.db.create_tables_if_not_exists()

        # 株式情報を挿入
        stock_query = "INSERT INTO stocks (symbol, name, sector) VALUES (?, ?, ?)"
        self.db.execute_safe_update(stock_query, ("TEST", "Test Company", "Technology"))

        # 株価データを挿入
        result = self.db.insert_stock_data_safe(
            symbol="TEST",
            date="2023-01-01",
            open_price=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000,
        )
        assert result == 1

        # 株価データを取得
        stock_data = self.db.get_stock_data_safe("TEST", limit=10)
        assert len(stock_data) == 1
        # stock_prices テーブルからの取得なので symbol フィールドがないことを確認
        assert stock_data[0]["close"] == 102.0
        assert stock_data[0]["open"] == 100.0
        assert stock_data[0]["high"] == 105.0
        assert stock_data[0]["low"] == 95.0

    def test_stock_search(self):
        """株式検索のテスト"""
        self.db.create_tables_if_not_exists()

        # テストデータを挿入
        stocks = [
            ("AAPL", "Apple Inc.", "Technology"),
            ("GOOGL", "Alphabet Inc.", "Technology"),
            ("TSLA", "Tesla Inc.", "Automotive"),
        ]

        for symbol, name, sector in stocks:
            query = "INSERT INTO stocks (symbol, name, sector) VALUES (?, ?, ?)"
            self.db.execute_safe_update(query, (symbol, name, sector))

        # 検索テスト
        results = self.db.search_stocks_safe("Apple")
        assert len(results) == 1
        assert results[0]["symbol"] == "AAPL"

        results = self.db.search_stocks_safe("Inc")
        assert len(results) == 3  # All have "Inc" in name

        results = self.db.search_stocks_safe("GOOGL")
        assert len(results) == 1
        assert results[0]["symbol"] == "GOOGL"

    def test_parameter_validation(self):
        """パラメータ検証のテスト"""
        self.db.create_tables_if_not_exists()

        # 不正なパラメータ型
        with pytest.raises(Exception):  # sqlite3.Error or similar
            self.db.insert_stock_data_safe(
                symbol="TEST",
                date="2023-01-01",
                open_price="invalid",  # 文字列を数値として
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000000,
            )

    def test_connection_context_manager(self):
        """接続コンテキストマネージャーのテスト"""
        self.db.create_tables_if_not_exists()

        # 正常なケース
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1

        # 例外発生時のロールバックテスト
        with pytest.raises(sqlite3.Error):
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INVALID SQL SYNTAX")

    def test_update_query_validation(self):
        """更新クエリ検証のテスト"""
        self.db.create_tables_if_not_exists()

        # 有効な更新クエリ
        valid_queries = [
            "INSERT INTO stocks (symbol, name) VALUES (?, ?)",
            "UPDATE stocks SET name = ? WHERE symbol = ?",
            "DELETE FROM stocks WHERE symbol = ?",
        ]

        for query in valid_queries:
            # 検証だけ行う（実際には実行しない）
            try:
                self.db._validate_query(query)
            except ValueError:
                pytest.fail(f"Valid query rejected: {query}")

        # 無効なクエリ（SELECT文を更新メソッドで実行）
        with pytest.raises(ValueError):
            self.db.execute_safe_update("SELECT * FROM stocks")


class TestGlobalDatabaseInstance:
    """グローバルデータベースインスタンスのテスト"""

    def test_singleton_behavior(self):
        """シングルトン動作のテスト"""
        db1 = get_secure_db()
        db2 = get_secure_db()

        # 同じインスタンスが返されることを確認
        assert db1 is db2

    @patch("utils.database_security.SecureDatabase")
    def test_database_initialization(self, mock_db_class):
        """データベース初期化のテスト"""
        mock_instance = MagicMock()
        mock_db_class.return_value = mock_instance

        # グローバルインスタンスをリセット
        import utils.database_security

        utils.database_security._db_instance = None

        # インスタンス取得
        db = get_secure_db()

        # データベースクラスが呼ばれたことを確認
        mock_db_class.assert_called_once()
        mock_instance.create_tables_if_not_exists.assert_called_once()

        # クリーンアップ
        utils.database_security._db_instance = None
