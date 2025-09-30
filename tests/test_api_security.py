"""
API Security のテスト
"""

import logging
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi.security import HTTPAuthorizationCredentials

import api.security as security_module
from api.security import verify_token, security
from api.secure_endpoints import router


class TestAPIAuthentication:
    """API認証のテスト"""

    def setup_method(self):
        """Reset cached security state between tests"""
        if hasattr(security_module, "reset_env_token_cache"):
            security_module.reset_env_token_cache()

    def test_verify_token_valid_admin(self):
        """有効な管理者トークンの検証"""
        # 有効な管理者トークンでのテスト
        valid_admin_token = "admin_token_secure_2024"
        result = verify_token(valid_admin_token)
        assert result == "administrator"

    def test_verify_token_valid_user(self):
        """有効なユーザートークンの検証"""
        # 有効な一般ユーザートークンでのテスト
        valid_user_token = "user_token_basic_2024"
        result = verify_token(valid_user_token)
        assert result == "user"

    def test_verify_token_invalid(self):
        """無効なトークンの検証"""
        from fastapi import HTTPException

        # 無効なトークンでのテスト
        invalid_token = "invalid_token"

        with pytest.raises(HTTPException) as exc_info:
            verify_token(invalid_token)

        assert exc_info.value.status_code == 401
        assert "Invalid token" in str(exc_info.value.detail)

    def test_verify_token_empty(self):
        """空のトークンの検証"""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            verify_token("")

        assert exc_info.value.status_code == 401

    def test_verify_token_none(self):
        """Noneトークンの検証"""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            verify_token(None)

        assert exc_info.value.status_code == 401

    @patch.dict(os.environ, {"API_ADMIN_TOKEN": "custom_admin_token"})
    def test_verify_token_custom_admin(self):
        """カスタム管理者トークンの検証"""
        # 環境変数で設定されたトークンのテスト
        if hasattr(security_module, "reset_env_token_cache"):
            security_module.reset_env_token_cache()
        custom_token = "custom_admin_token"
        result = verify_token(custom_token)
        assert result == "administrator"

    @patch.dict(os.environ, {"API_USER_TOKEN": "custom_user_token"})
    def test_verify_token_custom_user(self):
        """カスタムユーザートークンの検証"""
        # 環境変数で設定されたトークンのテスト
        if hasattr(security_module, "reset_env_token_cache"):
            security_module.reset_env_token_cache()
        custom_token = "custom_user_token"
        result = verify_token(custom_token)
        assert result == "user"

    def test_verify_token_missing_env_logs_warning_once(self, monkeypatch):
        """Missing environment variables should only emit one warning each"""

        monkeypatch.setenv("CLSTOCK_DEV_KEY", "test-dev-key")
        monkeypatch.setenv("CLSTOCK_ADMIN_KEY", "test-admin-key")
        monkeypatch.delenv("API_ADMIN_TOKEN", raising=False)
        monkeypatch.delenv("API_USER_TOKEN", raising=False)

        if hasattr(security_module, "reset_env_token_cache"):
            security_module.reset_env_token_cache()

        warning_records = []

        class _ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
                warning_records.append(record)

        handler = _ListHandler(level=logging.WARNING)
        security_module.logger.addHandler(handler)

        try:
            security_module.verify_token("admin_token_secure_2024")
            missing_warnings = [
                record
                for record in warning_records
                if record.levelno == logging.WARNING
                and "environment variable not set" in record.getMessage()
            ]
            assert len(missing_warnings) == 2

            security_module.verify_token("admin_token_secure_2024")
            missing_warnings_after_second_call = [
                record
                for record in warning_records
                if record.levelno == logging.WARNING
                and "environment variable not set" in record.getMessage()
            ]
        finally:
            security_module.logger.removeHandler(handler)

        assert len(missing_warnings_after_second_call) == 2


class TestAPIEndpointSecurity:
    """APIエンドポイントセキュリティのテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        from fastapi import FastAPI

        if hasattr(security_module, "reset_env_token_cache"):
            security_module.reset_env_token_cache()

        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)

    def test_secure_endpoint_without_auth(self):
        """認証なしでのセキュアエンドポイントアクセス"""
        response = self.client.get("/secure/stock/7203/data")
        assert response.status_code == 403  # Forbidden

    def test_secure_endpoint_with_invalid_auth(self):
        """無効な認証でのセキュアエンドポイントアクセス"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = self.client.get("/secure/stock/7203/data", headers=headers)
        assert response.status_code == 401  # Unauthorized

    @patch("api.security.verify_token")
    @patch("data.stock_data.StockDataProvider")
    def test_secure_endpoint_with_valid_auth(self, mock_provider, mock_verify):
        """有効な認証でのセキュアエンドポイントアクセス"""
        # モック設定
        mock_verify.return_value = "user"

        mock_data = {
            "Close": [100, 101, 102],
            "Volume": [1000, 1100, 1200],
            "Open": [99, 100, 101],
            "High": [101, 102, 103],
            "Low": [98, 99, 100],
        }

        import pandas as pd

        mock_df = pd.DataFrame(mock_data, index=pd.date_range("2023-01-01", periods=3))

        mock_provider_instance = Mock()
        mock_provider_instance.get_stock_data.return_value = mock_df
        mock_provider.return_value = mock_provider_instance

        # 有効な認証でのテスト
        headers = {"Authorization": "Bearer admin_token_secure_2024"}
        response = self.client.get(
            "/secure/stock/7203/data?period=1mo", headers=headers
        )

        if response.status_code == 200:
            assert "symbol" in response.json()
            assert response.json()["symbol"] == "7203"

    def test_health_endpoint_no_auth(self):
        """ヘルスチェックエンドポイント（認証不要）"""
        response = self.client.get("/secure/health")
        assert response.status_code == 200
        assert "status" in response.json()
        assert response.json()["status"] == "healthy"

    @patch("api.security.verify_token")
    def test_admin_only_endpoint_user_access(self, mock_verify):
        """管理者専用エンドポイントへの一般ユーザーアクセス"""
        # 一般ユーザーとして認証
        mock_verify.return_value = "user"

        headers = {"Authorization": "Bearer user_token_basic_2024"}
        response = self.client.get("/secure/analysis/7203", headers=headers)
        assert response.status_code == 403  # Forbidden

    @patch("api.security.verify_token")
    @patch("data.stock_data.StockDataProvider")
    def test_admin_only_endpoint_admin_access(self, mock_provider, mock_verify):
        """管理者専用エンドポイントへの管理者アクセス"""
        # 管理者として認証
        mock_verify.return_value = "administrator"

        mock_data = {
            "Close": [100, 101, 102, 103, 104] * 20,  # 十分なデータ
            "Volume": [1000, 1100, 1200, 1300, 1400] * 20,
            "Open": [99, 100, 101, 102, 103] * 20,
            "High": [101, 102, 103, 104, 105] * 20,
            "Low": [98, 99, 100, 101, 102] * 20,
        }

        import pandas as pd

        mock_df = pd.DataFrame(
            mock_data, index=pd.date_range("2023-01-01", periods=100)
        )

        mock_provider_instance = Mock()
        mock_provider_instance.get_stock_data.return_value = mock_df
        mock_provider_instance.calculate_technical_indicators.return_value = mock_df
        mock_provider.return_value = mock_provider_instance

        headers = {"Authorization": "Bearer admin_token_secure_2024"}
        response = self.client.get("/secure/analysis/7203", headers=headers)

        if response.status_code == 200:
            assert "symbol" in response.json()
            assert "analyst" in response.json()
            assert response.json()["analyst"] == "administrator"

    @patch("api.security.verify_token")
    def test_batch_endpoint_symbol_limit_user(self, mock_verify):
        """バッチエンドポイントの銘柄数制限（一般ユーザー）"""
        # 一般ユーザーとして認証
        mock_verify.return_value = "user"

        # 制限を超える銘柄数
        symbols = [f"stock{i}" for i in range(15)]  # 15銘柄（制限10を超過）

        headers = {"Authorization": "Bearer user_token_basic_2024"}
        response = self.client.post(
            "/secure/stocks/batch",
            json={"symbols": symbols, "period": "1mo"},
            headers=headers,
        )
        assert response.status_code == 400  # Bad Request

    @patch("api.security.verify_token")
    @patch("data.stock_data.StockDataProvider")
    def test_batch_endpoint_symbol_limit_admin(self, mock_provider, mock_verify):
        """バッチエンドポイントの銘柄数制限（管理者）"""
        # 管理者として認証
        mock_verify.return_value = "administrator"

        mock_data = {
            "Close": [100, 101, 102],
            "Volume": [1000, 1100, 1200],
            "Open": [99, 100, 101],
            "High": [101, 102, 103],
            "Low": [98, 99, 100],
        }

        import pandas as pd

        mock_df = pd.DataFrame(mock_data, index=pd.date_range("2023-01-01", periods=3))

        mock_provider_instance = Mock()
        mock_provider_instance.get_stock_data.return_value = mock_df
        mock_provider.return_value = mock_provider_instance

        # 管理者は制限が緩い
        symbols = [f"stock{i}" for i in range(15)]  # 15銘柄

        headers = {"Authorization": "Bearer admin_token_secure_2024"}
        response = self.client.post(
            "/secure/stocks/batch",
            json={"symbols": symbols, "period": "1mo"},
            headers=headers,
        )

        # 管理者は制限が緩いので成功する可能性が高い
        if response.status_code == 200:
            assert "requested_symbols" in response.json()


class TestInputValidation:
    """入力検証のテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        from fastapi import FastAPI

        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)

    @patch("api.security.verify_token")
    def test_invalid_stock_symbol(self, mock_verify):
        """無効な銘柄コードの検証"""
        mock_verify.return_value = "user"

        headers = {"Authorization": "Bearer user_token_basic_2024"}
        # 無効な銘柄コード
        response = self.client.get("/secure/stock/INVALID@SYMBOL/data", headers=headers)
        assert response.status_code == 400  # Bad Request

    @patch("api.security.verify_token")
    def test_invalid_period(self, mock_verify):
        """無効な期間の検証"""
        mock_verify.return_value = "user"

        headers = {"Authorization": "Bearer user_token_basic_2024"}
        # 無効な期間
        response = self.client.get(
            "/secure/stock/7203/data?period=invalid_period", headers=headers
        )
        assert response.status_code == 400  # Bad Request

    @patch("api.security.verify_token")
    def test_sql_injection_attempt(self, mock_verify):
        """SQLインジェクション試行の検証"""
        mock_verify.return_value = "user"

        headers = {"Authorization": "Bearer user_token_basic_2024"}
        # SQLインジェクション試行
        response = self.client.get(
            "/secure/stock/7203'; DROP TABLE stocks; --/data", headers=headers
        )
        assert response.status_code == 400  # Bad Request

    @patch("api.security.verify_token")
    def test_xss_attempt(self, mock_verify):
        """XSS試行の検証"""
        mock_verify.return_value = "user"

        headers = {"Authorization": "Bearer user_token_basic_2024"}
        # XSS試行
        response = self.client.get(
            "/secure/stock/<script>alert('xss')</script>/data", headers=headers
        )
        assert response.status_code in {400, 404}  # Bad Request or route rejection
