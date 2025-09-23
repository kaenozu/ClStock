"""
バリデーター関数のテスト
"""

import pytest
from utils.validators import (
    validate_stock_symbol,
    validate_period,
    validate_numeric_range,
    validate_email,
    sanitize_string,
    validate_api_key,
    validate_symbols_list,
    validate_date_range,
    ValidationError,
)


class TestStockSymbolValidation:
    """株式銘柄コード検証のテスト"""

    def test_valid_symbols(self):
        """有効な銘柄コード"""
        assert validate_stock_symbol("AAPL") == "AAPL"
        assert validate_stock_symbol("7203") == "7203"
        assert validate_stock_symbol("BRK.A") == "BRK.A"
        assert validate_stock_symbol("aapl") == "AAPL"  # 大文字変換
        assert validate_stock_symbol(" MSFT ") == "MSFT"  # トリム

    def test_invalid_symbols(self):
        """無効な銘柄コード"""
        with pytest.raises(ValidationError):
            validate_stock_symbol("")  # 空文字

        with pytest.raises(ValidationError):
            validate_stock_symbol("ABC@")  # 無効な文字

        with pytest.raises(ValidationError):
            validate_stock_symbol("A" * 25)  # 長すぎる

        with pytest.raises(ValidationError):
            validate_stock_symbol("ABC DEF")  # スペース


class TestPeriodValidation:
    """期間検証のテスト"""

    def test_valid_periods(self):
        """有効な期間"""
        assert validate_period("1d") == "1d"
        assert validate_period("1Y") == "1y"  # 小文字変換
        assert validate_period(" 1mo ") == "1mo"  # トリム

    def test_invalid_periods(self):
        """無効な期間"""
        with pytest.raises(ValidationError):
            validate_period("")

        with pytest.raises(ValidationError):
            validate_period("1week")

        with pytest.raises(ValidationError):
            validate_period("invalid")


class TestNumericRangeValidation:
    """数値範囲検証のテスト"""

    def test_valid_numbers(self):
        """有効な数値"""
        assert validate_numeric_range("100") == 100.0
        assert validate_numeric_range(50.5) == 50.5
        assert validate_numeric_range(0, min_val=0) == 0.0

    def test_invalid_numbers(self):
        """無効な数値"""
        with pytest.raises(ValidationError):
            validate_numeric_range("abc")

        with pytest.raises(ValidationError):
            validate_numeric_range(5, min_val=10)

        with pytest.raises(ValidationError):
            validate_numeric_range(15, max_val=10)

    def test_range_validation(self):
        """範囲検証"""
        # 範囲内
        assert validate_numeric_range(5, min_val=0, max_val=10) == 5.0

        # 範囲外
        with pytest.raises(ValidationError):
            validate_numeric_range(-1, min_val=0, max_val=10)

        with pytest.raises(ValidationError):
            validate_numeric_range(15, min_val=0, max_val=10)


class TestEmailValidation:
    """メールアドレス検証のテスト"""

    def test_valid_emails(self):
        """有効なメールアドレス"""
        assert validate_email("test@example.com") == "test@example.com"
        assert validate_email("USER@DOMAIN.COM") == "user@domain.com"  # 小文字変換
        assert validate_email(" test@example.org ") == "test@example.org"  # トリム

    def test_invalid_emails(self):
        """無効なメールアドレス"""
        with pytest.raises(ValidationError):
            validate_email("")

        with pytest.raises(ValidationError):
            validate_email("invalid-email")

        with pytest.raises(ValidationError):
            validate_email("test@")

        with pytest.raises(ValidationError):
            validate_email("@example.com")

        with pytest.raises(ValidationError):
            validate_email("a" * 250 + "@example.com")  # 長すぎる


class TestStringValidation:
    """文字列検証のテスト"""

    def test_sanitize_string(self):
        """文字列サニタイズ"""
        assert sanitize_string("normal text") == "normal text"
        assert sanitize_string("  spaced  ") == "spaced"  # トリム
        assert (
            sanitize_string("text\x00with\x1fcontrol") == "textwithcontrol"
        )  # 制御文字除去

    def test_string_too_long(self):
        """長すぎる文字列"""
        with pytest.raises(ValidationError):
            sanitize_string("a" * 1001)

    def test_custom_max_length(self):
        """カスタム最大長"""
        assert sanitize_string("test", max_length=10) == "test"

        with pytest.raises(ValidationError):
            sanitize_string("very long string", max_length=5)


class TestApiKeyValidation:
    """API キー検証のテスト"""

    def test_valid_api_keys(self):
        """有効なAPI キー"""
        assert validate_api_key("abcd1234") == "abcd1234"
        assert validate_api_key("test-key-123") == "test-key-123"
        assert validate_api_key("A1B2C3D4_E5F6") == "A1B2C3D4_E5F6"

    def test_invalid_api_keys(self):
        """無効なAPI キー"""
        with pytest.raises(ValidationError):
            validate_api_key("")

        with pytest.raises(ValidationError):
            validate_api_key("short")  # 短すぎる

        with pytest.raises(ValidationError):
            validate_api_key("a" * 65)  # 長すぎる

        with pytest.raises(ValidationError):
            validate_api_key("invalid@key")  # 無効な文字


class TestSymbolsListValidation:
    """銘柄リスト検証のテスト"""

    def test_valid_symbols_list(self):
        """有効な銘柄リスト"""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        result = validate_symbols_list(symbols)
        assert result == ["AAPL", "GOOGL", "MSFT"]

    def test_duplicate_removal(self):
        """重複除去"""
        symbols = ["AAPL", "GOOGL", "AAPL", "MSFT"]
        result = validate_symbols_list(symbols)
        assert result == ["AAPL", "GOOGL", "MSFT"]

    def test_case_normalization(self):
        """大文字正規化"""
        symbols = ["aapl", "GOOGL", "msft"]
        result = validate_symbols_list(symbols)
        assert result == ["AAPL", "GOOGL", "MSFT"]

    def test_invalid_symbols_list(self):
        """無効な銘柄リスト"""
        with pytest.raises(ValidationError):
            validate_symbols_list([])  # 空リスト

        with pytest.raises(ValidationError):
            validate_symbols_list("not_a_list")  # リストでない

        with pytest.raises(ValidationError):
            validate_symbols_list(["A"] * 101)  # 多すぎる

        with pytest.raises(ValidationError):
            validate_symbols_list(["VALID", "INVALID@"])  # 無効な銘柄含む


class TestDateRangeValidation:
    """日付範囲検証のテスト"""

    def test_valid_date_ranges(self):
        """有効な日付範囲"""
        start, end = validate_date_range("2023-01-01", "2023-12-31")
        assert start == "2023-01-01"
        assert end == "2023-12-31"

        # 片方だけ指定
        start, end = validate_date_range("2023-01-01", None)
        assert start == "2023-01-01"
        assert end is None

    def test_invalid_date_formats(self):
        """無効な日付形式"""
        with pytest.raises(ValidationError):
            validate_date_range("2023/01/01", "2023/12/31")

        with pytest.raises(ValidationError):
            validate_date_range("01-01-2023", "31-12-2023")

        with pytest.raises(ValidationError):
            validate_date_range("invalid", "2023-12-31")

    def test_invalid_date_order(self):
        """無効な日付順序"""
        with pytest.raises(ValidationError):
            validate_date_range("2023-12-31", "2023-01-01")

    def test_date_range_too_long(self):
        """期間が長すぎる"""
        with pytest.raises(ValidationError):
            validate_date_range("2010-01-01", "2025-01-01")  # 15年
