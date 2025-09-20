import pytest
import asyncio
from unittest.mock import patch, Mock
from io import StringIO
import sys
import json

# recommend.pyから関数をインポート
from recommend import (
    format_currency, format_percentage, print_recommendation,
    print_header, print_footer, main
)


class TestRecommendModule:
    """recommend.pyモジュールのテストクラス"""

    @pytest.mark.unit
    def test_format_currency(self):
        """通貨フォーマットのテスト"""
        assert format_currency(1000) == "1,000円"
        assert format_currency(1234567) == "1,234,567円"
        assert format_currency(0) == "0円"

    @pytest.mark.unit
    def test_format_percentage(self):
        """パーセンテージフォーマットのテスト"""
        assert format_percentage(110, 100) == "+10.0%"
        assert format_percentage(90, 100) == "-10.0%"
        assert format_percentage(100, 100) == "+0.0%"

    @pytest.mark.unit
    def test_print_recommendation(self, sample_recommendation, capsys):
        """推奨情報表示のテスト"""
        print_recommendation(sample_recommendation)
        captured = capsys.readouterr()

        assert sample_recommendation.company_name in captured.out
        assert sample_recommendation.symbol in captured.out
        assert "[買]" in captured.out
        assert "[価]" in captured.out
        assert "[損]" in captured.out
        assert "[利]" in captured.out

    @pytest.mark.unit
    def test_print_header(self, capsys):
        """ヘッダー表示のテスト"""
        print_header()
        captured = capsys.readouterr()

        assert "★" in captured.out
        assert "今週のおすすめ銘柄" in captured.out
        assert "=" in captured.out

    @pytest.mark.unit
    def test_print_footer(self, capsys):
        """フッター表示のテスト"""
        print_footer()
        captured = capsys.readouterr()

        assert "[時]" in captured.out
        assert "[注]" in captured.out
        assert "投資は自己責任" in captured.out

    @pytest.mark.unit
    def test_main_list_option(self, capsys):
        """--listオプションのテスト"""
        with patch('sys.argv', ['recommend.py', '--list']):
            with patch('data.stock_data.StockDataProvider') as mock_provider_class:
                mock_provider = Mock()
                mock_provider.jp_stock_codes = {
                    "7203": "トヨタ自動車",
                    "6758": "ソニーグループ"
                }
                mock_provider_class.return_value = mock_provider

                asyncio.run(main())
                captured = capsys.readouterr()

                assert "[一覧]" in captured.out
                assert "7203: トヨタ自動車" in captured.out
                assert "6758: ソニーグループ" in captured.out

    @pytest.mark.unit
    def test_main_single_symbol(self, sample_recommendation, capsys):
        """--symbolオプションのテスト"""
        with patch('sys.argv', ['recommend.py', '--symbol', '7203']):
            with patch('models.predictor.StockPredictor') as mock_predictor_class:
                mock_predictor = Mock()
                mock_predictor.generate_recommendation.return_value = sample_recommendation
                mock_predictor_class.return_value = mock_predictor

                asyncio.run(main())
                captured = capsys.readouterr()

                assert sample_recommendation.company_name in captured.out
                assert sample_recommendation.symbol in captured.out

    @pytest.mark.unit
    def test_main_json_output(self, sample_recommendation, capsys):
        """--jsonオプションのテスト"""
        with patch('sys.argv', ['recommend.py', '--symbol', '7203', '--json']):
            with patch('models.predictor.StockPredictor') as mock_predictor_class:
                mock_predictor = Mock()
                mock_predictor.generate_recommendation.return_value = sample_recommendation
                mock_predictor_class.return_value = mock_predictor

                asyncio.run(main())
                captured = capsys.readouterr()

                # JSON形式で出力されているかチェック
                try:
                    json_data = json.loads(captured.out)
                    assert json_data["symbol"] == "7203"
                    assert json_data["company_name"] == "トヨタ自動車"
                except json.JSONDecodeError:
                    pytest.fail("Output is not valid JSON")

    @pytest.mark.unit
    def test_main_top_recommendations(self, sample_recommendation, capsys):
        """デフォルト（上位推奨）のテスト"""
        with patch('sys.argv', ['recommend.py']):
            with patch('models.predictor.StockPredictor') as mock_predictor_class:
                mock_predictor = Mock()
                mock_predictor.get_top_recommendations.return_value = [sample_recommendation]
                mock_predictor_class.return_value = mock_predictor

                asyncio.run(main())
                captured = capsys.readouterr()

                assert "★" in captured.out  # ヘッダーの星マーク
                assert sample_recommendation.company_name in captured.out

    @pytest.mark.unit
    def test_main_top_with_parameter(self, sample_recommendation, capsys):
        """--topパラメータのテスト"""
        with patch('sys.argv', ['recommend.py', '--top', '3']):
            with patch('models.predictor.StockPredictor') as mock_predictor_class:
                mock_predictor = Mock()
                recommendations = [sample_recommendation] * 3
                for i, rec in enumerate(recommendations):
                    rec.rank = i + 1
                mock_predictor.get_top_recommendations.return_value = recommendations
                mock_predictor_class.return_value = mock_predictor

                asyncio.run(main())
                captured = capsys.readouterr()

                # 3件の推奨が表示されているかチェック
                assert captured.out.count("[1位]") >= 1
                assert captured.out.count("[2位]") >= 1
                assert captured.out.count("[3位]") >= 1

    @pytest.mark.unit
    def test_main_error_handling(self, capsys):
        """エラーハンドリングのテスト"""
        with patch('sys.argv', ['recommend.py', '--symbol', 'INVALID']):
            with patch('models.predictor.StockPredictor') as mock_predictor_class:
                mock_predictor = Mock()
                mock_predictor.generate_recommendation.side_effect = Exception("Test error")
                mock_predictor_class.return_value = mock_predictor

                asyncio.run(main())
                captured = capsys.readouterr()

                assert "[ERROR]" in captured.out
                assert "INVALID" in captured.out

    @pytest.mark.unit
    def test_main_keyboard_interrupt(self, capsys):
        """キーボード割り込みのテスト"""
        with patch('sys.argv', ['recommend.py']):
            with patch('models.predictor.StockPredictor') as mock_predictor_class:
                mock_predictor = Mock()
                mock_predictor.get_top_recommendations.side_effect = KeyboardInterrupt()
                mock_predictor_class.return_value = mock_predictor

                asyncio.run(main())
                captured = capsys.readouterr()

                assert "[WARN]" in captured.out
                assert "中断されました" in captured.out

    @pytest.mark.unit
    def test_main_json_top_recommendations(self, sample_recommendation, capsys):
        """上位推奨のJSON出力テスト"""
        with patch('sys.argv', ['recommend.py', '--top', '2', '--json']):
            with patch('models.predictor.StockPredictor') as mock_predictor_class:
                mock_predictor = Mock()
                recommendations = [sample_recommendation] * 2
                mock_predictor.get_top_recommendations.return_value = recommendations
                mock_predictor_class.return_value = mock_predictor

                asyncio.run(main())
                captured = capsys.readouterr()

                try:
                    json_data = json.loads(captured.out)
                    assert "recommendations" in json_data
                    assert "generated_at" in json_data
                    assert "market_status" in json_data
                    assert len(json_data["recommendations"]) == 2
                except json.JSONDecodeError:
                    pytest.fail("Output is not valid JSON")

    @pytest.mark.unit
    def test_unicode_output_handling(self, sample_recommendation, capsys):
        """Unicode出力処理のテスト"""
        # Windows環境でのUnicode出力が正常に動作するかテスト
        with patch('sys.argv', ['recommend.py', '--symbol', '7203']):
            with patch('models.predictor.StockPredictor') as mock_predictor_class:
                mock_predictor = Mock()
                mock_predictor.generate_recommendation.return_value = sample_recommendation
                mock_predictor_class.return_value = mock_predictor

                try:
                    asyncio.run(main())
                    captured = capsys.readouterr()

                    # 日本語文字が含まれているかチェック
                    assert "トヨタ自動車" in captured.out
                    assert "円" in captured.out
                except UnicodeEncodeError:
                    pytest.fail("Unicode encoding error occurred")