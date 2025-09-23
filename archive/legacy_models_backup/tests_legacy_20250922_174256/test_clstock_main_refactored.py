"""
Test for clstock_main refactored functions
"""

import pytest
from unittest.mock import patch, Mock
import sys
from io import StringIO
from clstock_main import (
    _format_basic_prediction_result,
    run_basic_prediction,
    _format_advanced_prediction_result,
    run_advanced_prediction,
    _format_sentiment_analysis_result,
    run_sentiment_analysis,
    _format_integrated_analysis_result,
    run_integrated_analysis,
    _format_portfolio_backtest_result,
    run_portfolio_backtest,
    _format_auto_retraining_status,
    run_auto_retraining_status
)


class TestClStockMainRefactored:
    """Test refactored functions in clstock_main.py"""

    def test_format_basic_prediction_result(self):
        """Test _format_basic_prediction_result function"""
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        result = {
            "direction": 1,
            "confidence": 0.85,
            "predicted_price": 10000.0
        }
        
        _format_basic_prediction_result("7203", result)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "基本予測システム結果" in output
        assert "7203" in output
        assert "上昇" in output
        assert "85.0%" in output

    def test_run_basic_prediction_success(self):
        """Test run_basic_prediction function with success"""
        with patch('clstock_main._execute_basic_prediction') as mock_execute, \
             patch('clstock_main._format_basic_prediction_result') as mock_format:
            
            mock_execute.return_value = {
                "direction": 1,
                "confidence": 0.85,
                "predicted_price": 10000.0
            }
            
            result = run_basic_prediction("7203")
            
            assert result["direction"] == 1
            assert result["confidence"] == 0.85
            mock_execute.assert_called_once_with("7203")
            mock_format.assert_called_once()

    def test_run_basic_prediction_exception(self):
        """Test run_basic_prediction function with exception"""
        with patch('clstock_main._execute_basic_prediction') as mock_execute:
            mock_execute.side_effect = Exception("Test error")
            
            result = run_basic_prediction("7203")
            
            assert "error" in result
            assert "予期しないエラー" in result["error"]

    def test_format_advanced_prediction_result(self):
        """Test _format_advanced_prediction_result function"""
        captured_output = StringIO()
        sys.stdout = captured_output

        result = {
            "prediction": 1,
            "confidence": 0.9
        }
        
        _format_advanced_prediction_result("6758", result)
        
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "個別銘柄特化予測結果" in output
        assert "6758" in output

    def test_run_advanced_prediction_success(self):
        """Test run_advanced_prediction function with success"""
        with patch('clstock_main._execute_advanced_prediction') as mock_execute, \
             patch('clstock_main._format_advanced_prediction_result') as mock_format:
            
            mock_execute.return_value = {
                "prediction": 1,
                "confidence": 0.9
            }
            
            result = run_advanced_prediction("6758")
            
            assert result["prediction"] == 1
            assert result["confidence"] == 0.9
            mock_execute.assert_called_once_with("6758")
            mock_format.assert_called_once()

    def test_format_sentiment_analysis_result(self):
        """Test _format_sentiment_analysis_result function"""
        captured_output = StringIO()
        sys.stdout = captured_output

        sentiment = {
            "sentiment_score": 0.7
        }
        
        _format_sentiment_analysis_result("9984", sentiment)
        
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "ニュースセンチメント分析結果" in output
        assert "9984" in output

    def test_run_sentiment_analysis_success(self):
        """Test run_sentiment_analysis function with success"""
        with patch('clstock_main._execute_sentiment_analysis') as mock_execute, \
             patch('clstock_main._format_sentiment_analysis_result') as mock_format:
            
            mock_execute.return_value = {
                "sentiment_score": 0.7
            }
            
            result = run_sentiment_analysis("9984")
            
            assert result["sentiment_score"] == 0.7
            mock_execute.assert_called_once_with("9984")
            mock_format.assert_called_once()

    def test_format_integrated_analysis_result(self):
        """Test _format_integrated_analysis_result function"""
        captured_output = StringIO()
        sys.stdout = captured_output

        integrated = {
            "technical": {"confidence": 0.8},
            "sentiment": {"sentiment_score": 0.7},
            "integrated": {"integrated_signal": 1}
        }
        
        _format_integrated_analysis_result("8306", integrated)
        
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "統合分析結果" in output
        assert "8306" in output
        assert "技術分析信頼度" in output

    def test_run_integrated_analysis_success(self):
        """Test run_integrated_analysis function with success"""
        with patch('clstock_main._execute_integrated_analysis') as mock_execute, \
             patch('clstock_main._format_integrated_analysis_result') as mock_format:
            
            mock_execute.return_value = {
                "technical": {"confidence": 0.8},
                "sentiment": {"sentiment_score": 0.7},
                "integrated": {"integrated_signal": 1}
            }
            
            result = run_integrated_analysis("8306")
            
            assert "technical" in result
            assert "sentiment" in result
            assert "integrated" in result
            mock_execute.assert_called_once_with("8306")
            mock_format.assert_called_once()

    def test_format_portfolio_backtest_result(self):
        """Test _format_portfolio_backtest_result function"""
        captured_output = StringIO()
        sys.stdout = captured_output

        _format_portfolio_backtest_result()
        
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "ポートフォリオバックテスト実行" in output
        assert "50銘柄投資システム" in output

    def test_run_portfolio_backtest_success(self):
        """Test run_portfolio_backtest function with success"""
        with patch('clstock_main._execute_portfolio_backtest') as mock_execute, \
             patch('clstock_main._format_portfolio_backtest_result') as mock_format:
            
            mock_execute.return_value = {"backtest": "completed", "result": {"return": 0.033}}
            
            result = run_portfolio_backtest()
            
            assert result["backtest"] == "completed"
            mock_execute.assert_called_once()
            mock_format.assert_called_once()

    def test_format_auto_retraining_status(self):
        """Test _format_auto_retraining_status function"""
        captured_output = StringIO()
        sys.stdout = captured_output

        status = {"status": "running"}
        
        _format_auto_retraining_status(status)
        
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "自動再学習システム状態" in output

    def test_run_auto_retraining_status_success(self):
        """Test run_auto_retraining_status function with success"""
        with patch('clstock_main._execute_auto_retraining_status') as mock_execute, \
             patch('clstock_main._format_auto_retraining_status') as mock_format:
            
            mock_execute.return_value = {"status": "running"}
            
            result = run_auto_retraining_status()
            
            assert result["status"] == "running"
            mock_execute.assert_called_once()
            mock_format.assert_called_once()