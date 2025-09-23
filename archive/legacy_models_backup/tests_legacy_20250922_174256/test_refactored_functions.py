import pytest
import pandas as pd
from unittest.mock import patch, Mock
from clstock_main import (
    _format_basic_prediction_result,
    _execute_basic_prediction,
    _format_advanced_prediction_result,
    _execute_advanced_prediction,
    _format_sentiment_analysis_result,
    _execute_sentiment_analysis,
    _format_integrated_analysis_result,
    _execute_integrated_analysis,
    _format_portfolio_backtest_result,
    _execute_portfolio_backtest,
    _format_auto_retraining_status,
    _execute_auto_retraining_status
)


class TestRefactoredFunctions:
    """Refactored functions test class"""

    def test_format_basic_prediction_result(self, capsys):
        """Test basic prediction result formatting"""
        symbol = "7203"
        result = {
            "direction": 1,
            "confidence": 0.85,
            "predicted_price": 10000
        }
        
        _format_basic_prediction_result(symbol, result)
        captured = capsys.readouterr()
        
        assert "基本予測システム結果" in captured.out
        assert symbol in captured.out
        assert "上昇" in captured.out

    def test_execute_basic_prediction(self):
        """Test basic prediction execution"""
        with patch('clstock_main.TrendFollowingPredictor') as mock_predictor_class:
            mock_predictor_instance = Mock()
            mock_predictor_class.return_value = mock_predictor_instance
            mock_predictor_instance.predict_stock.return_value = {
                "direction": 1,
                "confidence": 0.85
            }
            
            result = _execute_basic_prediction("7203")
            
            assert result["direction"] == 1
            assert result["confidence"] == 0.85
            mock_predictor_instance.predict_stock.assert_called_once_with("7203")

    def test_format_advanced_prediction_result(self, capsys):
        """Test advanced prediction result formatting"""
        symbol = "6758"
        result = {"prediction": 1, "confidence": 0.9}
        
        _format_advanced_prediction_result(symbol, result)
        captured = capsys.readouterr()
        
        assert "個別銘柄特化予測結果" in captured.out
        assert symbol in captured.out

    def test_execute_advanced_prediction(self):
        """Test advanced prediction execution"""
        with patch('clstock_main.StockSpecificPredictor') as mock_predictor_class:
            mock_predictor_instance = Mock()
            mock_predictor_class.return_value = mock_predictor_instance
            mock_predictor_instance.predict_symbol.return_value = {
                "prediction": 1,
                "confidence": 0.9
            }
            
            result = _execute_advanced_prediction("6758")
            
            assert result["prediction"] == 1
            assert result["confidence"] == 0.9
            mock_predictor_instance.predict_symbol.assert_called_once_with("6758")

    def test_format_sentiment_analysis_result(self, capsys):
        """Test sentiment analysis result formatting"""
        symbol = "9984"
        sentiment = {"sentiment_score": 0.7}
        
        _format_sentiment_analysis_result(symbol, sentiment)
        captured = capsys.readouterr()
        
        assert "ニュースセンチメント分析結果" in captured.out
        assert symbol in captured.out

    def test_execute_sentiment_analysis(self):
        """Test sentiment analysis execution"""
        with patch('clstock_main.MarketSentimentAnalyzer') as mock_analyzer_class:
            mock_analyzer_instance = Mock()
            mock_analyzer_class.return_value = mock_analyzer_instance
            mock_analyzer_instance.analyze_news_sentiment.return_value = {
                "sentiment_score": 0.7
            }
            
            result = _execute_sentiment_analysis("9984")
            
            assert result["sentiment_score"] == 0.7
            mock_analyzer_instance.analyze_news_sentiment.assert_called_once_with("9984")

    def test_format_integrated_analysis_result(self, capsys):
        """Test integrated analysis result formatting"""
        symbol = "8306"
        integrated = {
            "technical": {"confidence": 0.8},
            "sentiment": {},
            "integrated": {}
        }
        
        _format_integrated_analysis_result(symbol, integrated)
        captured = capsys.readouterr()
        
        assert "統合分析結果" in captured.out
        assert symbol in captured.out
        assert "技術分析信頼度" in captured.out

    def test_execute_integrated_analysis(self):
        """Test integrated analysis execution"""
        with patch('clstock_main.TrendFollowingPredictor') as mock_trend_class, \
             patch('clstock_main.MarketSentimentAnalyzer') as mock_sentiment_class:
            
            # Mock trend predictor
            mock_trend_instance = Mock()
            mock_trend_class.return_value = mock_trend_instance
            mock_trend_instance.predict_stock.return_value = {
                "direction": 1,
                "confidence": 0.8
            }
            
            # Mock sentiment analyzer
            mock_sentiment_instance = Mock()
            mock_sentiment_class.return_value = mock_sentiment_instance
            mock_sentiment_instance.analyze_news_sentiment.return_value = {
                "sentiment_score": 0.7
            }
            mock_sentiment_instance.integrate_with_technical_analysis.return_value = {
                "integrated_signal": 1
            }
            
            result = _execute_integrated_analysis("8306")
            
            assert "technical" in result
            assert "sentiment" in result
            assert "integrated" in result
            mock_trend_instance.predict_stock.assert_called_once_with("8306")
            mock_sentiment_instance.analyze_news_sentiment.assert_called_once_with("8306")

    def test_format_portfolio_backtest_result(self, capsys):
        """Test portfolio backtest result formatting"""
        _format_portfolio_backtest_result()
        captured = capsys.readouterr()
        
        assert "ポートフォリオバックテスト実行" in captured.out
        assert "50銘柄投資システム" in captured.out

    def test_execute_portfolio_backtest(self):
        """Test portfolio backtest execution"""
        with patch('clstock_main.run_investment_system') as mock_investment_system:
            mock_investment_system.return_value = {"return": 0.033}
            
            result = _execute_portfolio_backtest()
            
            assert result["backtest"] == "completed"
            assert "result" in result

    def test_format_auto_retraining_status(self, capsys):
        """Test auto retraining status formatting"""
        status = {"status": "running"}
        
        _format_auto_retraining_status(status)
        captured = capsys.readouterr()
        
        assert "自動再学習システム状態" in captured.out

    def test_execute_auto_retraining_status(self):
        """Test auto retraining status execution"""
        with patch('clstock_main.RetrainingOrchestrator') as mock_orchestrator_class:
            mock_orchestrator_instance = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator_instance
            mock_orchestrator_instance.get_comprehensive_status.return_value = {
                "status": "running"
            }
            
            result = _execute_auto_retraining_status()
            
            assert result["status"] == "running"
            mock_orchestrator_instance.get_comprehensive_status.assert_called_once()