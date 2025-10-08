import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from models.advanced.market_sentiment_analyzer import (
    MarketSentimentAnalyzer,
    SentimentData,
)
from models.advanced.risk_management_framework import (
    PortfolioRisk,
    RiskLevel,
    RiskManager,
)
from models.advanced.trading_strategy_generator import (
    AutoTradingStrategyGenerator,
    StrategyType,
    TradingStrategy,
    TradingSignal,
)
from models.ml_stock_predictor import MLStockPredictor
from models.recommendation import StockRecommendation
from data.stock_data import StockDataProvider
from models.core.interfaces import ModelConfiguration, ModelType, PredictionMode


@pytest.fixture
def mock_data_provider():
    mock = MagicMock()
    mock.get_stock_data.return_value = pd.DataFrame(
        {
            "Close": np.random.rand(100) * 100,
            "Volume": np.random.randint(1000, 10000, 100),
        },
        index=pd.to_datetime(pd.date_range(start="2023-01-01", periods=100)),
    )
    return mock


@pytest.fixture
def mock_predictor():
    mock = MagicMock()
    mock.generate_recommendation.return_value = StockRecommendation(
        symbol="TEST",
        company_name="Test Co",
        predicted_price=110.0,
        confidence=0.8,
        accuracy=0.85,
        recommendation_level="buy",
        predicted_change_percent=0.1,
        current_price=100.0,
        buy_timing="now",
        profit_target_1=110.0,
        profit_target_2=120.0,
        stop_loss=95.0,
        holding_period="1-2 months",
        recommendation_reason="Strong momentum",
    )
    return mock


@pytest.fixture
def mock_ml_predictor():
    mock = MagicMock()
    mock.predict_score.return_value = 75.0
    return mock


@pytest.fixture
def mock_sentiment_data():
    return {
        "current_sentiment": {"score": 0.6, "confidence": 0.8},
        "sentiment_score": 0.6,
        "confidence": 0.8,
    }


@pytest.fixture
def mock_price_data():
    return pd.DataFrame(
        {
            "Close": np.random.rand(100) * 100,
            "Volume": np.random.randint(1000, 10000, 100),
            "Open": np.random.rand(100) * 100,
            "High": np.random.rand(100) * 100,
            "Low": np.random.rand(100) * 100,
        },
        index=pd.to_datetime(pd.date_range(start="2023-01-01", periods=100)),
    )


class TestMarketSentimentAnalyzer:
    def test_analyze_news_sentiment(self):
        analyzer = MarketSentimentAnalyzer()
        news_texts = ["良いニュース", "悪いニュース", "中立なニュース"]
        sentiment = analyzer.news_analyzer.analyze_news_sentiment(news_texts)
        assert isinstance(sentiment, float)

    def test_analyze_social_sentiment(self):
        analyzer = MarketSentimentAnalyzer()
        social_posts = [
            {"text": "株価爆上げ🚀", "likes": 10, "retweets": 5},
            {"text": "損切りした📉", "likes": 2, "retweets": 1},
        ]
        sentiment, volume = analyzer.social_analyzer.analyze_social_sentiment(
            social_posts
        )
        assert isinstance(sentiment, float)
        assert isinstance(volume, float)

    def test_analyze_technical_sentiment(self, mock_price_data):
        analyzer = MarketSentimentAnalyzer()
        tech_sentiment = analyzer.technical_analyzer.analyze_technical_sentiment(
            mock_price_data
        )
        assert isinstance(tech_sentiment, dict)
        assert "trend_sentiment" in tech_sentiment

    def test_analyze_comprehensive_sentiment(self, mock_price_data):
        analyzer = MarketSentimentAnalyzer()
        sentiment_data = analyzer.analyze_comprehensive_sentiment(
            "TEST",
            news_data=["良いニュース"],
            social_data=[{"text": "株価爆上げ🚀"}],
            price_data=mock_price_data,
        )
        assert isinstance(sentiment_data, SentimentData)
        assert sentiment_data.symbol == "TEST"


class TestAutoTradingStrategyGenerator:
    def test_generate_momentum_strategy(self, mock_price_data):
        generator = AutoTradingStrategyGenerator()
        strategy = generator.strategy_generator.generate_momentum_strategy(
            "TEST", mock_price_data
        )
        assert isinstance(strategy, TradingStrategy)
        assert strategy.strategy_type == StrategyType.MOMENTUM

    def test_generate_trading_signals(self, mock_price_data, mock_sentiment_data):
        generator = AutoTradingStrategyGenerator()
        signals = generator.generate_trading_signals(
            "TEST", mock_price_data, mock_sentiment_data
        )
        assert isinstance(signals, list)
        if signals:
            assert isinstance(signals[0], TradingSignal)
