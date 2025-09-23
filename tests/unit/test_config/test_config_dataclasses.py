"""Tests for configuration loading and environment variable handling."""

import os
import pytest
from unittest.mock import patch

from config.settings import (
    AppSettings, DatabaseConfig, PredictionConfig, ModelConfig,
    BacktestConfig, TradingConfig, APIConfig, LoggingConfig,
    ProcessConfig, RealTimeConfig, get_settings, load_from_env
)


class TestConfigDataclasses:
    """Tests for configuration dataclasses."""

    def test_database_config_defaults(self):
        """Test DatabaseConfig default values."""
        config = DatabaseConfig()
        assert "personal_portfolio.db" in str(config.personal_portfolio_db)
        assert "prediction_history.db" in str(config.prediction_history_db)
        assert "backtest_results.db" in str(config.backtest_results_db)

    def test_prediction_config_defaults(self):
        """Test PredictionConfig default values."""
        config = PredictionConfig()
        assert config.target_accuracy == 87.0
        assert config.achieved_accuracy == 89.18
        assert config.baseline_accuracy == 84.6
        assert config.rsi_period == 14
        assert config.bollinger_period == 20

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        assert config.xgb_n_estimators == 200
        assert config.lgb_n_estimators == 200
        assert config.train_test_split == 0.8
        assert config.minimum_model_accuracy == 0.8

    def test_backtest_config_defaults(self):
        """Test BacktestConfig default values."""
        config = BacktestConfig()
        assert config.default_initial_capital == 1000000
        assert config.default_rebalance_frequency == 5
        assert config.default_data_period == "3y"

    def test_trading_config_defaults(self):
        """Test TradingConfig default values."""
        config = TradingConfig()
        assert config.market_open_hour == 9
        assert config.market_close_hour == 15
        assert config.recommendation_data_period == "6mo"

    def test_api_config_defaults(self):
        """Test APIConfig default values."""
        config = APIConfig()
        assert config.title == "ClStock API"
        assert config.version == "1.0.0"
        assert "1mo" in config.available_periods
        assert "3y" in config.available_periods

    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.log_file == "logs/clstock.log"
        assert config.max_file_size == 10 * 1024 * 1024

    def test_process_config_defaults(self):
        """Test ProcessConfig default values."""
        config = ProcessConfig()
        assert config.max_concurrent_processes == 10
        assert config.process_timeout_seconds == 300
        assert "dashboard" in config.process_priorities

    def test_realtime_config_defaults(self):
        """Test RealTimeConfig default values."""
        config = RealTimeConfig()
        assert config.update_interval_seconds == 60
        assert config.pattern_confidence_threshold == 0.846
        assert config.market_open_time == "09:00"
        assert config.market_close_time == "15:00"


class TestSettingsIntegration:
    """Integration tests for settings."""

    def test_app_settings_structure(self):
        """Test that AppSettings contains all required sub-configurations."""
        settings = AppSettings()
        
        assert isinstance(settings.database, DatabaseConfig)
        assert isinstance(settings.prediction, PredictionConfig)
        assert isinstance(settings.model, ModelConfig)
        assert isinstance(settings.backtest, BacktestConfig)
        assert isinstance(settings.trading, TradingConfig)
        assert isinstance(settings.api, APIConfig)
        assert isinstance(settings.logging, LoggingConfig)
        assert isinstance(settings.process, ProcessConfig)
        assert isinstance(settings.realtime, RealTimeConfig)
        assert isinstance(settings.target_stocks, dict)
        assert len(settings.target_stocks) > 0

    def test_get_settings_singleton(self):
        """Test that get_settings returns a singleton instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    @patch.dict(os.environ, {
        "CLSTOCK_API_TITLE": "Custom API Title",
        "CLSTOCK_LOG_LEVEL": "DEBUG",
        "CLSTOCK_INITIAL_CAPITAL": "5000000",
        "CLSTOCK_SCORE_THRESHOLD": "80"
    })
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        # Get settings instance
        settings = get_settings()
        
        # Store original values
        original_api_title = settings.api.title
        original_log_level = settings.logging.level
        original_initial_capital = settings.backtest.default_initial_capital
        original_score_threshold = settings.backtest.default_score_threshold
        
        try:
            # Load from environment
            load_from_env()
            
            # Check that environment variables were loaded
            assert settings.api.title == "Custom API Title"
            assert settings.logging.level == "DEBUG"
            assert settings.backtest.default_initial_capital == 5000000
            assert settings.backtest.default_score_threshold == 80
        finally:
            # Restore original values
            settings.api.title = original_api_title
            settings.logging.level = original_log_level
            settings.backtest.default_initial_capital = original_initial_capital
            settings.backtest.default_score_threshold = original_score_threshold

    def test_target_stocks_config(self):
        """Test that target stocks are properly configured."""
        settings = get_settings()
        
        # Check that we have the expected number of target stocks
        assert len(settings.target_stocks) >= 50  # Should have at least 50 stocks
        
        # Check a few specific stocks
        assert "7203" in settings.target_stocks  # Toyota
        assert "6758" in settings.target_stocks  # Sony
        assert settings.target_stocks["7203"] == "トヨタ自動車"
        assert settings.target_stocks["6758"] == "ソニーグループ"