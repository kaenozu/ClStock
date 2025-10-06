"""Comprehensive tests for the configuration system."""

import os
from pathlib import Path
from unittest.mock import patch

from config.settings import (
    APIConfig,
    AppSettings,
    BacktestConfig,
    DatabaseConfig,
    LoggingConfig,
    ModelConfig,
    PredictionConfig,
    ProcessConfig,
    RealTimeConfig,
    TradingConfig,
    create_settings,
    get_settings,
)
from config.target_universe import get_target_universe


class TestConfigurationSystem:
    """Comprehensive configuration system tests."""

    def test_settings_singleton_pattern(self):
        """Test that get_settings implements singleton pattern correctly."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
        assert isinstance(settings1, AppSettings)

    def test_all_config_dataclasses_initialization(self):
        """Test that all configuration dataclasses initialize with correct defaults."""
        # Test each config class individually
        configs = [
            DatabaseConfig(),
            PredictionConfig(),
            ModelConfig(),
            BacktestConfig(),
            TradingConfig(),
            APIConfig(),
            LoggingConfig(),
            ProcessConfig(),
            RealTimeConfig(),
        ]

        for config in configs:
            assert config is not None

    def test_app_settings_complete_structure(self):
        """Test that AppSettings contains all required sub-configurations."""
        settings = AppSettings()

        # Verify all sub-configurations exist
        assert hasattr(settings, "database")
        assert hasattr(settings, "prediction")
        assert hasattr(settings, "model")
        assert hasattr(settings, "backtest")
        assert hasattr(settings, "trading")
        assert hasattr(settings, "api")
        assert hasattr(settings, "logging")
        assert hasattr(settings, "realtime")
        assert hasattr(settings, "process")
        assert hasattr(settings, "target_stocks")

        # Verify types
        assert isinstance(settings.database, DatabaseConfig)
        assert isinstance(settings.prediction, PredictionConfig)
        assert isinstance(settings.model, ModelConfig)
        assert isinstance(settings.backtest, BacktestConfig)
        assert isinstance(settings.trading, TradingConfig)
        assert isinstance(settings.api, APIConfig)
        assert isinstance(settings.logging, LoggingConfig)
        assert isinstance(settings.realtime, RealTimeConfig)
        assert isinstance(settings.process, ProcessConfig)
        assert isinstance(settings.target_stocks, dict)

    def test_database_config_paths(self):
        """Test that database configuration paths are correctly set."""
        config = DatabaseConfig()

        # Check that paths are pathlib.Path objects
        assert isinstance(config.personal_portfolio_db, Path)
        assert isinstance(config.prediction_history_db, Path)
        assert isinstance(config.backtest_results_db, Path)

        # Check that paths contain expected filenames
        assert "personal_portfolio.db" in str(config.personal_portfolio_db)
        assert "prediction_history.db" in str(config.prediction_history_db)
        assert "backtest_results.db" in str(config.backtest_results_db)

    def test_target_stocks_completeness(self):
        """Test that target stocks configuration is complete."""
        settings = get_settings()
        universe = get_target_universe()

        # settings.target_stocks が universe.english_names と等しいことを確認
        assert settings.target_stocks == universe.english_names

        # README記載の通り、デフォルトでは31銘柄が設定されていることを確認
        assert len(settings.target_stocks) >= 31

        # Check specific well-known stocks
        expected_stocks = {
            "7203": "Toyota Motor Corp",
            "6758": "Sony Group Corp",
            "9432": "Nippon Telegraph and Telephone",
            "8316": "Sumitomo Mitsui Financial Group",
        }

        for code, name in expected_stocks.items():
            assert code in settings.target_stocks
            assert settings.target_stocks[code] == name

    @patch.dict(
        os.environ,
        {
            "CLSTOCK_API_TITLE": "Test API Title",
            "CLSTOCK_LOG_LEVEL": "DEBUG",
            "CLSTOCK_INITIAL_CAPITAL": "2500000",
            "CLSTOCK_SCORE_THRESHOLD": "70",
        },
    )
    def test_environment_variable_loading_comprehensive(self):
        """Test comprehensive environment variable loading functionality."""
        settings = create_settings()

        assert settings.api.title == "Test API Title"
        assert settings.logging.level == "DEBUG"
        assert settings.backtest.default_initial_capital == 2500000
        assert settings.backtest.default_score_threshold == 70

    def test_environment_variable_partial_loading(self):
        """Test partial environment variable loading (some variables set, others not)."""
        defaults = create_settings(env={})

        with patch.dict(
            os.environ,
            {
                "CLSTOCK_API_TITLE": "Partial Test API",
                # Note: Not setting LOG_LEVEL, so it should remain unchanged
            },
        ):
            updated = create_settings()

            # API title should change
            assert updated.api.title == "Partial Test API"
            # Log level should remain unchanged
            assert updated.logging.level == defaults.logging.level

    def test_config_dataclass_immutability(self):
        """Test that config dataclasses maintain their structure."""
        settings = get_settings()

        # Verify that we can access nested attributes without errors
        assert hasattr(settings.api, "title")
        assert hasattr(settings.database, "personal_portfolio_db")
        assert hasattr(settings.prediction, "target_accuracy")
        assert hasattr(settings.model, "xgb_n_estimators")
        assert hasattr(settings.backtest, "default_initial_capital")
        assert hasattr(settings.trading, "market_open_hour")
        assert hasattr(settings.logging, "level")
        assert hasattr(settings.process, "max_concurrent_processes")
        assert hasattr(settings.realtime, "update_interval_seconds")

    def test_api_config_post_init_behavior(self):
        """Test APIConfig post-initialization behavior."""
        config = APIConfig()

        # Verify that available_periods is properly initialized in __post_init__
        assert config.available_periods is not None
        assert isinstance(config.available_periods, list)
        assert "1mo" in config.available_periods
        assert "3y" in config.available_periods

    def test_process_config_priorities(self):
        """Test ProcessConfig priority settings."""
        config = ProcessConfig()

        # Verify process priorities dictionary exists and has expected keys
        assert isinstance(config.process_priorities, dict)
        expected_services = [
            "dashboard",
            "demo_trading",
            "investment_system",
            "deep_learning",
        ]
        for service in expected_services:
            assert service in config.process_priorities

    def test_realtime_config_defaults(self):
        """Test RealTimeConfig default values."""
        config = RealTimeConfig()

        # Verify key realtime configuration values
        assert config.update_interval_seconds == 60
        assert config.market_open_time == "09:00"
        assert config.market_close_time == "15:00"
        assert config.pattern_confidence_threshold == 0.846
