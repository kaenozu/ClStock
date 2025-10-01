"""Tests for the configuration system."""

import os
from unittest.mock import patch

from config.settings import AppSettings, create_settings, get_settings


class TestConfiguration:
    """Configuration system tests."""

    def test_settings_singleton(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
        assert isinstance(settings1, AppSettings)

    def test_default_values(self):
        """Test that default values are properly set."""
        settings = get_settings()

        # Test database defaults
        assert settings.database.personal_portfolio_db.name == "personal_portfolio.db"
        assert settings.database.prediction_history_db.name == "prediction_history.db"

        # Test prediction defaults
        assert settings.prediction.target_accuracy == 87.0
        assert settings.prediction.achieved_accuracy == 89.18

        # Test model defaults
        assert settings.model.xgb_n_estimators == 200
        assert settings.model.lgb_n_estimators == 200

        # Test backtest defaults
        assert settings.backtest.default_initial_capital == 1000000
        assert settings.backtest.default_score_threshold == 60

        # Test API defaults
        assert settings.api.title == "ClStock API"
        assert "1mo" in settings.api.available_periods
        assert "3y" in settings.api.available_periods

    @patch.dict(
        os.environ,
        {
            "CLSTOCK_API_TITLE": "Test API",
            "CLSTOCK_LOG_LEVEL": "DEBUG",
            "CLSTOCK_INITIAL_CAPITAL": "2000000",
            "CLSTOCK_SCORE_THRESHOLD": "75",
        },
    )
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        settings = create_settings()

        assert settings.api.title == "Test API"
        assert settings.logging.level == "DEBUG"
        assert settings.backtest.default_initial_capital == 2000000
        assert settings.backtest.default_score_threshold == 75
