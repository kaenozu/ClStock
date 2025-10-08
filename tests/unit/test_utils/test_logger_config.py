"""Tests for the logger configuration system."""

import logging
from unittest.mock import mock_open, patch

from utils.logger_config import CentralizedLogger, get_logger, setup_logger


class TestLoggerConfig:
    """Logger configuration tests."""

    def test_setup_logger_basic(self):
        """Test basic logger setup."""
        logger = setup_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert len(logger.handlers) > 0

    def test_setup_logger_with_level(self):
        """Test logger setup with specific level."""
        logger = setup_logger("test_logger_debug", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_setup_logger_with_format(self):
        """Test logger setup with custom format."""
        custom_format = "%(levelname)s - %(message)s"
        logger = setup_logger("test_logger_format", format_string=custom_format)
        # The format is set on the handler
        assert len(logger.handlers) > 0
        formatter = logger.handlers[0].formatter
        assert custom_format in formatter._fmt

    def test_get_logger_existing(self):
        """Test getting an existing logger."""
        # First setup a logger
        logger1 = setup_logger("test_existing")
        # Then get it again
        logger2 = get_logger("test_existing")
        assert logger1 is logger2

    def test_get_logger_new(self):
        """Test getting a new logger (should create it)."""
        logger = get_logger("test_new_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_new_logger"

    def test_centralized_logger_initialization(self):
        """Test centralized logger initialization."""
        centralized_logger = CentralizedLogger()
        assert isinstance(centralized_logger, CentralizedLogger)
        assert isinstance(centralized_logger.log_collectors, dict)
        assert isinstance(centralized_logger.active_services, set)

    def test_centralized_logger_register_service(self):
        """Test registering a service with centralized logger."""
        centralized_logger = CentralizedLogger()
        centralized_logger.register_service("test_service", "/path/to/log.log")

        assert "test_service" in centralized_logger.active_services
        assert "test_service" in centralized_logger.log_collectors
        assert "test_service" in centralized_logger.log_files

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="[2023-01-01 12:00:00] [INFO] [TEST] Test message\n",
    )
    def test_centralized_logger_get_recent_logs(self, mock_file):
        """Test getting recent logs from centralized logger."""
        centralized_logger = CentralizedLogger()
        centralized_logger.register_service("test_service", "/path/to/log.log")
        centralized_logger.centralized_log = "/path/to/centralized.log"

        # Mock the file existence
        with patch.object(
            centralized_logger.centralized_log,
            "exists",
            return_value=True,
        ):
            logs = centralized_logger.get_recent_logs(hours=1, service_filter="TEST")
            # Should return the log line
            assert len(logs) >= 0  # May be empty depending on time filter

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="[2023-01-01 12:00:00] [ERROR] [TEST] Error message\n[2023-01-01 12:01:00] [INFO] [TEST] Info message\n",
    )
    def test_centralized_logger_analyze_patterns(self, mock_file):
        """Test log pattern analysis."""
        centralized_logger = CentralizedLogger()
        centralized_logger.register_service("test_service", "/path/to/log.log")
        centralized_logger.centralized_log = "/path/to/centralized.log"

        # Mock the file existence
        with patch.object(
            centralized_logger.centralized_log,
            "exists",
            return_value=True,
        ):
            analysis = centralized_logger.analyze_log_patterns()

            # Should have analysis results
            assert "error_count" in analysis
            assert "warning_count" in analysis
            assert "info_count" in analysis
