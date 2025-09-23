"""Comprehensive tests for the ClStock CLI."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
from datetime import datetime

from clstock_cli import cli
from systems.process_manager import ProcessInfo, ProcessStatus


class TestClStockCLI:
    """Comprehensive tests for the ClStock CLI."""

    def setup_method(self):
        """Setup method to initialize CLI runner."""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "ClStock 統合管理CLI" in result.output
        assert "service" in result.output
        assert "system" in result.output
        assert "data" in result.output

    def test_verbose_mode(self):
        """Test verbose mode activation."""
        with patch("clstock_cli.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = self.runner.invoke(cli, ["--verbose", "--help"])

            assert result.exit_code == 0
            # Should have called setLevel on logger
            mock_get_logger.assert_called()

    def test_service_start_without_name(self):
        """Test service start command without specifying a service name."""
        with patch("clstock_cli.get_process_manager") as mock_get_pm:
            mock_pm = Mock()
            mock_pm.list_services.return_value = [
                ProcessInfo(
                    name="test_service",
                    command="python test.py",
                    status=ProcessStatus.STOPPED,
                )
            ]
            mock_get_pm.return_value = mock_pm

            result = self.runner.invoke(cli, ["service", "start"])

            assert result.exit_code == 0
            assert "利用可能なサービス" in result.output
            assert "test_service" in result.output

    def test_service_start_with_name_success(self):
        """Test service start command with a specific service name - success case."""
        with patch("clstock_cli.get_process_manager") as mock_get_pm:
            mock_pm = Mock()
            mock_pm.start_service.return_value = True
            mock_get_pm.return_value = mock_pm

            result = self.runner.invoke(cli, ["service", "start", "test_service"])

            assert result.exit_code == 0
            assert "[成功] サービス開始: test_service" in result.output
            mock_pm.start_service.assert_called_once_with("test_service")

    def test_service_start_with_name_failure(self):
        """Test service start command with a specific service name - failure case."""
        with patch("clstock_cli.get_process_manager") as mock_get_pm:
            mock_pm = Mock()
            mock_pm.start_service.return_value = False
            mock_get_pm.return_value = mock_pm

            result = self.runner.invoke(cli, ["service", "start", "test_service"])

            assert result.exit_code == 1
            assert "[失敗] サービス開始失敗: test_service" in result.output

    def test_service_stop_without_name(self):
        """Test service stop command without specifying a service name."""
        with patch("clstock_cli.get_process_manager") as mock_get_pm, patch(
            "clstock_cli.click.confirm"
        ) as mock_confirm:

            mock_pm = Mock()
            mock_get_pm.return_value = mock_pm
            mock_confirm.return_value = True  # User confirms

            result = self.runner.invoke(cli, ["service", "stop"])

            assert result.exit_code == 0
            assert "[成功] 全サービス停止完了" in result.output
            mock_pm.stop_all_services.assert_called_once_with(force=False)

    def test_service_stop_with_name_success(self):
        """Test service stop command with a specific service name - success case."""
        with patch("clstock_cli.get_process_manager") as mock_get_pm:
            mock_pm = Mock()
            mock_pm.stop_service.return_value = True
            mock_get_pm.return_value = mock_pm

            result = self.runner.invoke(cli, ["service", "stop", "test_service"])

            assert result.exit_code == 0
            assert "[成功] サービス停止: test_service" in result.output
            mock_pm.stop_service.assert_called_once_with("test_service", force=False)

    def test_service_stop_with_force_flag(self):
        """Test service stop command with force flag."""
        with patch("clstock_cli.get_process_manager") as mock_get_pm:
            mock_pm = Mock()
            mock_pm.stop_service.return_value = True
            mock_get_pm.return_value = mock_pm

            result = self.runner.invoke(
                cli, ["service", "stop", "--force", "test_service"]
            )

            assert result.exit_code == 0
            mock_pm.stop_service.assert_called_once_with("test_service", force=True)

    def test_service_restart_success(self):
        """Test service restart command - success case."""
        with patch("clstock_cli.get_process_manager") as mock_get_pm:
            mock_pm = Mock()
            mock_pm.restart_service.return_value = True
            mock_get_pm.return_value = mock_pm

            result = self.runner.invoke(cli, ["service", "restart", "test_service"])

            assert result.exit_code == 0
            assert "[成功] サービス再起動: test_service" in result.output
            mock_pm.restart_service.assert_called_once_with("test_service")

    def test_service_restart_failure(self):
        """Test service restart command - failure case."""
        with patch("clstock_cli.get_process_manager") as mock_get_pm:
            mock_pm = Mock()
            mock_pm.restart_service.return_value = False
            mock_get_pm.return_value = mock_pm

            result = self.runner.invoke(cli, ["service", "restart", "test_service"])

            assert result.exit_code == 1
            assert "[失敗] サービス再起動失敗: test_service" in result.output

    def test_service_status_without_watch(self):
        """Test service status command without watch flag."""
        with patch("clstock_cli.get_process_manager") as mock_get_pm:
            mock_pm = Mock()
            mock_pm.get_system_status.return_value = {
                "total_services": 2,
                "running": 1,
                "failed": 0,
                "monitoring_active": True,
                "timestamp": datetime.now(),
            }
            mock_pm.list_services.return_value = [
                ProcessInfo(
                    name="running_service",
                    command="python run.py",
                    status=ProcessStatus.RUNNING,
                    pid=12345,
                    start_time=datetime.now(),
                ),
                ProcessInfo(
                    name="stopped_service",
                    command="python stop.py",
                    status=ProcessStatus.STOPPED,
                ),
            ]
            mock_get_pm.return_value = mock_pm

            result = self.runner.invoke(cli, ["service", "status"])

            assert result.exit_code == 0
            assert "[システム] ClStock システム状態" in result.output
            assert "running_service" in result.output
            assert "stopped_service" in result.output
            assert "[実行]" in result.output
            assert "[停止]" in result.output

    def test_service_monitor_toggle(self):
        """Test service monitor command to toggle monitoring."""
        with patch("clstock_cli.get_process_manager") as mock_get_pm:
            mock_pm = Mock()
            mock_pm.monitoring_active = False  # Initially inactive
            mock_get_pm.return_value = mock_pm

            result = self.runner.invoke(cli, ["service", "monitor"])

            assert result.exit_code == 0
            assert "👀 監視開始" in result.output
            mock_pm.start_monitoring.assert_called_once()

    def test_system_dashboard_success(self):
        """Test system dashboard command - success case."""
        with patch("clstock_cli.get_process_manager") as mock_get_pm:
            mock_pm = Mock()
            mock_pm.start_service.return_value = True
            mock_get_pm.return_value = mock_pm

            result = self.runner.invoke(cli, ["system", "dashboard"])

            assert result.exit_code == 0
            assert "[成功] ダッシュボード起動完了" in result.output
            assert "http://localhost:8000" in result.output
            mock_pm.start_service.assert_called_once_with("dashboard")

    def test_system_demo_success(self):
        """Test system demo command - success case."""
        with patch("clstock_cli.get_process_manager") as mock_get_pm:
            mock_pm = Mock()
            mock_pm.start_service.return_value = True
            mock_get_pm.return_value = mock_pm

            result = self.runner.invoke(cli, ["system", "demo"])

            assert result.exit_code == 0
            assert "[成功] デモ取引開始完了" in result.output
            mock_pm.start_service.assert_called_once_with("demo_trading")

    def test_system_predict_success(self):
        """Test system predict command - success case."""
        mock_result = {
            "final_prediction": 75.5,
            "final_confidence": 0.85,
            "final_accuracy": 87.0,
            "precision_87_achieved": True,
        }

        with patch("clstock_cli.Precision87BreakthroughSystem") as mock_system_class:
            mock_system = Mock()
            mock_system.predict_with_87_precision.return_value = mock_result
            mock_system_class.return_value = mock_system

            result = self.runner.invoke(cli, ["system", "predict", "--symbol", "7203"])

            assert result.exit_code == 0
            assert "💡 予測結果:" in result.output
            assert "価格予測: 75.5" in result.output
            assert "信頼度: 85.0%" in result.output
            assert "[成功] YES" in result.output

    def test_system_predict_invalid_symbol(self):
        """Test system predict command with invalid symbol."""
        result = self.runner.invoke(cli, ["system", "predict", "--symbol", "ABC"])

        assert result.exit_code == 1
        assert "[失敗] 銘柄コードは数値のみ有効です" in result.output

    def test_system_optimize_success(self):
        """Test system optimize command - success case."""
        with patch("clstock_cli.get_process_manager") as mock_get_pm:
            mock_pm = Mock()
            mock_pm.start_service.return_value = True
            mock_get_pm.return_value = mock_pm

            result = self.runner.invoke(cli, ["system", "optimize"])

            assert result.exit_code == 0
            assert "[成功] 最適化システム起動完了" in result.output
            mock_pm.start_service.assert_called_once_with("optimized_system")

    def test_data_fetch_default_symbols(self):
        """Test data fetch command with default symbols."""
        with patch("clstock_cli.StockDataProvider") as mock_provider_class:
            mock_provider = Mock()
            mock_data = Mock()
            mock_data.empty = False
            mock_data.__getitem__ = Mock(return_value=Mock())
            mock_data.__getitem__.return_value.iloc = [-100.0]  # Mock latest price
            mock_provider.get_stock_data.return_value = mock_data
            mock_provider_class.return_value = mock_provider

            result = self.runner.invoke(cli, ["data", "fetch"])

            assert result.exit_code == 0
            assert "[成功] データ取得完了" in result.output

    def test_data_fetch_custom_symbols(self):
        """Test data fetch command with custom symbols."""
        with patch("clstock_cli.StockDataProvider") as mock_provider_class:
            mock_provider = Mock()
            mock_data = Mock()
            mock_data.empty = False
            mock_data.__getitem__ = Mock(return_value=Mock())
            mock_data.__getitem__.return_value.iloc = [-100.0]  # Mock latest price
            mock_provider.get_stock_data.return_value = mock_data
            mock_provider_class.return_value = mock_provider

            result = self.runner.invoke(
                cli, ["data", "fetch", "--symbol", "1234", "--symbol", "5678"]
            )

            assert result.exit_code == 0
            assert "1234" in result.output
            assert "5678" in result.output

    def test_data_fetch_invalid_period(self):
        """Test data fetch command with invalid period."""
        result = self.runner.invoke(cli, ["data", "fetch", "--period", "invalid"])

        assert result.exit_code == 1
        assert "[失敗] 無効な期間" in result.output

    def test_data_fetch_invalid_symbol(self):
        """Test data fetch command with invalid symbol."""
        result = self.runner.invoke(cli, ["data", "fetch", "--symbol", "ABC"])

        assert result.exit_code == 1
        assert "[失敗] 無効な銘柄コード: ABC" in result.output

    def test_setup_command_success(self):
        """Test setup command - success case."""
        with patch("clstock_cli.PROJECT_ROOT") as mock_project_root:
            # Mock the directory paths
            mock_logs_dir = Mock()
            mock_logs_dir.exists.return_value = False
            mock_logs_dir.__truediv__ = Mock(return_value=mock_logs_dir)

            mock_data_dir = Mock()
            mock_data_dir.exists.return_value = False
            mock_data_dir.__truediv__ = Mock(return_value=mock_data_dir)

            mock_cache_dir = Mock()
            mock_cache_dir.exists.return_value = False
            mock_cache_dir.__truediv__ = Mock(return_value=mock_cache_dir)

            mock_project_root.__truediv__ = Mock(
                side_effect=[mock_logs_dir, mock_data_dir, mock_cache_dir]
            )

            result = self.runner.invoke(cli, ["setup"])

            assert result.exit_code == 0
            assert "[成功] セットアップ完了" in result.output

    def test_version_command(self):
        """Test version command."""
        result = self.runner.invoke(cli, ["version"])

        assert result.exit_code == 0
        assert "ClStock v1.0.0" in result.output
        assert "高精度株価予測システム" in result.output
