"""Tests for the process manager system."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from ClStock.systems.process_manager import (
    ProcessInfo,
    ProcessManager,
    ProcessStatus,
    settings,
)


class TestProcessManager:
    """Process manager tests."""

    def test_process_manager_initialization(self):
        """Test that process manager initializes correctly."""
        pm = ProcessManager()
        assert isinstance(pm, ProcessManager)
        assert isinstance(pm.processes, dict)
        assert len(pm.processes) > 0  # Should have default services
        assert not pm.monitoring_active

    def test_define_default_services(self):
        """Test that default services are properly defined."""
        pm = ProcessManager()

        # Check that expected services are defined
        assert "dashboard" in pm.processes
        assert "demo_trading" in pm.processes
        assert "investment_system" in pm.processes
        assert "integration_test" in pm.processes

        # Check dashboard service configuration
        dashboard = pm.processes["dashboard"]
        assert isinstance(dashboard, ProcessInfo)
        assert dashboard.name == "dashboard"
        assert dashboard.command == "python app/personal_dashboard.py"
        assert dashboard.status == ProcessStatus.STOPPED

        investment = pm.processes["investment_system"]
        assert investment.command == "python full_auto_system.py"

        deep_learning = pm.processes["deep_learning"]
        assert deep_learning.command == "python research/big_data_deep_learning.py"

        integration = pm.processes["integration_test"]
        assert integration.command == "python integration_test_enhanced.py"

        optimized = pm.processes["optimized_system"]
        assert optimized.command == "python ultra_optimized_system.py"

        selective = pm.processes["selective_system"]
        assert selective.command == "python selective_system.py"

    def test_selective_system_command_points_to_existing_file(self):
        """The selective system should point to an existing executable."""
        pm = ProcessManager()

        selective = pm.processes["selective_system"]

        # Extract the path to the executable/script from the command string.
        command_parts = selective.command.split(maxsplit=1)
        assert len(command_parts) == 2, "Command should include interpreter and script"

        script_path = Path(command_parts[1])
        assert script_path.exists(), f"Executable '{script_path}' should exist"

    def test_process_info_initialization(self):
        """Test ProcessInfo dataclass initialization."""
        process_info = ProcessInfo(name="test_service", command="python test.py")

        assert process_info.name == "test_service"
        assert process_info.command == "python test.py"
        assert process_info.status == ProcessStatus.STOPPED
        assert process_info.restart_count == 0
        assert process_info.pid is None

    @patch("ClStock.systems.process_manager.subprocess.Popen")
    def test_start_service_success(self, mock_popen):
        """Test successful service start."""
        # Mock the subprocess
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        pm = ProcessManager()

        # Try to start a service
        result = pm.start_service("dashboard")

        # Check that the service was started
        assert result is True
        assert pm.processes["dashboard"].status == ProcessStatus.RUNNING
        assert pm.processes["dashboard"].pid == 12345
        mock_popen.assert_called_once()

    @patch("ClStock.systems.process_manager.subprocess.Popen")
    def test_start_service_already_running(self, mock_popen):
        """Test starting a service that's already running."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        pm = ProcessManager()

        # Start the service once
        pm.start_service("dashboard")

        # Try to start it again
        result = pm.start_service("dashboard")

        # Should return True but not call Popen again
        assert result is True
        mock_popen.assert_called_once()  # Only called once

    @patch("ClStock.systems.process_manager.subprocess.Popen")
    def test_start_service_inherits_streams_when_logging_disabled(self, mock_popen):
        """Processes should inherit stdout/stderr when logging is disabled."""
        mock_process = Mock(pid=12345)
        mock_popen.return_value = mock_process

        pm = ProcessManager()

        with patch.object(
            settings.process,
            "log_process_output",
            False,
        ):
            pm.start_service("dashboard")

        assert mock_popen.called, "Popen should be invoked to start the process"
        _, kwargs = mock_popen.call_args

        assert kwargs.get("stdout") is None
        assert kwargs.get("stderr") is None

    @patch("ClStock.systems.process_manager.subprocess.Popen")
    def test_start_service_file_not_found(self, mock_popen):
        """Test service start failure due to file not found."""
        mock_popen.side_effect = FileNotFoundError("python not found")

        pm = ProcessManager()

        # Try to start a service
        result = pm.start_service("dashboard")

        # Check that the service failed to start
        assert result is False
        assert pm.processes["dashboard"].status == ProcessStatus.FAILED
        assert "not found" in pm.processes["dashboard"].last_error

    def test_stop_service_not_running(self):
        """Test stopping a service that's not running."""
        pm = ProcessManager()

        # Try to stop a service that's not running
        result = pm.stop_service("dashboard")

        # Should return True (no error)
        assert result is True

    @patch("ClStock.systems.process_manager.subprocess.Popen")
    def test_get_service_status(self, mock_popen):
        """Test getting service status."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        pm = ProcessManager()

        # Initially stopped
        status = pm.get_service_status("dashboard")
        assert status.status == ProcessStatus.STOPPED

        # Start the service
        pm.start_service("dashboard")

        # Now running
        status = pm.get_service_status("dashboard")
        assert status.status == ProcessStatus.RUNNING

    def test_list_services(self):
        """Test getting status of all services."""
        pm = ProcessManager()

        # Get all services status
        statuses = pm.list_services()

        # Should return a dictionary with all services
        assert isinstance(statuses, list)
        assert len(statuses) > 0
        assert "dashboard" in [s.name for s in statuses]

    def test_register_service(self):
        """Test registering a new service."""
        pm = ProcessManager()

        # Register a new service
        service_info = ProcessInfo(
            name="test_service", command="python test_service.py",
        )

        pm.register_service(service_info)

        # Check that the service was registered
        assert "test_service" in pm.processes
        assert pm.processes["test_service"] is service_info

    @patch("ClStock.systems.process_manager.subprocess.Popen")
    def test_start_service_with_quoted_arguments(self, mock_popen):
        """Service commands with quoted arguments should be parsed correctly."""
        pm = ProcessManager()
        service_info = ProcessInfo(
            name="quoted_service",
            command="python -c \"print('hello world')\"",
        )
        pm.register_service(service_info)

        mock_process = Mock(pid=6789)
        mock_popen.return_value = mock_process

        with patch.object(ProcessManager, "_check_resource_limits", return_value=True):
            assert pm.start_service("quoted_service") is True

        mock_popen.assert_called_once()
        args, _ = mock_popen.call_args
        assert args[0] == ["python", "-c", "print('hello world')"]

    @patch("ClStock.systems.process_manager.psutil")
    def test_get_system_status(self, mock_psutil):
        """Test getting system resource information."""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 25.5
        mock_virtual_memory = Mock()
        mock_virtual_memory.percent = 60.0
        mock_psutil.virtual_memory.return_value = mock_virtual_memory

        pm = ProcessManager()

        # Get system resources
        resources = pm.get_system_status()

        # Check the returned values
        assert "total_services" in resources
        assert "running" in resources
        assert "failed" in resources

    def test_default_service_scripts_exist(self):
        """Ensure default service commands reference existing scripts."""
        import shlex
        from pathlib import Path

        pm = ProcessManager()

        for service in pm.list_services():
            command_parts = shlex.split(service.command)
            assert len(command_parts) >= 2, (
                f"Command for {service.name} must include script path"
            )

            script_path = Path(service.working_dir) / command_parts[1]
            assert script_path.exists(), (
                f"Script for {service.name} not found: {script_path}"
            )

    def test_graceful_shutdown_uses_supported_executor_signature(self):
        """Ensure graceful shutdown only passes supported args to executor.shutdown."""
        pm = ProcessManager()

        # Stub out heavy dependencies
        pm.stop_all_services = MagicMock()

        executor_mock = MagicMock()
        executor_mock.shutdown = MagicMock()
        executor_mock._threads = set()  # No threads to join during shutdown
        pm._executor = executor_mock

        pm._shutdown_event.set()

        with patch("ClStock.systems.process_manager.os._exit") as mock_exit:
            pm._graceful_shutdown()

        pm.stop_all_services.assert_called_once_with(force=True)

        executor_mock.shutdown.assert_called_once()
        args, kwargs = executor_mock.shutdown.call_args
        assert args == ()
        assert kwargs == {"wait": False}

        assert not pm._shutdown_event.is_set()
        mock_exit.assert_called_once_with(0)
