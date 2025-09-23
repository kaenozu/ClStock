"""Tests for the process manager system."""

import pytest
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from systems.process_manager import (
    ProcessManager, ProcessInfo, ProcessStatus
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
        
        # Check dashboard service configuration
        dashboard = pm.processes["dashboard"]
        assert isinstance(dashboard, ProcessInfo)
        assert dashboard.name == "dashboard"
        assert dashboard.command == "python app/personal_dashboard.py"
        assert dashboard.status == ProcessStatus.STOPPED

    def test_process_info_initialization(self):
        """Test ProcessInfo dataclass initialization."""
        process_info = ProcessInfo(
            name="test_service",
            command="python test.py"
        )
        
        assert process_info.name == "test_service"
        assert process_info.command == "python test.py"
        assert process_info.status == ProcessStatus.STOPPED
        assert process_info.restart_count == 0
        assert process_info.pid is None

    @patch('systems.process_manager.subprocess.Popen')
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

    @patch('systems.process_manager.subprocess.Popen')
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

    @patch('systems.process_manager.subprocess.Popen')
    def test_start_service_file_not_found(self, mock_popen):
        """Test service start failure due to file not found."""
        from FileNotFoundError import FileNotFoundError
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

    @patch('systems.process_manager.subprocess.Popen')
    def test_get_service_status(self, mock_popen):
        """Test getting service status."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        pm = ProcessManager()
        
        # Initially stopped
        status = pm.get_service_status("dashboard")
        assert status == ProcessStatus.STOPPED
        
        # Start the service
        pm.start_service("dashboard")
        
        # Now running
        status = pm.get_service_status("dashboard")
        assert status == ProcessStatus.RUNNING

    def test_get_all_services_status(self):
        """Test getting status of all services."""
        pm = ProcessManager()
        
        # Get all services status
        statuses = pm.get_all_services_status()
        
        # Should return a dictionary with all services
        assert isinstance(statuses, dict)
        assert len(statuses) > 0
        assert "dashboard" in statuses
        assert statuses["dashboard"] == ProcessStatus.STOPPED

    def test_register_service(self):
        """Test registering a new service."""
        pm = ProcessManager()
        
        # Register a new service
        service_info = ProcessInfo(
            name="test_service",
            command="python test_service.py"
        )
        
        pm.register_service(service_info)
        
        # Check that the service was registered
        assert "test_service" in pm.processes
        assert pm.processes["test_service"] is service_info

    @patch('systems.process_manager.psutil')
    def test_get_system_resources(self, mock_psutil):
        """Test getting system resource information."""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 25.5
        mock_virtual_memory = Mock()
        mock_virtual_memory.percent = 60.0
        mock_psutil.virtual_memory.return_value = mock_virtual_memory
        
        pm = ProcessManager()
        
        # Get system resources
        resources = pm.get_system_resources()
        
        # Check the returned values
        assert "cpu_percent" in resources
        assert "memory_percent" in resources
        assert resources["cpu_percent"] == 25.5
        assert resources["memory_percent"] == 60.0