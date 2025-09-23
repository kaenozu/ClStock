"""Comprehensive tests for the process manager functionality."""

import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from systems.process_manager import (
    ProcessManager, ProcessInfo, ProcessStatus, get_process_manager
)


class TestProcessManagerComprehensive:
    """Comprehensive process manager tests."""

    def test_process_manager_singleton(self):
        """Test that get_process_manager returns singleton instance."""
        pm1 = get_process_manager()
        pm2 = get_process_manager()
        assert pm1 is pm2
        assert isinstance(pm1, ProcessManager)

    def test_process_manager_initialization_with_defaults(self):
        """Test process manager initialization with default services."""
        pm = ProcessManager()
        
        # Should have default services registered
        assert len(pm.processes) > 0
        
        # Check for expected default services
        expected_services = [
            "dashboard", "demo_trading", "investment_system",
            "deep_learning", "ensemble_test", "clstock_main",
            "optimized_system", "selective_system"
        ]
        
        for service in expected_services:
            assert service in pm.processes
            assert isinstance(pm.processes[service], ProcessInfo)
            assert pm.processes[service].status == ProcessStatus.STOPPED

    def test_process_info_dataclass_comprehensive(self):
        """Test ProcessInfo dataclass with all parameters."""
        process_info = ProcessInfo(
            name="test_service",
            command="python test_service.py",
            working_dir="/test/dir",
            env_vars={"TEST_VAR": "test_value"},
            auto_restart=False,
            max_restart_attempts=5,
            restart_delay=10,
            timeout=600
        )
        
        # Verify all attributes are set correctly
        assert process_info.name == "test_service"
        assert process_info.command == "python test_service.py"
        assert process_info.working_dir == "/test/dir"
        assert process_info.env_vars == {"TEST_VAR": "test_value"}
        assert process_info.auto_restart is False
        assert process_info.max_restart_attempts == 5
        assert process_info.restart_delay == 10
        assert process_info.timeout == 600
        
        # Verify default attributes
        assert process_info.status == ProcessStatus.STOPPED
        assert process_info.pid is None
        assert process_info.process is None
        assert process_info.start_time is None
        assert process_info.restart_count == 0
        assert process_info.last_error is None

    @patch('systems.process_manager.subprocess.Popen')
    def test_service_lifecycle_complete(self, mock_popen):
        """Test complete service lifecycle: register, start, stop, restart."""
        mock_process = Mock()
        mock_process.pid = 99999
        mock_popen.return_value = mock_process
        
        pm = ProcessManager()
        
        # Register a new service
        service_info = ProcessInfo(
            name="lifecycle_test",
            command="python lifecycle_test.py"
        )
        assert pm.register_service(service_info) is True
        
        # Start the service
        assert pm.start_service("lifecycle_test") is True
        assert pm.processes["lifecycle_test"].status == ProcessStatus.RUNNING
        assert pm.processes["lifecycle_test"].pid == 99999
        
        # Stop the service
        assert pm.stop_service("lifecycle_test") is True
        assert pm.processes["lifecycle_test"].status == ProcessStatus.STOPPED
        assert pm.processes["lifecycle_test"].pid is None
        
        # Restart the service
        assert pm.restart_service("lifecycle_test") is True
        assert pm.processes["lifecycle_test"].status == ProcessStatus.RUNNING

    @patch('systems.process_manager.subprocess.Popen')
    def test_start_service_various_failure_scenarios(self, mock_popen):
        """Test service start with various failure scenarios."""
        pm = ProcessManager()
        
        # Test FileNotFoundError
        mock_popen.side_effect = FileNotFoundError("python not found")
        result = pm.start_service("dashboard")
        assert result is False
        assert pm.processes["dashboard"].status == ProcessStatus.FAILED
        assert "not found" in pm.processes["dashboard"].last_error
        
        # Test PermissionError
        mock_popen.side_effect = PermissionError("permission denied")
        result = pm.start_service("demo_trading")
        assert result is False
        assert pm.processes["demo_trading"].status == ProcessStatus.FAILED
        assert "permission" in pm.processes["demo_trading"].last_error.lower()
        
        # Test generic exception
        mock_popen.side_effect = Exception("generic error")
        result = pm.start_service("investment_system")
        assert result is False
        assert pm.processes["investment_system"].status == ProcessStatus.FAILED
        assert pm.processes["investment_system"].last_error == "generic error"

    @patch('systems.process_manager.subprocess.Popen')
    def test_stop_service_scenarios(self, mock_popen):
        """Test various service stop scenarios."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        pm = ProcessManager()
        
        # Test stopping non-existent service
        result = pm.stop_service("non_existent_service")
        assert result is False
        
        # Test stopping service that's not running
        result = pm.stop_service("dashboard")
        assert result is True  # Should succeed even if not running
        
        # Test stopping running service
        pm.start_service("dashboard")  # Start first
        assert pm.processes["dashboard"].status == ProcessStatus.RUNNING
        result = pm.stop_service("dashboard")
        assert result is True
        assert pm.processes["dashboard"].status == ProcessStatus.STOPPED

    @patch('systems.process_manager.subprocess.Popen')
    def test_stop_service_with_force(self, mock_popen):
        """Test force stopping a service."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        pm = ProcessManager()
        
        # Start a service
        pm.start_service("dashboard")
        assert pm.processes["dashboard"].status == ProcessStatus.RUNNING
        
        # Force stop the service
        result = pm.stop_service("dashboard", force=True)
        assert result is True
        assert pm.processes["dashboard"].status == ProcessStatus.STOPPED

    def test_service_status_queries(self):
        """Test various service status query methods."""
        pm = ProcessManager()
        
        # Test get_service_status for existing service
        status = pm.get_service_status("dashboard")
        assert isinstance(status, ProcessInfo)
        assert status.name == "dashboard"
        
        # Test get_service_status for non-existent service
        status = pm.get_service_status("non_existent")
        assert status is None
        
        # Test list_services
        services = pm.list_services()
        assert isinstance(services, list)
        assert len(services) > 0
        assert all(isinstance(service, ProcessInfo) for service in services)
        
        # Test get_system_status
        system_status = pm.get_system_status()
        assert isinstance(system_status, dict)
        assert "total_services" in system_status
        assert "running" in system_status
        assert "failed" in system_status
        assert "monitoring_active" in system_status
        assert system_status["total_services"] == len(pm.processes)

    @patch('systems.process_manager.subprocess.Popen')
    def test_auto_restart_functionality(self, mock_popen):
        """Test auto-restart functionality."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        pm = ProcessManager()
        
        # Register a service with auto-restart enabled
        service_info = ProcessInfo(
            name="restart_test",
            command="python restart_test.py",
            auto_restart=True,
            max_restart_attempts=3,
            restart_delay=1  # Short delay for testing
        )
        pm.register_service(service_info)
        
        # Start the service
        assert pm.start_service("restart_test") is True
        
        # Verify it's running
        assert pm.processes["restart_test"].status == ProcessStatus.RUNNING
        assert pm.processes["restart_test"].restart_count == 0

    @patch('systems.process_manager.subprocess.Popen')
    @patch('systems.process_manager.psutil.Process')
    def test_process_health_check(self, mock_psutil_process, mock_popen):
        """Test process health check functionality."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Process is still running
        mock_popen.return_value = mock_process
        
        mock_psutil_proc = Mock()
        mock_psutil_proc.memory_info.return_value.rss = 500 * 1024 * 1024  # 500MB
        mock_psutil_process.return_value = mock_psutil_proc
        
        pm = ProcessManager()
        
        # Start a service
        pm.start_service("dashboard")
        
        # Manually trigger health check (normally done by monitor thread)
        pm._check_process_health(pm.processes["dashboard"])
        
        # Process should still be running
        assert pm.processes["dashboard"].status == ProcessStatus.RUNNING

    @patch('systems.process_manager.subprocess.Popen')
    def test_stop_all_services(self, mock_popen):
        """Test stopping all services."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        pm = ProcessManager()
        
        # Start multiple services
        services_to_start = ["dashboard", "demo_trading", "investment_system"]
        for service in services_to_start:
            pm.start_service(service)
            assert pm.processes[service].status == ProcessStatus.RUNNING
        
        # Stop all services
        pm.stop_all_services()
        
        # All services should be stopped
        for service in services_to_start:
            assert pm.processes[service].status == ProcessStatus.STOPPED

    def test_register_service_edge_cases(self):
        """Test service registration edge cases."""
        pm = ProcessManager()
        
        # Test registering None (should fail gracefully)
        with pytest.raises(Exception):
            pm.register_service(None)
        
        # Test registering service with existing name (should overwrite)
        service1 = ProcessInfo(name="test_service", command="python test1.py")
        service2 = ProcessInfo(name="test_service", command="python test2.py")
        
        assert pm.register_service(service1) is True
        assert pm.register_service(service2) is True
        assert pm.processes["test_service"].command == "python test2.py"

    @patch('systems.process_manager.subprocess.Popen')
    def test_concurrent_service_operations(self, mock_popen):
        """Test concurrent service operations."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        pm = ProcessManager()
        
        # Start multiple services concurrently (simulated)
        services = ["dashboard", "demo_trading", "investment_system"]
        results = []
        
        for service in services:
            result = pm.start_service(service)
            results.append(result)
        
        # All should succeed
        assert all(results)
        
        # Check all are running
        for service in services:
            assert pm.processes[service].status == ProcessStatus.RUNNING