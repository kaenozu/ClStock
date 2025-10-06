"""Integration tests for the optimization history menu workflow."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from menu import optimization_history_menu, rollback_to_record, show_history_list


class TestMenuIntegration:
    """Integration-level expectations for exposed menu functions."""

    def test_menu_functions_exist(self):
        """Ensure the exported menu functions are callable."""
        assert callable(optimization_history_menu)
        assert callable(show_history_list)
        assert callable(rollback_to_record)

    def test_optimization_history_menu_exit(self):
        """Selecting option 0 exits without invoking other handlers."""
        with patch("menu.clear_screen"), patch(
            "builtins.input", return_value="0",
        ) as mock_input:
            with patch("menu.show_history_list") as mock_show, patch(
                "menu.rollback_to_record",
            ) as mock_rollback, patch("menu.show_history_statistics") as mock_stats:
                optimization_history_menu()

        mock_input.assert_called_once()
        mock_show.assert_not_called()
        mock_rollback.assert_not_called()
        mock_stats.assert_not_called()

    def test_optimization_history_menu_list_flow(self):
        """Selecting option 1 invokes show_history_list."""
        with patch("menu.clear_screen"), patch("builtins.input", return_value="1"):
            with patch("menu.show_history_list") as mock_show, patch(
                "menu.rollback_to_record",
            ), patch("menu.show_history_statistics"):
                optimization_history_menu()

        mock_show.assert_called_once_with()

    def test_optimization_history_menu_rollback_flow(self):
        """Selecting option 2 requests record id and calls rollback."""
        with patch("menu.clear_screen"), patch(
            "builtins.input", side_effect=["2", "target-record"],
        ):
            with patch("menu.rollback_to_record") as mock_rollback, patch(
                "menu.show_history_list",
            ), patch("menu.show_history_statistics"):
                optimization_history_menu()

        mock_rollback.assert_called_once_with("target-record")

    def test_show_history_list_basic(self):
        """show_history_list retrieves records via the manager."""
        manager = Mock()
        manager.list_optimization_records.return_value = []

        stub_module = SimpleNamespace(OptimizationHistoryManager=lambda: manager)

        with patch("builtins.input", return_value=""), patch.dict(
            sys.modules, {"systems.optimization_history": stub_module},
        ):
            show_history_list()

        manager.list_optimization_records.assert_called_once()

    def test_rollback_to_record_basic(self):
        """rollback_to_record delegates to the manager."""
        manager = Mock()
        manager.rollback_to_configuration.return_value = True

        stub_module = SimpleNamespace(OptimizationHistoryManager=lambda: manager)

        with patch("builtins.input", return_value=""), patch.dict(
            sys.modules, {"systems.optimization_history": stub_module},
        ):
            rollback_to_record("record-1")

        manager.rollback_to_configuration.assert_called_once_with("record-1")


class TestMenuWorkflow:
    """Workflow-style integration scenarios."""

    def test_complete_workflow_simulation(self):
        """Simulate showing history followed by a rollback and exit."""
        with patch("menu.clear_screen"), patch(
            "builtins.input", side_effect=["1", "2", "target", "0"],
        ) as mock_input:
            with patch("menu.show_history_list") as mock_show, patch(
                "menu.rollback_to_record",
            ) as mock_rollback, patch("menu.show_history_statistics"):
                optimization_history_menu()
                optimization_history_menu()
                optimization_history_menu()

        mock_show.assert_called_once()
        mock_rollback.assert_called_once_with("target")
        assert mock_input.call_count == 4
