"""Comprehensive tests for the performance monitor components."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
from models.core.interfaces import PredictionResult
from models.monitoring.performance_monitor import ModelPerformanceMonitor


class TestModelPerformanceMonitor:
    """Comprehensive tests for ModelPerformanceMonitor."""

    def test_initialization_default(self):
        """Test performance monitor initialization with default parameters."""
        monitor = ModelPerformanceMonitor()

        # Check that collections are initialized
        assert isinstance(monitor.performance_history, list)
        assert isinstance(monitor.alerts, list)
        assert isinstance(monitor.prediction_records, list)
        assert len(monitor.performance_history) == 0
        assert len(monitor.alerts) == 0
        assert len(monitor.prediction_records) == 0

        # Check default thresholds
        expected_thresholds = {
            "rmse": 15.0,
            "r2": 0.1,
            "direction_accuracy": 0.55,
            "mae": 10.0,
        }
        assert monitor.thresholds == expected_thresholds

    def test_initialization_custom_file(self):
        """Test performance monitor initialization with custom data file."""
        custom_file = "test/custom_performance.json"
        monitor = ModelPerformanceMonitor(data_file=custom_file)

        assert str(monitor.data_file) == custom_file

    def test_record_prediction(self):
        """Test recording a prediction result."""
        monitor = ModelPerformanceMonitor()

        # Create a sample prediction result
        result = PredictionResult(
            prediction=75.0,
            confidence=0.85,
            accuracy=0.92,
            timestamp=datetime.now(),
            symbol="AAPL",
            metadata={"model": "test_model", "version": "1.0"},
        )

        # Record the prediction
        monitor.record_prediction(result)

        # Check that it was recorded
        assert len(monitor.prediction_records) == 1
        recorded = monitor.prediction_records[0]

        # Check that all fields were recorded correctly
        assert recorded["prediction"] == 75.0
        assert recorded["confidence"] == 0.85
        assert recorded["accuracy"] == 0.92
        assert recorded["symbol"] == "AAPL"
        assert recorded["metadata"] == {"model": "test_model", "version": "1.0"}
        # Check that timestamp is in ISO format
        assert isinstance(recorded["timestamp"], str)

    def test_get_accuracy_metrics_empty(self):
        """Test getting accuracy metrics with no data."""
        monitor = ModelPerformanceMonitor()

        metrics = monitor.get_accuracy_metrics()

        # Should return basic metrics even with no data
        assert "count" in metrics
        assert metrics["count"] == 0

    def test_get_accuracy_metrics_with_data(self):
        """Test getting accuracy metrics with data."""
        monitor = ModelPerformanceMonitor()

        # Add some sample predictions
        base_time = datetime.now()
        for i in range(5):
            result = PredictionResult(
                prediction=70.0 + i,
                confidence=0.8 + i * 0.02,
                accuracy=0.85 + i * 0.01,
                timestamp=base_time - timedelta(days=i),
                symbol="AAPL",
                metadata={},
            )
            monitor.record_prediction(result)

        # Get metrics for the last 10 days (should include all)
        metrics = monitor.get_accuracy_metrics(period_days=10)

        # Check that metrics are calculated correctly
        assert metrics["count"] == 5
        assert metrics["avg_accuracy"] == 0.87
        assert metrics["avg_confidence"] == 0.84
        assert metrics["avg_prediction"] == 72.0

    def test_get_accuracy_metrics_with_filtering(self):
        """Test getting accuracy metrics with time filtering."""
        monitor = ModelPerformanceMonitor()

        # Add some old and recent predictions
        base_time = datetime.now()
        old_time = base_time - timedelta(days=45)  # 45 days ago
        recent_time = base_time - timedelta(days=5)  # 5 days ago

        # Add old prediction
        old_result = PredictionResult(
            prediction=60.0,
            confidence=0.7,
            accuracy=0.8,
            timestamp=old_time,
            symbol="AAPL",
            metadata={},
        )
        monitor.record_prediction(old_result)

        # Add recent prediction
        recent_result = PredictionResult(
            prediction=80.0,
            confidence=0.9,
            accuracy=0.95,
            timestamp=recent_time,
            symbol="AAPL",
            metadata={},
        )
        monitor.record_prediction(recent_result)

        # Get metrics for the last 30 days (should only include recent)
        metrics = monitor.get_accuracy_metrics(period_days=30)

        assert metrics["count"] == 1
        assert metrics["avg_accuracy"] == 0.95
        assert metrics["avg_confidence"] == 0.9

    def test_get_performance_report_empty(self):
        """Test getting performance report with no data."""
        monitor = ModelPerformanceMonitor()

        report = monitor.get_performance_report()

        # Should return a basic report even with no data
        assert report["status"] == "No performance data available"
        assert "summary" in report
        assert "alerts" in report
        assert report["alerts"] == []

    def test_evaluate_model_performance(self):
        """Test evaluating model performance."""
        monitor = ModelPerformanceMonitor()

        # Create mock test data
        X_test = np.array([[1, 2], [3, 4], [5, 6]])
        y_test = np.array([10, 20, 30])
        predictions = np.array([11, 19, 31])  # Close to actual values

        # Create a mock model
        mock_model = Mock()
        mock_model.predict.return_value = predictions

        # Evaluate performance
        with patch(
            "models.monitoring.performance_monitor.mean_squared_error",
        ) as mock_mse, patch(
            "models.monitoring.performance_monitor.mean_absolute_error",
        ) as mock_mae, patch(
            "models.monitoring.performance_monitor.r2_score",
        ) as mock_r2, patch(
            "models.monitoring.performance_monitor.accuracy_score",
        ) as mock_accuracy:
            # Mock the sklearn functions to return specific values
            mock_mse.return_value = 1.0
            mock_mae.return_value = 0.67
            mock_r2.return_value = 0.99
            mock_accuracy.return_value = 0.67

            result = monitor.evaluate_model_performance(
                mock_model,
                X_test,
                y_test,
                model_name="TestModel",
            )

        # Check that performance was recorded
        assert len(monitor.performance_history) == 1
        recorded = monitor.performance_history[0]

        assert recorded["model_name"] == "TestModel"
        assert recorded["mse"] == 1.0
        assert recorded["rmse"] == 1.0  # sqrt(1.0)
        assert recorded["mae"] == 0.67
        assert recorded["r2_score"] == 0.99
        assert recorded["direction_accuracy"] == 0.67
        assert recorded["sample_size"] == 3

    def test_evaluate_model_performance_with_alerts(self):
        """Test evaluating model performance that triggers alerts."""
        monitor = ModelPerformanceMonitor()

        # Create test data that will trigger alerts
        X_test = np.array([[1, 2], [3, 4], [5, 6]])
        y_test = np.array([10, 20, 30])
        predictions = np.array([50, 60, 70])  # Very different from actual values

        # Create a mock model
        mock_model = Mock()
        mock_model.predict.return_value = predictions

        # Evaluate performance with poor results that will trigger alerts
        with patch(
            "models.monitoring.performance_monitor.mean_squared_error",
        ) as mock_mse, patch(
            "models.monitoring.performance_monitor.mean_absolute_error",
        ) as mock_mae, patch(
            "models.monitoring.performance_monitor.r2_score",
        ) as mock_r2, patch(
            "models.monitoring.performance_monitor.accuracy_score",
        ) as mock_accuracy:
            # Mock the sklearn functions to return poor values that will trigger alerts
            mock_mse.return_value = 25.0  # High MSE
            mock_mae.return_value = 15.0  # High MAE
            mock_r2.return_value = 0.05  # Low R2
            mock_accuracy.return_value = 0.33  # Low accuracy

            result = monitor.evaluate_model_performance(
                mock_model,
                X_test,
                y_test,
                model_name="PoorModel",
            )

        # Check that alerts were generated
        assert len(monitor.alerts) > 0
        alert = monitor.alerts[0]

        assert alert["model_name"] == "PoorModel"
        assert len(alert["alerts"]) > 0
        assert "High RMSE" in alert["alerts"][0] or "Low R²" in alert["alerts"][0]

    def test_calculate_trend(self):
        """Test trend calculation."""
        monitor = ModelPerformanceMonitor()

        # Test with insufficient data
        trend = monitor._calculate_trend([1.0])
        assert trend == "insufficient_data"

        # Test with improving trend
        improving_values = [10.0, 8.0, 6.0, 5.0, 4.0, 3.0]
        trend = monitor._calculate_trend(improving_values)
        assert trend == "improving"

        # Test with declining trend
        declining_values = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        trend = monitor._calculate_trend(declining_values)
        assert trend == "declining"

        # Test with stable trend
        stable_values = [5.0, 5.1, 4.9, 5.05, 4.95, 5.0]
        trend = monitor._calculate_trend(stable_values)
        assert trend == "stable"

    def test_calculate_performance_grade(self):
        """Test performance grade calculation."""
        monitor = ModelPerformanceMonitor()

        # Test excellent performance
        excellent_metrics = {
            "rmse": 3.0,  # Very low
            "r2_score": 0.8,  # Very high
            "direction_accuracy": 0.8,  # Very high
        }
        grade = monitor._calculate_performance_grade(excellent_metrics)
        assert grade in ["A+", "A"]  # Should be high grade

        # Test poor performance
        poor_metrics = {
            "rmse": 20.0,  # Very high
            "r2_score": 0.05,  # Very low
            "direction_accuracy": 0.4,  # Very low
        }
        grade = monitor._calculate_performance_grade(poor_metrics)
        assert grade in ["C", "D"]  # Should be low grade

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        monitor = ModelPerformanceMonitor()

        # Test with poor metrics that trigger recommendations
        poor_metrics = {
            "rmse": 20.0,  # High
            "r2_score": 0.05,  # Low
            "direction_accuracy": 0.4,  # Low
        }

        recommendations = monitor._generate_recommendations(
            poor_metrics,
            "declining",
            "declining",
        )

        # Should have multiple recommendations
        assert len(recommendations) > 1
        # Should include specific recommendations
        assert any(
            "モデルの正則化パラメータを調整してください" in rec
            for rec in recommendations
        )
        assert any(
            "特徴量エンジニアリングの改善を検討してください" in rec
            for rec in recommendations
        )
        assert any("モデルの再訓練が必要です" in rec for rec in recommendations)

    def test_save_performance_data(self):
        """Test saving performance data to file."""
        monitor = ModelPerformanceMonitor()

        # Add some sample data
        monitor.performance_history = [{"test": "performance"}]
        monitor.alerts = [{"test": "alert"}]
        monitor.prediction_records = [{"test": "record"}]

        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(
            mode="w+",
            delete=False,
            suffix=".json",
        ) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Override the data file path
            monitor.data_file = temp_path

            # Save data
            monitor.save_performance_data()

            # Check that file was created
            assert os.path.exists(temp_path)

            # Check that data was saved correctly
            with open(temp_path) as f:
                saved_data = json.load(f)

            assert "performance_history" in saved_data
            assert "alerts" in saved_data
            assert "prediction_records" in saved_data
            assert saved_data["performance_history"] == [{"test": "performance"}]
            assert saved_data["alerts"] == [{"test": "alert"}]
            assert saved_data["prediction_records"] == [{"test": "record"}]

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_performance_data(self):
        """Test loading performance data from file."""
        monitor = ModelPerformanceMonitor()

        # Create test data
        test_data = {
            "performance_history": [{"test": "loaded_performance"}],
            "alerts": [{"test": "loaded_alert"}],
            "prediction_records": [{"test": "loaded_record"}],
            "thresholds": {"test_threshold": 1.0},
        }

        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(
            mode="w+",
            delete=False,
            suffix=".json",
        ) as tmp_file:
            temp_path = tmp_file.name
            json.dump(test_data, tmp_file)

        try:
            # Override the data file path
            monitor.data_file = temp_path

            # Load data
            success = monitor.load_performance_data()

            assert success is True
            assert monitor.performance_history == [{"test": "loaded_performance"}]
            assert monitor.alerts == [{"test": "loaded_alert"}]
            assert monitor.prediction_records == [{"test": "loaded_record"}]
            assert monitor.thresholds["test_threshold"] == 1.0

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_performance_data_nonexistent_file(self):
        """Test loading performance data from nonexistent file."""
        monitor = ModelPerformanceMonitor()

        # Set to a nonexistent file
        monitor.data_file = "nonexistent_file.json"

        # Should return False and not raise exception
        success = monitor.load_performance_data()
        assert success is False

    def test_get_performance_summary(self):
        """Test getting performance summary."""
        monitor = ModelPerformanceMonitor()

        # Test with no data
        summary = monitor.get_performance_summary()
        assert "No performance data available" in summary

        # Add some performance data
        test_performance = {
            "timestamp": datetime.now().isoformat(),
            "model_name": "TestModel",
            "mse": 1.0,
            "rmse": 1.0,
            "mae": 0.5,
            "r2_score": 0.95,
            "direction_accuracy": 0.85,
            "sample_size": 100,
        }
        monitor.performance_history = [test_performance] * 5

        # Test with data
        summary = monitor.get_performance_summary(last_n_records=3)

        # Should contain performance metrics
        assert "Performance Summary" in summary
        assert "Average RMSE" in summary
        assert "Average R²" in summary
        assert "Average Direction Accuracy" in summary
