"""Core Interfaces のテスト
"""

from datetime import datetime

from models.core.interfaces import (
    BatchPredictionResult,
    ModelConfiguration,
    ModelType,
    PredictionMode,
    PredictionResult,
)


class TestPredictionResult:
    """PredictionResult のテスト"""

    def test_prediction_result_creation(self):
        """PredictionResult の基本的な作成"""
        result = PredictionResult(
            prediction=100.0,
            confidence=0.85,
            accuracy=0.90,
            timestamp=datetime.now(),
            symbol="TEST001",
        )

        assert result.prediction == 100.0
        assert result.confidence == 0.85
        assert result.accuracy == 0.90
        assert result.symbol == "TEST001"
        assert result.model_type == ModelType.ENSEMBLE  # デフォルト値
        assert result.execution_time == 0.0  # デフォルト値
        assert result.metadata == {}  # デフォルト値

    def test_prediction_result_with_all_fields(self):
        """全フィールド指定でのPredictionResult作成"""
        timestamp = datetime.now()
        metadata = {"source": "test", "version": "1.0"}

        result = PredictionResult(
            prediction=150.0,
            confidence=0.95,
            accuracy=0.88,
            timestamp=timestamp,
            symbol="TEST002",
            model_type=ModelType.HYBRID,
            execution_time=0.5,
            metadata=metadata,
        )

        assert result.prediction == 150.0
        assert result.confidence == 0.95
        assert result.accuracy == 0.88
        assert result.timestamp == timestamp
        assert result.symbol == "TEST002"
        assert result.model_type == ModelType.HYBRID
        assert result.execution_time == 0.5
        assert result.metadata == metadata

    def test_prediction_result_to_dict(self):
        """to_dict メソッドのテスト"""
        timestamp = datetime.now()
        result = PredictionResult(
            prediction=200.0,
            confidence=0.75,
            accuracy=0.82,
            timestamp=timestamp,
            symbol="TEST003",
        )

        result_dict = result.to_dict()

        assert result_dict["prediction"] == 200.0
        assert result_dict["confidence"] == 0.75
        assert result_dict["accuracy"] == 0.82
        assert result_dict["timestamp"] == timestamp.isoformat()
        assert result_dict["symbol"] == "TEST003"
        assert result_dict["model_type"] == ModelType.ENSEMBLE.value
        assert result_dict["execution_time"] == 0.0


class TestBatchPredictionResult:
    """BatchPredictionResult のテスト"""

    def test_batch_prediction_result_creation(self):
        """BatchPredictionResult の基本的な作成"""
        predictions = {"STOCK001": 100.0, "STOCK002": 200.0}
        errors = {"STOCK003": "Data not found"}

        result = BatchPredictionResult(predictions=predictions, errors=errors)

        assert result.predictions == predictions
        assert result.errors == errors
        assert result.metadata == {}  # デフォルト値

    def test_batch_prediction_result_with_metadata(self):
        """メタデータ付きBatchPredictionResult作成"""
        predictions = {"STOCK001": 150.0}
        errors = {}
        metadata = {"processing_time": 2.5, "success_rate": 1.0}

        result = BatchPredictionResult(
            predictions=predictions,
            errors=errors,
            metadata=metadata,
        )

        assert result.predictions == predictions
        assert result.errors == errors
        assert result.metadata == metadata


class TestModelConfiguration:
    """ModelConfiguration のテスト"""

    def test_model_configuration_defaults(self):
        """ModelConfiguration のデフォルト値"""
        config = ModelConfiguration()

        assert config.model_type == ModelType.ENSEMBLE
        assert config.prediction_mode == PredictionMode.BALANCED
        assert config.cache_enabled is True
        assert config.parallel_enabled is True
        assert config.max_workers == 4
        assert config.cache_size == 1000
        assert config.timeout_seconds == 300
        assert config.custom_params == {}

    def test_model_configuration_custom(self):
        """カスタムModelConfiguration"""
        custom_params = {"learning_rate": 0.01, "epochs": 100}

        config = ModelConfiguration(
            model_type=ModelType.DEEP_LEARNING,
            prediction_mode=PredictionMode.AGGRESSIVE,
            cache_enabled=False,
            parallel_enabled=False,
            max_workers=8,
            cache_size=2000,
            timeout_seconds=600,
            custom_params=custom_params,
        )

        assert config.model_type == ModelType.DEEP_LEARNING
        assert config.prediction_mode == PredictionMode.AGGRESSIVE
        assert config.cache_enabled is False
        assert config.parallel_enabled is False
        assert config.max_workers == 8
        assert config.cache_size == 2000
        assert config.timeout_seconds == 600
        assert config.custom_params == custom_params


class TestEnums:
    """Enum クラスのテスト"""

    def test_model_type_enum(self):
        """ModelType enum の値"""
        assert ModelType.ENSEMBLE.value == "ensemble"
        assert ModelType.DEEP_LEARNING.value == "deep_learning"
        assert ModelType.HYBRID.value == "hybrid"
        assert ModelType.PRECISION_87.value == "precision_87"
        assert ModelType.PARALLEL.value == "parallel"

    def test_prediction_mode_enum(self):
        """PredictionMode enum の値"""
        assert PredictionMode.CONSERVATIVE.value == "conservative"
        assert PredictionMode.BALANCED.value == "balanced"
        assert PredictionMode.AGGRESSIVE.value == "aggressive"
        assert PredictionMode.ULTRA_FAST.value == "ultra_fast"
        assert PredictionMode.HIGH_PRECISION.value == "high_precision"
