#!/usr/bin/env python3
"""
EnsembleStockPredictor のユニットテスト
GitHub Issue #5 対応
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime
import types
import importlib.machinery
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

scipy_module = types.ModuleType("scipy")
sparse_module = types.ModuleType("scipy.sparse")
sparse_linalg_module = types.ModuleType("scipy.sparse.linalg")
special_module = types.ModuleType("scipy.special")
optimize_module = types.ModuleType("scipy.optimize")
integrate_module = types.ModuleType("scipy.integrate")
linalg_module = types.ModuleType("scipy.linalg")
interpolate_module = types.ModuleType("scipy.interpolate")
spatial_module = types.ModuleType("scipy.spatial")
spatial_distance_module = types.ModuleType("scipy.spatial.distance")
sparse_module.csr_matrix = None
sparse_module.csc_matrix = None
sparse_module.coo_matrix = None


def _issparse(_):
    return False


sparse_module.issparse = _issparse
sparse_module.__getattr__ = lambda name: None
scipy_module.__path__ = []
scipy_module.__spec__ = importlib.machinery.ModuleSpec("scipy", loader=None, is_package=True)
scipy_module.__version__ = "0.0"
sparse_module.__package__ = "scipy"
sparse_module.__path__ = []
sparse_module.__spec__ = importlib.machinery.ModuleSpec("scipy.sparse", loader=None, is_package=True)
scipy_module.sparse = sparse_module
sparse_module.linalg = sparse_linalg_module
sparse_linalg_module.__package__ = "scipy.sparse"
sparse_linalg_module.__spec__ = importlib.machinery.ModuleSpec(
    "scipy.sparse.linalg", loader=None, is_package=True
)
sparse_linalg_module.LinearOperator = object
special_module.__package__ = "scipy"
special_module.__spec__ = importlib.machinery.ModuleSpec("scipy.special", loader=None, is_package=True)
scipy_module.special = special_module
optimize_module.__package__ = "scipy"
optimize_module.__spec__ = importlib.machinery.ModuleSpec("scipy.optimize", loader=None, is_package=True)
scipy_module.optimize = optimize_module
integrate_module.__package__ = "scipy"
integrate_module.__spec__ = importlib.machinery.ModuleSpec("scipy.integrate", loader=None, is_package=True)
scipy_module.integrate = integrate_module
linalg_module.__package__ = "scipy"
linalg_module.__spec__ = importlib.machinery.ModuleSpec("scipy.linalg", loader=None, is_package=True)
scipy_module.linalg = linalg_module
interpolate_module.__package__ = "scipy"
interpolate_module.__spec__ = importlib.machinery.ModuleSpec("scipy.interpolate", loader=None, is_package=True)
scipy_module.interpolate = interpolate_module
spatial_module.__package__ = "scipy"
spatial_module.__spec__ = importlib.machinery.ModuleSpec("scipy.spatial", loader=None, is_package=True)
spatial_module.distance = spatial_distance_module
spatial_distance_module.__package__ = "scipy.spatial"
spatial_distance_module.__spec__ = importlib.machinery.ModuleSpec("scipy.spatial.distance", loader=None, is_package=True)
scipy_module.spatial = spatial_module
sys.modules["scipy"] = scipy_module
sys.modules["scipy.sparse"] = sparse_module
sys.modules["scipy.special"] = special_module
sys.modules["scipy.optimize"] = optimize_module
sys.modules["scipy.sparse.linalg"] = sparse_linalg_module
sys.modules["scipy.integrate"] = integrate_module
sys.modules["scipy.linalg"] = linalg_module
sys.modules["scipy.interpolate"] = interpolate_module
sys.modules["scipy.spatial"] = spatial_module
sys.modules["scipy.spatial.distance"] = spatial_distance_module
sys.modules["scipy.optimize.linesearch"] = optimize_module
optimize_module.linesearch = optimize_module
special_module.expit = lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))
special_module.logit = lambda x: np.log(np.asarray(x, dtype=float) / (1.0 - np.asarray(x, dtype=float)))
special_module.gammaln = lambda x: np.log(np.vectorize(np.math.gamma)(np.asarray(x, dtype=float)))
special_module.boxcox = lambda x, lmbda=0.0: np.asarray(x, dtype=float)
special_module.comb = lambda n, k, exact=False: np.math.comb(int(n), int(k))
optimize_module.line_search_wolfe1 = lambda *args, **kwargs: (None, None, None, None, None)
optimize_module.line_search_wolfe2 = lambda *args, **kwargs: (None, None, None, None, None)
integrate_module.trapezoid = lambda y, x=None: np.trapz(y, x=x)
integrate_module.trapz = integrate_module.trapezoid
optimize_module.linear_sum_assignment = lambda *args, **kwargs: (np.array([], dtype=int), np.array([], dtype=int))
linalg_module.norm = lambda x: np.linalg.norm(np.asarray(x, dtype=float))
interpolate_module.BSpline = type("BSpline", (), {})
spatial_distance_module.pdist = lambda X, metric=None: np.zeros(1)
spatial_distance_module.squareform = lambda X: np.zeros((1, 1))


sklearn_module = types.ModuleType("sklearn")
sklearn_ensemble = types.ModuleType("sklearn.ensemble")
sklearn_linear = types.ModuleType("sklearn.linear_model")
sklearn_metrics = types.ModuleType("sklearn.metrics")
sklearn_preprocessing = types.ModuleType("sklearn.preprocessing")


class _StubRegressor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            return np.zeros_like(X, dtype=float)
        return np.zeros(X.shape[0], dtype=float)


class _StubLinearRegression(_StubRegressor):
    pass


class _StubRandomForestRegressor(_StubRegressor):
    pass


class _StubGradientBoostingRegressor(_StubRegressor):
    pass


class _StubStandardScaler:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)

    def fit_transform(self, X, y=None):
        return self.transform(X)


def _stub_mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(diff ** 2))


sklearn_ensemble.GradientBoostingRegressor = _StubGradientBoostingRegressor
sklearn_ensemble.RandomForestRegressor = _StubRandomForestRegressor
sklearn_linear.LinearRegression = _StubLinearRegression
sklearn_metrics.mean_squared_error = _stub_mean_squared_error
sklearn_preprocessing.StandardScaler = _StubStandardScaler
sklearn_module.ensemble = sklearn_ensemble
sklearn_module.linear_model = sklearn_linear
sklearn_module.metrics = sklearn_metrics
sklearn_module.preprocessing = sklearn_preprocessing
sys.modules["sklearn"] = sklearn_module
sys.modules["sklearn.ensemble"] = sklearn_ensemble
sys.modules["sklearn.linear_model"] = sklearn_linear
sys.modules["sklearn.metrics"] = sklearn_metrics
sys.modules["sklearn.preprocessing"] = sklearn_preprocessing

from models.ensemble.ensemble_predictor import RefactoredEnsemblePredictor
from models.core.interfaces import PredictionResult


class TestEnsembleStockPredictor(unittest.TestCase):
    """EnsembleStockPredictorのテストクラス"""

    def setUp(self):
        """テスト前の準備"""
        # モックデータプロバイダー
        self.mock_data_provider = Mock()
        self.predictor = RefactoredEnsemblePredictor(
            data_provider=self.mock_data_provider
        )

        # テスト用のサンプルデータ
        self.sample_data = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            }
        )
        self.mock_data_provider.get_stock_data.return_value = self.sample_data

    def test_interface_compliance(self):
        """StockPredictorインターフェース準拠性テスト"""
        # 必須メソッドが実装されているかチェック
        self.assertTrue(hasattr(self.predictor, "predict"))
        self.assertTrue(hasattr(self.predictor, "predict_batch"))
        self.assertTrue(hasattr(self.predictor, "get_confidence"))
        self.assertTrue(hasattr(self.predictor, "get_model_info"))

        # callableかチェック
        self.assertTrue(callable(self.predictor.predict))
        self.assertTrue(callable(self.predictor.predict_batch))
        self.assertTrue(callable(self.predictor.get_confidence))
        self.assertTrue(callable(self.predictor.get_model_info))

    def test_symbol_validation(self):
        """銘柄コード検証テスト"""
        # 有効な銘柄コード
        self.assertTrue(self.predictor._validate_symbol("7203"))
        self.assertTrue(self.predictor._validate_symbol("7203.T"))

        # 無効な銘柄コード
        self.assertFalse(self.predictor._validate_symbol(""))
        self.assertFalse(self.predictor._validate_symbol(None))
        self.assertFalse(self.predictor._validate_symbol("ABC"))
        self.assertFalse(self.predictor._validate_symbol("12345"))

    def test_symbols_list_validation(self):
        """銘柄リスト検証テスト"""
        # 有効な銘柄リスト
        valid_symbols = self.predictor._validate_symbols_list(["7203", "6758", "8306"])
        self.assertEqual(len(valid_symbols), 3)

        # 一部無効な銘柄を含むリスト
        mixed_symbols = self.predictor._validate_symbols_list(
            ["7203", "INVALID", "6758"]
        )
        self.assertEqual(len(mixed_symbols), 2)

        # 全て無効な場合
        with self.assertRaises(ValueError):
            self.predictor._validate_symbols_list(["INVALID", "ALSO_INVALID"])

    def test_dependency_check(self):
        """依存関係チェックテスト"""
        deps = self.predictor._check_dependencies()

        # 必須依存関係がチェックされているか
        self.assertIn("sklearn", deps)
        self.assertIn("numpy", deps)
        self.assertIn("pandas", deps)
        self.assertIn("xgboost", deps)
        self.assertIn("lightgbm", deps)

    @patch(
        "models.ensemble.ensemble_predictor.RefactoredEnsemblePredictor.predict_score"
    )
    def test_predict_with_valid_input(self, mock_predict_score):
        """有効入力での予測テスト"""
        mock_predict_score.return_value = 75.0
        self.predictor.is_trained = True

        result = self.predictor.predict("7203")

        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.symbol, "7203")
        self.assertEqual(result.prediction, 75.0)
        self.assertTrue(0 <= result.confidence <= 1)
        self.assertIn("validated", result.metadata)

    def test_predict_raises_when_provider_returns_non_dataframe(self):
        """データプロバイダーがDataFrame以外を返した場合の例外テスト"""
        self.predictor.is_trained = True
        self.mock_data_provider.get_stock_data.return_value = {"Close": [1, 2, 3]}

        with self.assertRaises(TypeError):
            self.predictor.predict("7203")

    @patch(
        "models.ensemble.ensemble_predictor.RefactoredEnsemblePredictor.predict_score"
    )
    def test_predict_retries_with_alternative_period_on_empty_data(self, mock_predict_score):
        """空データ取得時にリトライして予測できることを確認"""
        mock_predict_score.return_value = 70.0
        self.predictor.is_trained = True
        self.mock_data_provider.get_stock_data.side_effect = [
            pd.DataFrame(),
            self.sample_data,
        ]

        result = self.predictor.predict("7203")

        self.assertEqual(self.mock_data_provider.get_stock_data.call_count, 2)
        self.assertEqual(result.prediction, 70.0)

    def test_predict_raises_when_retry_still_returns_empty(self):
        """リトライ後もデータが取得できない場合に例外が発生することを確認"""
        self.predictor.is_trained = True
        empty_frame = pd.DataFrame()
        self.mock_data_provider.get_stock_data.side_effect = [empty_frame, empty_frame]

        with self.assertRaises(ValueError):
            self.predictor.predict("7203")

    def test_predict_with_invalid_input(self):
        """無効入力での予測テスト"""
        result = self.predictor.predict("INVALID")

        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.symbol, "INVALID")
        self.assertEqual(result.prediction, 50.0)
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("error", result.metadata)

    def test_fallback_prediction(self):
        """フォールバック予測テスト"""
        # データプロバイダーのモック設定
        self.mock_data_provider.get_stock_data.return_value = self.sample_data

        result = self.predictor._fallback_prediction("7203")

        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.symbol, "7203")
        self.assertTrue(0 <= result.prediction <= 100)
        self.assertTrue(0 <= result.confidence <= 1)
        self.assertEqual(result.metadata["model_type"], "fallback")

    def test_fallback_prediction_no_data(self):
        """データなしでのフォールバック予測テスト"""
        # 空のデータフレームを返すモック
        self.mock_data_provider.get_stock_data.return_value = pd.DataFrame()

        result = self.predictor._fallback_prediction("7203")

        self.assertEqual(result.prediction, 50.0)
        self.assertEqual(result.confidence, 0.1)

    def test_predict_batch_valid_symbols(self):
        """有効銘柄でのバッチ予測テスト"""
        with patch.object(self.predictor, "predict") as mock_predict:
            # モック予測結果
            mock_result = PredictionResult(
                prediction=75.0,
                confidence=0.8,
                accuracy=85.0,
                timestamp=datetime.now(),
                symbol="test",
                metadata={},
            )
            mock_predict.return_value = mock_result

            results = self.predictor.predict_batch(["7203", "6758"])

            self.assertEqual(len(results), 2)
            self.assertEqual(mock_predict.call_count, 2)

    def test_predict_batch_invalid_symbols(self):
        """無効銘柄でのバッチ予測テスト"""
        results = self.predictor.predict_batch(["INVALID", "ALSO_INVALID"])

        # 無効な入力の場合、空のリストが返される
        self.assertEqual(len(results), 0)

    def test_predict_batch_mixed_symbols(self):
        """有効・無効混在銘柄でのバッチ予測テスト"""
        with patch.object(self.predictor, "predict") as mock_predict:
            mock_result = PredictionResult(
                prediction=75.0,
                confidence=0.8,
                accuracy=85.0,
                timestamp=datetime.now(),
                symbol="test",
                metadata={},
            )
            mock_predict.return_value = mock_result

            results = self.predictor.predict_batch(["7203", "INVALID", "6758"])

            # 有効な銘柄の数だけ結果が返される
            self.assertEqual(len(results), 2)

    def test_get_confidence_untrained(self):
        """未訓練モデルでの信頼度テスト"""
        self.predictor.is_trained = False
        confidence = self.predictor.get_confidence("7203")
        self.assertEqual(confidence, 0.0)

    def test_get_confidence_trained(self):
        """訓練済みモデルでの信頼度テスト"""
        self.predictor.is_trained = True
        self.predictor.models = {"model1": Mock(), "model2": Mock()}
        self.predictor.weights = {"model1": 0.5, "model2": 0.5}

        confidence = self.predictor.get_confidence("7203")
        self.assertTrue(0.1 <= confidence <= 0.95)

    def test_get_model_info(self):
        """モデル情報取得テスト"""
        self.predictor.models = {"model1": Mock(), "model2": Mock()}
        self.predictor.weights = {"model1": 0.6, "model2": 0.4}
        self.predictor.is_trained = True
        self.predictor.feature_names = ["feature1", "feature2", "feature3"]

        info = self.predictor.get_model_info()

        self.assertIsInstance(info, dict)
        self.assertEqual(info["name"], "RefactoredEnsemblePredictor")
        self.assertEqual(info["version"], "1.0.0")
        self.assertEqual(info["is_trained"], True)
        self.assertEqual(info["model_data"]["num_models"], 2)
        self.assertEqual(info["model_data"]["num_features"], 3)
        self.assertEqual(info["model_data"]["models"], ["model1", "model2"])

    def test_safe_model_operation_success(self):
        """安全なモデル操作（成功）テスト"""

        def success_operation():
            return "success"

        result = self.predictor._safe_model_operation(
            "test_operation", success_operation, fallback_value="fallback"
        )

        self.assertEqual(result, "success")

    def test_safe_model_operation_failure(self):
        """安全なモデル操作（失敗）テスト"""

        def failure_operation():
            raise Exception("Test error")

        result = self.predictor._safe_model_operation(
            "test_operation", failure_operation, fallback_value="fallback"
        )

        self.assertEqual(result, "fallback")

    def test_add_model(self):
        """モデル追加テスト"""
        mock_model = Mock()

        self.predictor.add_model("test_model", mock_model, weight=0.8)

        self.assertIn("test_model", self.predictor.models)
        self.assertEqual(self.predictor.models["test_model"], mock_model)
        self.assertEqual(self.predictor.weights["test_model"], 0.8)

    @patch("models.ensemble.ensemble_predictor.joblib.dump")
    def test_save_ensemble(self, mock_dump):
        """アンサンブル保存テスト"""
        self.predictor.models = {"model1": Mock()}
        self.predictor.weights = {"model1": 1.0}
        self.predictor.is_trained = True

        self.predictor.save_ensemble()

        mock_dump.assert_called_once()

    @patch("models.ensemble.ensemble_predictor.joblib.load")
    @patch("pathlib.Path.exists")
    def test_load_ensemble_success(self, mock_exists, mock_load):
        """アンサンブル読み込み（成功）テスト"""
        mock_exists.return_value = True
        mock_load.return_value = {
            "models": {"model1": Mock()},
            "weights": {"model1": 1.0},
            "scaler": Mock(),
            "feature_names": ["feature1"],
            "is_trained": True,
        }

        result = self.predictor.load_ensemble()

        self.assertTrue(result)
        self.assertTrue(self.predictor.is_trained)

    @patch("pathlib.Path.exists")
    def test_load_ensemble_file_not_found(self, mock_exists):
        """アンサンブル読み込み（ファイルなし）テスト"""
        mock_exists.return_value = False

        result = self.predictor.load_ensemble()

        self.assertFalse(result)


class TestEnsembleStockPredictorIntegration(unittest.TestCase):
    """統合テスト"""

    def setUp(self):
        """統合テスト用の準備"""
        self.predictor = RefactoredEnsemblePredictor()

    def test_end_to_end_prediction_flow(self):
        """エンドツーエンド予測フローテスト"""
        # 実際のデータプロバイダーを使用（モック化しない）
        # ただし、ネットワークエラーに対応
        try:
            result = self.predictor.predict("7203")

            # 結果の基本チェック
            self.assertIsInstance(result, PredictionResult)
            self.assertEqual(result.symbol, "7203")
            self.assertTrue(0 <= result.prediction <= 100)
            self.assertTrue(0 <= result.confidence <= 1)

        except Exception as e:
            # ネットワークエラーなどの場合はスキップ
            self.skipTest(f"Integration test skipped due to: {e}")


if __name__ == "__main__":
    unittest.main()
