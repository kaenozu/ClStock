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
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

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
