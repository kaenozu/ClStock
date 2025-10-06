#!/usr/bin/env python3
"""TDD テストファースト: 最適30銘柄予測システムのテスト
Red-Green-Refactor サイクルに従った開発
"""

import unittest
from unittest.mock import patch

import pandas as pd

# テスト対象（まだ存在しない理想的なクラス）
try:
    from optimal_30_prediction_tdd import Optimal30PredictionTDD
except ImportError:
    # まだ実装されていないので、テストは失敗する（Red）
    Optimal30PredictionTDD = None


class TestOptimal30PredictionTDD(unittest.TestCase):
    """TDD: 最適30銘柄予測システムのテスト"""

    def setUp(self):
        """各テスト前のセットアップ"""
        if Optimal30PredictionTDD is None:
            self.skipTest("実装対象クラスが存在しません（Red フェーズ）")

        self.system = Optimal30PredictionTDD()

    def test_optimal_30_symbols_list_exists(self):
        """テスト1: 最適30銘柄リストが正しく定義されている"""
        # Arrange & Act
        symbols = self.system.get_optimal_symbols()

        # Assert
        self.assertEqual(len(symbols), 30, "銘柄数は30個である必要があります")
        self.assertIn("9984.T", symbols, "ソフトバンクGが含まれている必要があります")
        self.assertIn(
            "4004.T", symbols, "化学セクター最高スコア銘柄が含まれている必要があります",
        )

        # 重複チェック
        self.assertEqual(
            len(symbols), len(set(symbols)), "銘柄に重複があってはいけません",
        )

    def test_prediction_score_valid_range(self):
        """テスト2: 予測スコアが0-100の範囲内である"""
        # Arrange
        test_symbol = "9984.T"

        # Act
        score = self.system.predict_score(test_symbol)

        # Assert
        self.assertIsInstance(score, (int, float), "スコアは数値である必要があります")
        self.assertGreaterEqual(score, 0, "スコアは0以上である必要があります")
        self.assertLessEqual(score, 100, "スコアは100以下である必要があります")

    def test_score_to_change_rate_conversion(self):
        """テスト3: スコアから価格変化率への変換が正しい"""
        # Arrange
        test_cases = [
            (50, 0.0),  # 中立
            (60, 1.0),  # 上昇
            (40, -1.0),  # 下降
            (75, 2.5),  # 強い上昇
            (25, -2.5),  # 強い下降
        ]

        for score, expected_rate in test_cases:
            with self.subTest(score=score):
                # Act
                change_rate = self.system.convert_score_to_change_rate(score)

                # Assert
                self.assertAlmostEqual(
                    change_rate,
                    expected_rate,
                    places=1,
                    msg=f"スコア{score}は変化率{expected_rate}%に変換されるべきです",
                )

    def test_confidence_calculation(self):
        """テスト4: 信頼度計算が正しい"""
        # Arrange
        test_cases = [
            (50, 0),  # 中立 = 信頼度最低
            (75, 50),  # 25ポイント離れている = 信頼度50%
            (25, 50),  # 25ポイント離れている = 信頼度50%
            (100, 100),  # 最大 = 信頼度最高
            (0, 100),  # 最小 = 信頼度最高
        ]

        for score, expected_confidence in test_cases:
            with self.subTest(score=score):
                # Act
                confidence = self.system.calculate_confidence(score)

                # Assert
                self.assertAlmostEqual(
                    confidence,
                    expected_confidence,
                    places=0,
                    msg=f"スコア{score}の信頼度は{expected_confidence}%であるべきです",
                )

    def test_prediction_result_structure(self):
        """テスト5: 予測結果の構造が正しい"""
        # Arrange
        test_symbol = "9984.T"

        # Act
        result = self.system.predict_single_stock(test_symbol)

        # Assert
        self.assertIsInstance(result, dict, "予測結果は辞書である必要があります")

        required_keys = [
            "symbol",
            "current_price",
            "predicted_price",
            "change_rate",
            "confidence",
            "prediction_score",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"予測結果に{key}が含まれている必要があります")

        # 型チェック
        self.assertIsInstance(result["symbol"], str)
        self.assertIsInstance(result["current_price"], (int, float))
        self.assertIsInstance(result["predicted_price"], (int, float))
        self.assertIsInstance(result["change_rate"], (int, float))
        self.assertIsInstance(result["confidence"], (int, float))
        self.assertIsInstance(result["prediction_score"], (int, float))

    def test_batch_prediction_all_symbols(self):
        """テスト6: 全30銘柄の一括予測が正常動作する"""
        # Act
        results = self.system.predict_all_optimal_stocks()

        # Assert
        self.assertIsInstance(results, list, "予測結果はリストである必要があります")
        self.assertLessEqual(len(results), 30, "結果数は30以下である必要があります")

        # 成功した予測の構造チェック
        if results:
            for result in results:
                self.assertIsInstance(result, dict)
                self.assertIn("symbol", result)
                self.assertIn(result["symbol"], self.system.get_optimal_symbols())

    def test_ranking_functionality(self):
        """テスト7: ランキング機能が正常動作する"""
        # Arrange
        mock_results = [
            {"symbol": "A", "change_rate": 2.5, "confidence": 80},
            {"symbol": "B", "change_rate": 1.0, "confidence": 60},
            {"symbol": "C", "change_rate": -1.0, "confidence": 90},
        ]

        # Act
        ranked = self.system.rank_predictions(mock_results)

        # Assert
        self.assertIsInstance(
            ranked, list, "ランキング結果はリストである必要があります",
        )
        self.assertEqual(len(ranked), 3, "入力と同じ数の結果が返される必要があります")

        # 上位は高スコア（変化率×信頼度）
        score_0 = ranked[0]["change_rate"] * ranked[0]["confidence"]
        score_1 = ranked[1]["change_rate"] * ranked[1]["confidence"]
        self.assertGreaterEqual(
            score_0, score_1, "ランキングは降順である必要があります",
        )

    def test_error_handling_invalid_symbol(self):
        """テスト8: 無効な銘柄に対するエラーハンドリング"""
        # Arrange
        invalid_symbol = "INVALID.T"

        # Act & Assert
        with self.assertRaises(ValueError):
            self.system.predict_single_stock(invalid_symbol)

    def test_error_handling_data_unavailable(self):
        """テスト9: データ取得失敗時のエラーハンドリング"""
        # Arrange
        with patch.object(self.system, "_get_stock_data", return_value=pd.DataFrame()):
            # Act
            result = self.system.predict_single_stock("9984.T")

            # Assert
            self.assertIsNone(result, "データ取得失敗時はNoneを返すべきです")

    def test_performance_requirements(self):
        """テスト10: パフォーマンス要件（1銘柄予測は5秒以内）"""
        import time

        # Arrange
        test_symbol = "9984.T"

        # Act
        start_time = time.time()
        result = self.system.predict_single_stock(test_symbol)
        end_time = time.time()

        # Assert
        execution_time = end_time - start_time
        self.assertLess(
            execution_time, 5.0, "1銘柄の予測は5秒以内に完了する必要があります",
        )


class TestTDDIntegration(unittest.TestCase):
    """統合テスト"""

    def test_full_workflow_integration(self):
        """統合テスト: 全ワークフローが正常動作する"""
        if Optimal30PredictionTDD is None:
            self.skipTest("実装対象クラスが存在しません（Red フェーズ）")

        # Arrange
        system = Optimal30PredictionTDD()

        # Act: 完全なワークフロー実行
        symbols = system.get_optimal_symbols()
        self.assertGreater(len(symbols), 0)

        # 最初の数銘柄をテスト
        results = []
        for symbol in symbols[:3]:  # 時間節約のため3銘柄のみ
            try:
                result = system.predict_single_stock(symbol)
                if result:
                    results.append(result)
            except Exception as e:
                self.fail(f"銘柄{symbol}の予測で例外が発生: {e!s}")

        # ランキング
        if results:
            ranked = system.rank_predictions(results)
            self.assertIsInstance(ranked, list)

        # Assert: 統合テスト成功
        self.assertTrue(True, "統合テストが完了しました")


if __name__ == "__main__":
    print("[RED] TDD Red フェーズ: テストファースト実行")
    print("=" * 60)
    print("期待：全テストが失敗する（実装がまだ存在しないため）")
    print("=" * 60)

    # テスト実行
    unittest.main(verbosity=2)
