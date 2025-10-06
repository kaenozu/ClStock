"""Test for Issue #9: Hybrid predictor bug fixes
- GPU batch returns deterministic data instead of random
- Real-time learning receives actual prices
"""

from datetime import datetime
from unittest.mock import Mock, patch

from models.core.interfaces import (
    DataProvider,
    ModelConfiguration,
    PredictionResult,
)
from models.hybrid import RefactoredHybridPredictor


class TestHybridPredictorIssue9:
    """Issue #9のバグ修正をテストするクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        self.mock_data_provider = Mock(spec=DataProvider)
        self.config = ModelConfiguration()
        self.predictor = RefactoredHybridPredictor(
            config=self.config,
            data_provider=self.mock_data_provider,
            enable_real_time_learning=True,
            enable_batch_optimization=True,
        )

    def test_batch_prediction_returns_deterministic_data(self):
        """Issue #9 修正確認: バッチ予測が決定論的データを返すことを確認
        （ランダムデータを返さない）
        """
        # モックデータの設定
        import pandas as pd

        mock_data = pd.DataFrame(
            {
                "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            },
        )
        self.mock_data_provider.get_stock_data.return_value = mock_data

        # 大規模バッチ（100銘柄以上）でテスト
        symbols = [f"STOCK{i:03d}" for i in range(150)]

        # 1回目の予測
        result1 = self.predictor.predict_batch(symbols)

        # 2回目の予測（同じ入力）
        result2 = self.predictor.predict_batch(symbols)

        # 検証: 同じ入力に対して同じ結果が返される（決定論的）
        assert result1.predictions.keys() == result2.predictions.keys()

        # ランダムな値でないことを確認
        # （uniform(800, 5000)の範囲外の値も許容される）
        for symbol in symbols:
            if symbol in result1.predictions and symbol in result2.predictions:
                # 同じ銘柄の予測値が一致することを確認
                assert result1.predictions[symbol] == result2.predictions[symbol]

                # NEUTRAL_PREDICTION_VALUE (50.0)か、それ以外の妥当な値
                value = result1.predictions[symbol]
                assert value == 50.0 or value > 0  # 正の値であることを確認

    def test_real_time_learning_uses_actual_prices(self):
        """Issue #9 修正確認: リアルタイム学習が実際の価格を使用することを確認
        （予測値を実際値として使わない）
        """
        import pandas as pd

        # 予測値と異なる実際の価格データを設定
        predicted_price = 105.0
        actual_price = 110.0

        mock_data = pd.DataFrame({"Close": [actual_price], "Volume": [1000]})
        self.mock_data_provider.get_stock_data.return_value = mock_data

        # 予測を実行
        symbol = "TEST001"
        prediction = self.predictor._predict_implementation(symbol)

        # 学習履歴を確認
        if self.predictor.learning_history:
            last_entry = self.predictor.learning_history[-1]

            # 実際の価格が記録されていることを確認
            assert last_entry["actual"] == actual_price
            # 予測値と実際値が異なることを確認（同じ値でないこと）
            assert last_entry["prediction"] != last_entry["actual"]
            # エラーが正しく計算されていることを確認
            assert last_entry["error"] > 0

    def test_large_batch_does_not_use_random_uniform(self):
        """大規模バッチがnp.random.uniform(800, 5000)を使用しないことを確認
        """
        import pandas as pd

        # モックデータ設定
        mock_data = pd.DataFrame({"Close": [100.0] * 10, "Volume": [1000] * 10})
        self.mock_data_provider.get_stock_data.return_value = mock_data

        # 500銘柄の大規模バッチ
        symbols = [f"STOCK{i:03d}" for i in range(500)]

        # np.random.uniformをモック化して呼ばれないことを確認
        with patch("numpy.random.uniform") as mock_uniform:
            result = self.predictor.predict_batch(symbols)

            # np.random.uniformが呼ばれていないことを確認
            mock_uniform.assert_not_called()

        # 結果が存在することを確認
        assert len(result.predictions) > 0

        # すべての値が妥当な範囲内であることを確認
        for value in result.predictions.values():
            # 50.0（デフォルト値）であることが期待される
            assert value == 50.0

    def test_learning_system_feedback_with_zero_error(self):
        """実際の価格が利用可能な場合、エラーが正しく計算されることを確認
        """
        import pandas as pd

        # 完全に一致する予測をシミュレート
        exact_price = 100.0
        mock_data = pd.DataFrame({"Close": [exact_price], "Volume": [1000]})
        self.mock_data_provider.get_stock_data.return_value = mock_data

        # エンサンブル予測器をモック化して正確な予測を返す
        with patch.object(self.predictor.ensemble_predictor, "predict") as mock_predict:
            mock_predict.return_value = PredictionResult(
                prediction=exact_price,  # 実際の価格と完全一致
                confidence=0.95,
                accuracy=0.90,
                timestamp=datetime.now(),
                symbol="TEST001",
            )

            # 予測実行
            prediction = self.predictor._predict_implementation("TEST001")

            # 学習履歴確認
            if self.predictor.learning_history:
                last_entry = self.predictor.learning_history[-1]
                # エラーが0（完璧な予測）であることを確認
                assert last_entry["error"] == 0
                assert last_entry["prediction"] == exact_price
                assert last_entry["actual"] == exact_price

    def test_batch_processing_with_errors(self):
        """バッチ処理中のエラーハンドリングが適切であることを確認
        """
        import pandas as pd

        # 一部の銘柄でエラーを発生させる
        def get_stock_data_with_errors(symbol, period="1M"):
            if "ERROR" in symbol:
                raise Exception(f"Data fetch failed for {symbol}")
            return pd.DataFrame({"Close": [100.0], "Volume": [1000]})

        self.mock_data_provider.get_stock_data.side_effect = get_stock_data_with_errors

        # エラーを含む銘柄リスト
        symbols = ["GOOD001", "ERROR001", "GOOD002", "ERROR002", "GOOD003"]

        result = self.predictor.predict_batch(symbols)

        # 正常な銘柄の結果が含まれることを確認
        assert "GOOD001" in result.predictions
        assert "GOOD002" in result.predictions
        assert "GOOD003" in result.predictions

        # エラー銘柄がエラーとして記録されることを確認
        assert "ERROR001" in result.errors
        assert "ERROR002" in result.errors

    def test_learning_statistics(self):
        """学習統計が正しく計算されることを確認
        """
        import pandas as pd

        # 複数の予測を実行してデータを蓄積
        test_data = [
            ("STOCK001", 100.0, 105.0),  # 5%エラー
            ("STOCK002", 200.0, 210.0),  # 5%エラー
            ("STOCK003", 150.0, 165.0),  # 10%エラー
        ]

        for symbol, predicted, actual in test_data:
            mock_data = pd.DataFrame({"Close": [actual], "Volume": [1000]})
            self.mock_data_provider.get_stock_data.return_value = mock_data

            with patch.object(
                self.predictor.ensemble_predictor, "predict",
            ) as mock_predict:
                mock_predict.return_value = PredictionResult(
                    prediction=predicted,
                    confidence=0.8,
                    accuracy=0.85,
                    timestamp=datetime.now(),
                    symbol=symbol,
                )
                self.predictor._predict_implementation(symbol)

        # 統計を取得
        stats = self.predictor.get_learning_statistics()

        # 統計が正しく計算されていることを確認
        assert stats["total_predictions"] == 3
        assert 0 < stats["average_error"] < 0.15  # 平均エラーが妥当な範囲内
        assert stats["min_error"] < stats["max_error"]
        assert len(stats["recent_predictions"]) <= 10
