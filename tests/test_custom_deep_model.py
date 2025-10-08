"""新モデルのテストと評価"""

import os
import sys
import types
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os
import types

# ClStockプロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis.model_interpretability import ModelInterpretability
from analysis.multimodal_integration import MultimodalIntegrator
from models.attention_mechanism import (
    AdaptiveTemporalAttention,
    MultiHeadTemporalAttention,
    TemporalAttention,
)
from models.custom_deep_predictor import CustomDeepPredictor


def _install_reinforcement_learning_stubs() -> None:
    if "gym" not in sys.modules:
        gym_stub = types.ModuleType("gym")

        class _DummyEnv:
            pass

        gym_stub.Env = _DummyEnv
        spaces_stub = types.ModuleType("gym.spaces")

        class _DummyDiscrete:
            def __init__(self, n):
                self.n = n

        class _DummyBox:
            def __init__(self, low, high, shape, dtype=None):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        spaces_stub.Discrete = _DummyDiscrete
        spaces_stub.Box = _DummyBox
        gym_stub.spaces = spaces_stub
        sys.modules.setdefault("gym", gym_stub)
        sys.modules.setdefault("gym.spaces", spaces_stub)

    if "stable_baselines3" not in sys.modules:
        sb3_stub = types.ModuleType("stable_baselines3")

        class _DummyPPO:
            def __init__(self, *_, **__):
                pass

            def learn(self, *_, **__):
                return self

            def predict(self, observation):
                return 0, None

        sb3_stub.PPO = _DummyPPO

        common_stub = types.ModuleType("stable_baselines3.common")
        vec_env_stub = types.ModuleType("stable_baselines3.common.vec_env")

        class _DummyVecEnv:
            def __init__(self, env_fns):
                self.envs = [fn() for fn in env_fns]

            def reset(self):
                return self.envs[0].reset()

        vec_env_stub.DummyVecEnv = _DummyVecEnv

        callbacks_stub = types.ModuleType("stable_baselines3.common.callbacks")

        class _DummyBaseCallback:
            def __init__(self, *_, **__):
                pass

        callbacks_stub.BaseCallback = _DummyBaseCallback

        sys.modules.setdefault("stable_baselines3", sb3_stub)
        sys.modules.setdefault("stable_baselines3.common", common_stub)
        sys.modules.setdefault("stable_baselines3.common.vec_env", vec_env_stub)
        sys.modules.setdefault("stable_baselines3.common.callbacks", callbacks_stub)


_install_reinforcement_learning_stubs()

from models.self_supervised_learning import (
    SelfSupervisedModel,
    TemporalSelfSupervisedModel,
)
from systems.reinforcement_trading_system import ReinforcementTradingSystem


class TestCustomDeepModel(unittest.TestCase):
    """新モデルのテストクラス"""

    def setUp(self):
        """テスト前準備"""
        # ダミーデータの作成
        self.n_samples = 1000
        self.n_features = 10
        self.n_timesteps = 50

        # 時系列データの作成
        self.X_time_series = np.random.random((self.n_samples, self.n_timesteps, 1))
        self.y_time_series = np.random.random((self.n_samples, 1))

        # 通常の特徴量データの作成
        self.X_features = np.random.random((self.n_samples, self.n_features))
        self.y_features = np.random.random((self.n_samples,))

        # マルチモーダルデータの作成
        dates = pd.date_range(start="2023-01-01", periods=self.n_samples, freq="D")
        self.price_data = pd.DataFrame(
            {
                "Date": dates,
                "Close": np.random.normal(100, 10, size=self.n_samples),
                "Open": np.random.normal(100, 10, size=self.n_samples),
                "High": np.random.normal(105, 10, size=self.n_samples),
                "Low": np.random.normal(95, 10, size=self.n_samples),
                "Volume": np.random.randint(100000, 1000000, size=self.n_samples),
            },
        )
        self.price_data.set_index("Date", inplace=True)

        self.fundamental_data = pd.DataFrame(
            {
                "PBR": np.random.uniform(0.5, 3.0, size=self.n_samples),
                "PER": np.random.uniform(5, 30, size=self.n_samples),
                "ROE": np.random.uniform(0.01, 0.2, size=self.n_samples),
            },
        )

        self.sentiment_data = pd.DataFrame(
            {"sentiment_score": np.random.uniform(-1, 1, size=self.n_samples)},
        )

    def test_custom_deep_predictor(self):
        """独自深層学習モデルのテスト"""
        # モデルの作成
        input_shape = (self.n_timesteps, 1)
        model = CustomDeepPredictor(input_shape=input_shape)

        # 訓練データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_time_series,
            self.y_time_series,
            test_size=0.2,
            random_state=42,
        )

        # モデル訓練
        model.train(X_train, y_train, X_test, y_test, epochs=5, batch_size=32)

        # 予測
        predictions = model.predict(X_test)

        # 評価
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"Custom Deep Predictor - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # 基本的な検証
        self.assertGreaterEqual(r2, -1.0)  # R2は-1以上
        self.assertGreater(mse, 0)  # MSEは正
        self.assertGreater(mae, 0)  # MAEは正

    def test_multimodal_integration(self):
        """マルチモーダル統合のテスト"""
        integrator = MultimodalIntegrator()

        # データ統合
        integrated_features = integrator.integrate_data(
            self.price_data,
            self.fundamental_data,
            self.sentiment_data,
        )

        # 期待される形状の検証（価格データの特徴量数 + ファンダメンタルデータ数 + センチメントデータ数）
        expected_features = (
            len(self.price_data.columns)
            + len(self.fundamental_data.columns)
            + len(self.sentiment_data.columns)
        )
        self.assertEqual(integrated_features.shape[1], expected_features)
        print(f"Multimodal Integration - Shape: {integrated_features.shape}")

    def test_attention_mechanism(self):
        """アテンション機構のテスト"""
        # Temporal Attentionのテスト
        attention_layer = TemporalAttention(attention_units=32)
        input_tensor = np.random.random((1, self.n_timesteps, self.n_features))
        output = attention_layer(input_tensor)

        # 出力形状の検証
        self.assertEqual(output.shape[1], 32)
        print(f"Temporal Attention - Output Shape: {output.shape}")

        # Multi-Head Temporal Attentionのテスト
        multi_head_attention = MultiHeadTemporalAttention(
            num_heads=4,
            attention_units_per_head=16,
        )
        output_multi = multi_head_attention(input_tensor)

        # 出力形状の検証
        self.assertEqual(output_multi.shape[1], 64)  # 4 heads * 16 units per head
        print(f"Multi-Head Temporal Attention - Output Shape: {output_multi.shape}")

        # Adaptive Temporal Attentionのテスト
        adaptive_attention = AdaptiveTemporalAttention(attention_units=48)
        output_adaptive = adaptive_attention(input_tensor)

        # 出力形状の検証
        self.assertEqual(output_adaptive.shape[1], 48)
        print(f"Adaptive Temporal Attention - Output Shape: {output_adaptive.shape}")

    class _StubMarketDataProvider:
        """強化学習システム用の市場データスタブ。"""

        def __init__(self):
            dates = pd.date_range(start="2023-01-01", periods=30, freq="B")
            self.data = pd.DataFrame(
                {
                    "Open": np.linspace(100, 105, len(dates)),
                    "High": np.linspace(101, 106, len(dates)),
                    "Low": np.linspace(99, 104, len(dates)),
                    "Close": np.linspace(100.5, 105.5, len(dates)),
                    "Volume": np.linspace(1_000, 2_000, len(dates)),
                },
                index=dates,
            )

        def get_stock_data(self, symbol: str, period: str) -> pd.DataFrame:
            return self.data.copy()

    def test_reinforcement_trading_system(self):
        """強化学習取引システムのテスト（最小限のテスト）"""
        trading_system = ReinforcementTradingSystem(
            data_provider=self._StubMarketDataProvider(),
        )
        # 訓練（短い期間でテスト）
        trading_system.train_model("TEST_SYMBOL", period="10D", timesteps=100)

        # 評価
        reward = trading_system.evaluate_strategy("TEST_SYMBOL", period="5D")

        print(f"Reinforcement Trading System - Total Reward: {reward}")

        # 報酬が数値であることを検証
        self.assertIsInstance(reward, (int, float))

    def test_self_supervised_learning(self):
        """自己教師あり学習のテスト"""
        # Autoencoderモデルのテスト
        input_shape = (self.n_features,)
        ssl_model = SelfSupervisedModel(input_shape=input_shape, encoding_dim=32)

        # 訓練
        ssl_model.train(self.X_features, epochs=5, batch_size=32)

        # 特徴量抽出
        extracted_features = ssl_model.extract_features(self.X_features)

        # 特徴量の形状検証
        self.assertEqual(extracted_features.shape[1], 32)
        print(
            f"Self-Supervised Learning - Extracted Features Shape: {extracted_features.shape}",
        )

        # 時系列用Autoencoderモデルのテスト
        temporal_ssl_model = TemporalSelfSupervisedModel(
            input_shape=(self.n_timesteps, 1),
            encoding_dim=16,
        )

        # 訓練
        temporal_ssl_model.train(self.X_time_series, epochs=5, batch_size=32)

        # 特徴量抽出
        temporal_extracted_features = temporal_ssl_model.extract_features(
            self.X_time_series,
        )

        # 特徴量の形状検証
        self.assertEqual(temporal_extracted_features.shape[1], 16)
        print(
            f"Temporal Self-Supervised Learning - Extracted Features Shape: {temporal_extracted_features.shape}",
        )

    def test_model_interpretability(self):
        """モデル解釈性のテスト（最小限のテスト）"""
        # ダミーモデルの作成（実際には学習済みモデルを使用）
        dummy_model = lambda x: np.mean(x, axis=1)  # ダミーの予測関数

        # 解釈性クラスの作成
        interpreter = ModelInterpretability(
            dummy_model,
            self.X_features,
            feature_names=[f"feature_{i}" for i in range(self.n_features)],
        )

        # Explainerのセットアップ
        interpreter.setup_explainer(method="kernel")

        # SHAP値の計算
        shap_values = interpreter.calculate_shap_values(
            self.X_features[:10],
        )  # 最初の10サンプルのみ

        # 基本的な検証
        self.assertEqual(len(shap_values), 10)
        print(f"Model Interpretability - SHAP Values Shape: {shap_values.shape}")

    def test_end_to_end_integration(self):
        """エンドツーエンドの統合テスト（最小限のテスト）"""
        # マルチモーダル統合
        integrator = MultimodalIntegrator()
        integrated_data = integrator.integrate_data(
            self.price_data,
            self.fundamental_data,
            self.sentiment_data,
        )

        # 時系列データに変換（CNNモデル用）
        if len(integrated_data.shape) == 2:
            integrated_time_series = integrated_data.reshape(
                integrated_data.shape[0],
                integrated_data.shape[1],
                1,
            )

        # 独自深層学習モデル
        input_shape = (integrated_time_series.shape[1], 1)
        model = CustomDeepPredictor(input_shape=input_shape)

        # ダミーの出力データ作成（実際には株価変動率などを使用）
        y_dummy = np.random.random((integrated_time_series.shape[0], 1))

        # 訓練データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(
            integrated_time_series,
            y_dummy,
            test_size=0.2,
            random_state=42,
        )

        # モデル訓練（短いエポック数でテスト）
        model.train(X_train, y_train, X_test, y_test, epochs=3, batch_size=32)

        # 予測
        predictions = model.predict(X_test)

        # 評価
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"End-to-End Integration - MSE: {mse:.4f}, R2: {r2:.4f}")

        # 基本的な検証
        self.assertGreaterEqual(r2, -1.0)  # R2は-1以上
        self.assertGreater(mse, 0)  # MSEは正


if __name__ == "__main__":
    unittest.main()
