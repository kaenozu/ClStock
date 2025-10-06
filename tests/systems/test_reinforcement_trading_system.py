import sys
import types

import numpy as np
import pandas as pd
import pytest


def _install_rl_stubs() -> None:
    """テスト環境用に強化学習関連のスタブモジュールを登録する。"""

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
        sys.modules["gym"] = gym_stub
        sys.modules["gym.spaces"] = spaces_stub

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

        sys.modules["stable_baselines3"] = sb3_stub
        sys.modules["stable_baselines3.common"] = common_stub
        sys.modules["stable_baselines3.common.vec_env"] = vec_env_stub
        sys.modules["stable_baselines3.common.callbacks"] = callbacks_stub


_install_rl_stubs()

from systems.reinforcement_trading_system import ReinforcementTradingSystem


class StubMarketDataProvider:
    """テスト用の市場データプロバイダー。"""

    def __init__(self):
        index = pd.date_range(end="2024-01-10", periods=10, freq="B")
        self._data = pd.DataFrame(
            {
                "Open": np.linspace(100, 109, len(index)),
                "High": np.linspace(101, 110, len(index)),
                "Low": np.linspace(99, 108, len(index)),
                "Close": np.linspace(100.5, 109.5, len(index)),
                "Volume": np.full(len(index), 1000, dtype=float),
            },
            index=index,
        )
        self.calls = []

    def get_stock_data(self, symbol: str, period: str) -> pd.DataFrame:
        self.calls.append((symbol, period))
        return self._data.copy()


def test_prepare_data_requires_provider():
    system = ReinforcementTradingSystem()

    with pytest.raises(ValueError) as excinfo:
        system.prepare_data("TEST")

    assert "data_provider" in str(excinfo.value)


def test_prepare_data_returns_provider_data_without_dummy_usage():
    provider = StubMarketDataProvider()
    system = ReinforcementTradingSystem(data_provider=provider)

    data = system.prepare_data("REAL", period="1mo")

    # プロバイダーが呼び出されたことを検証
    assert provider.calls == [("REAL", "1mo")]

    # 返されたデータが期待通りであることを検証
    pd.testing.assert_index_equal(data.index, provider._data.index)
    pd.testing.assert_frame_equal(data, provider._data)
