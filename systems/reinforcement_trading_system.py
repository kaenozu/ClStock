"""強化学習による取引戦略の最適化"""
<<<<<<< HEAD
from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable
=======

import logging
>>>>>>> 84b59d4 (feat(quality): Improve code quality and add documentation)

import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
<<<<<<< HEAD
=======

import numpy as np
import pandas as pd
>>>>>>> 84b59d4 (feat(quality): Improve code quality and add documentation)

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """株式取引環境"""

    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.total_shares = 0
        self.net_worth = initial_balance

        # 行動空間: -1 (売る), 0 (ホールド), 1 (買う)
        self.action_space = spaces.Discrete(3)

        # 観測空間: 株価、残高、保有株式数、純資産など
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(data.columns) + 4,), dtype=np.float32,
        )

        # 利益記録用
        self.profits = []
        self.actions = []

    def reset(self):
        """環境をリセット"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares = 0
        self.net_worth = self.initial_balance
        self.profits = []
        self.actions = []

        return self._get_observation()

    def _get_observation(self):
        """現在の観測を取得"""
        current_data = self.data.iloc[self.current_step].values
        observation = np.concatenate(
            [
                current_data,
                [
                    self.balance / self.initial_balance,  # 残高比率
                    self.shares_held,  # 保有株式数
                    self.net_worth / self.initial_balance,  # 純資産比率
                    self.current_step / len(self.data),
                ],  # 経過時間比率
            ],
        )
        return observation.astype(np.float32)

    def step(self, action):
        """1ステップ進める"""
        current_price = self.data.iloc[self.current_step]["Close"]

        # 行動に基づく処理
        if action == 0:  # ホールド
            reward = 0
        elif action == 1:  # 買う
            # 1株買う
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price
        elif action == 2:  # 売る
            # 1株売る
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price

        # 純資産を更新
        self.net_worth = self.balance + self.shares_held * current_price

        # 報酬計算（利益と損失）
        if len(self.profits) > 0:
            previous_net_worth = self.profits[-1]
            reward = (self.net_worth - previous_net_worth) / previous_net_worth
        else:
            reward = 0

        self.profits.append(self.net_worth)
        self.actions.append(action)

        # 次のステップへ
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # 終了時の報酬調整
        if done:
            # 最終的な利益を報酬に加える
            total_return = (
                self.net_worth - self.initial_balance
            ) / self.initial_balance
            reward += total_return

        return self._get_observation(), reward, done, {}

    def render(self, mode="human"):
        """環境の状態を表示"""
        profit = self.net_worth - self.initial_balance
        print(
            f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Profit: {profit:.2f}",
        )


class TradingCallback(BaseCallback):
    """強化学習のコールバック"""

    def __init__(self, verbose=0):
        super(TradingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # 各ステップで実行する処理
        return True

<<<<<<< HEAD
@runtime_checkable
class MarketDataProviderProtocol(Protocol):
    """強化学習システムが利用する市場データプロバイダーのプロトコル。"""

    def get_stock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """指定した銘柄・期間の株価データを返す。"""


class ReinforcementTradingSystem:
    """強化学習による取引戦略最適化システム。

    Notes
    -----
    ``data_provider`` には :class:`MarketDataProviderProtocol` を満たすオブジェクトを渡す必要がある。
    ``get_stock_data`` は以下の条件を満たす :class:`pandas.DataFrame` を返却しなければならない。

    * インデックスは昇順にソートされた :class:`pandas.DatetimeIndex`
    * 必須カラム: ``Open``, ``High``, ``Low``, ``Close``, ``Volume``
    * 実市場から取得した数値が格納されていること（テストでは同形式のスタブで代用可）
    """

    REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

    def __init__(self, data_provider: Optional[MarketDataProviderProtocol] = None):
        self.data_provider: Optional[MarketDataProviderProtocol] = data_provider
        self.model = None
        self.env = None

    def prepare_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """取引環境用データの準備。

        Parameters
        ----------
        symbol:
            取得対象の銘柄コード。
        period:
            データ取得対象期間（例: ``"1y"`` や ``"6mo"`` など）。

        Returns
        -------
        pandas.DataFrame
            強化学習環境が利用可能な株価データ。

        Raises
        ------
        ValueError
            ``data_provider`` が未設定、もしくは必須条件を満たさないデータが返却された場合。
        """

        if self.data_provider is None:
            raise ValueError(
                "data_provider が設定されていません。設定ファイルで市場データの取得先を指定するか、"
                "ReinforcementTradingSystem に対応した data_provider を注入してください。"
            )

        stock_data = self.data_provider.get_stock_data(symbol, period)

        if not isinstance(stock_data, pd.DataFrame):
            raise ValueError("data_provider.get_stock_data は pandas.DataFrame を返却する必要があります。")

        if not isinstance(stock_data.index, pd.DatetimeIndex):
            raise ValueError("株価データのインデックスは pandas.DatetimeIndex でなければなりません。")

        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in stock_data.columns]
        if missing_columns:
            raise ValueError(f"株価データに必須カラムが不足しています: {', '.join(missing_columns)}")

        if not stock_data.index.is_monotonic_increasing:
            stock_data = stock_data.sort_index()

        return stock_data
        
    def train_model(self, symbol: str, period: str = '1y', timesteps: int = 10000) -> None:
=======

class ReinforcementTradingSystem:
    """強化学習による取引戦略最適化システム"""

    def __init__(self, data_provider=None):
        self.data_provider = data_provider
        self.model = None
        self.env = None

    def prepare_data(self, symbol, period="1y"):
        """取引環境用データの準備"""
        if self.data_provider:
            stock_data = self.data_provider.get_stock_data(symbol, period)
        else:
            # ダミーデータの作成（デバッグ用）
            dates = pd.date_range(
                start="2023-01-01", periods=252, freq="D",
            )  # 1年分のデータ
            close_prices = np.random.normal(100, 10, size=len(dates))  # ダミー株価
            stock_data = pd.DataFrame(
                {
                    "Date": dates,
                    "Close": close_prices,
                    "Open": close_prices
                    * np.random.uniform(0.99, 1.01, size=len(dates)),
                    "High": close_prices
                    * np.random.uniform(1.0, 1.05, size=len(dates)),
                    "Low": close_prices * np.random.uniform(0.95, 1.0, size=len(dates)),
                    "Volume": np.random.randint(100000, 1000000, size=len(dates)),
                },
            )
            stock_data.set_index("Date", inplace=True)

        return stock_data

    def train_model(self, symbol, period="1y", timesteps=10000):
>>>>>>> 84b59d4 (feat(quality): Improve code quality and add documentation)
        """モデルの訓練"""
        # 取引環境の作成
        stock_data = self.prepare_data(symbol, period)
        self.env = DummyVecEnv([lambda: TradingEnv(stock_data)])

        # PPOモデルの作成と訓練
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log="./ppo_trading_tensorboard/",
        )
        self.model.learn(total_timesteps=timesteps, callback=TradingCallback())

        logger.info(f"強化学習モデルの訓練完了: {symbol}")

<<<<<<< HEAD
    def predict_action(self, observation) -> int:
=======
    def predict_action(self, observation):
>>>>>>> 84b59d4 (feat(quality): Improve code quality and add documentation)
        """行動の予測"""
        if self.model is None:
            raise ValueError("モデルが訓練されていません")

        action, _states = self.model.predict(observation)
        return action

<<<<<<< HEAD
    def evaluate_strategy(self, symbol: str, period: str = '6m') -> float:
=======
    def evaluate_strategy(self, symbol, period="6m"):
>>>>>>> 84b59d4 (feat(quality): Improve code quality and add documentation)
        """取引戦略の評価"""
        stock_data = self.prepare_data(symbol, period)
        test_env = TradingEnv(stock_data)
        obs = test_env.reset()

        total_reward = 0
        done = False

        while not done:
            action = self.predict_action(obs)
            obs, reward, done, info = test_env.step(action)
            total_reward += reward

        logger.info(f"取引戦略評価完了: {symbol}, 総報酬: {total_reward}")
        return total_reward
