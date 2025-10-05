"""強化学習による取引戦略の最適化"""
import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import logging

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
            low=-np.inf, high=np.inf, shape=(len(data.columns) + 4,), dtype=np.float32
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
        observation = np.concatenate([
            current_data,
            [self.balance / self.initial_balance,  # 残高比率
             self.shares_held,                     # 保有株式数
             self.net_worth / self.initial_balance, # 純資産比率
             self.current_step / len(self.data)]   # 経過時間比率
        ])
        return observation.astype(np.float32)
        
    def step(self, action):
        """1ステップ進める"""
        current_price = self.data.iloc[self.current_step]['Close']
        
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
            total_return = (self.net_worth - self.initial_balance) / self.initial_balance
            reward += total_return
            
        return self._get_observation(), reward, done, {}
        
    def render(self, mode='human'):
        """環境の状態を表示"""
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Profit: {profit:.2f}')

class TradingCallback(BaseCallback):
    """強化学習のコールバック"""
    
    def __init__(self, verbose=0):
        super(TradingCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        # 各ステップで実行する処理
        return True

class ReinforcementTradingSystem:
    """強化学習による取引戦略最適化システム"""
    
    def __init__(self, data_provider=None):
        self.data_provider = data_provider
        self.model = None
        self.env = None
        
    def prepare_data(self, symbol, period='1y'):
        """取引環境用データの準備"""
        if self.data_provider:
            stock_data = self.data_provider.get_stock_data(symbol, period)
        else:
            # ダミーデータの作成（デバッグ用）
            dates = pd.date_range(start='2023-01-01', periods=252, freq='D')  # 1年分のデータ
            close_prices = np.random.normal(100, 10, size=len(dates))  # ダミー株価
            stock_data = pd.DataFrame({
                'Date': dates,
                'Close': close_prices,
                'Open': close_prices * np.random.uniform(0.99, 1.01, size=len(dates)),
                'High': close_prices * np.random.uniform(1.0, 1.05, size=len(dates)),
                'Low': close_prices * np.random.uniform(0.95, 1.0, size=len(dates)),
                'Volume': np.random.randint(100000, 1000000, size=len(dates))
            })
            stock_data.set_index('Date', inplace=True)
            
        return stock_data
        
    def train_model(self, symbol, period='1y', timesteps=10000):
        """モデルの訓練"""
        # 取引環境の作成
        stock_data = self.prepare_data(symbol, period)
        self.env = DummyVecEnv([lambda: TradingEnv(stock_data)])
        
        # PPOモデルの作成と訓練
        self.model = PPO('MlpPolicy', self.env, verbose=1, tensorboard_log="./ppo_trading_tensorboard/")
        self.model.learn(total_timesteps=timesteps, callback=TradingCallback())
        
        logger.info(f"強化学習モデルの訓練完了: {symbol}")
        
    def predict_action(self, observation):
        """行動の予測"""
        if self.model is None:
            raise ValueError("モデルが訓練されていません")
            
        action, _states = self.model.predict(observation)
        return action
        
    def evaluate_strategy(self, symbol, period='6m'):
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