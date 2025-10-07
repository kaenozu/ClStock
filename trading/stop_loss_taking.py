"""
ストップロス・利確最適化モジュール
ATRベースストップ、トレーリングストップ、分割利確などを実装
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class ExitStrategyResult:
    """出口戦略計算結果"""
    stop_loss_price: float  # 損切り価格
    take_profit_price: float  # 利確価格
    trailing_stop_enabled: bool  # トレーリングストップ有効
    trailing_distance: float  # トレーリング距離
    exit_strategy_type: str  # 出口戦略タイプ


class ATRBasedStopLoss:
    """ATRベースストップロス設定"""
    
    def __init__(self, atr_period: int = 14, multiplier: float = 2.0):
        """
        Args:
            atr_period: ATR計算期間
            multiplier: ATR乗数
        """
        self.atr_period = atr_period
        self.multiplier = multiplier
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        ATR(True Range)を計算
        
        Args:
            df: OHLCデータ ('High', 'Low', 'Close'カラムが必要)
            
        Returns:
            ATRのSeries
        """
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()
        
        return atr
    
    def calculate_stop_loss(self, 
                           current_price: float, 
                           df: pd.DataFrame, 
                           direction: str = "long") -> Tuple[float, float]:
        """
        ATRベースのストップロス価格を計算
        
        Args:
            current_price: 現在価格
            df: OHLCデータ
            direction: "long" or "short"
            
        Returns:
            (ストップロス価格, ATR値)
        """
        atr_values = self.calculate_atr(df)
        current_atr = atr_values.iloc[-1] if not atr_values.empty else 0.0
        
        if direction.lower() == "long":
            # ロングポジションでは、現在価格からATR×乗数を引く
            stop_loss = current_price - (current_atr * self.multiplier)
        else:
            # ショートポジションでは、現在価格にATR×乗数を足す
            stop_loss = current_price + (current_atr * self.multiplier)
        
        return stop_loss, current_atr


class TrailingStop:
    """トレーリングストップモジュール"""
    
    def __init__(self, trail_percentage: float = 0.10, trail_amount: Optional[float] = None):
        """
        Args:
            trail_percentage: トレーリング割合（価格の何%離れるか）
            trail_amount: トレーリング金額（固定金額でのトレーリング）
        """
        self.trail_percentage = trail_percentage
        self.trail_amount = trail_amount
        self.highest_price = None  # ロングポジション用
        self.lowest_price = None   # ショートポジション用
    
    def update_price(self, current_price: float, direction: str = "long") -> float:
        """
        価格更新とストップ価格計算
        
        Args:
            current_price: 現在価格
            direction: "long" or "short"
            
        Returns:
            最適ストップ価格
        """
        direction = direction.lower()
        
        if direction == "long":
            # ロングポジション: 最高値を追跡し、一定割合/金額下回ったら損切り
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
            
            if self.trail_amount is not None:
                stop_price = self.highest_price - self.trail_amount
            else:
                stop_price = self.highest_price * (1 - self.trail_percentage)
                
        else:
            # ショートポジション: 最安値を追跡し、一定割合/金額上回ったら損切り
            if self.lowest_price is None or current_price < self.lowest_price:
                self.lowest_price = current_price
            
            if self.trail_amount is not None:
                stop_price = self.lowest_price + self.trail_amount
            else:
                stop_price = self.lowest_price * (1 + self.trail_percentage)
        
        return stop_price
    
    def should_exit(self, current_price: float, direction: str = "long") -> bool:
        """
        退出シグナルがあるか判定
        
        Args:
            current_price: 現在価格
            direction: "long" or "short"
            
        Returns:
            退出すべきかどうか
        """
        stop_price = self.update_price(current_price, direction)
        
        if direction == "long":
            return current_price <= stop_price
        else:
            return current_price >= stop_price
    
    def reset(self):
        """トレーリングストップをリセット"""
        self.highest_price = None
        self.lowest_price = None


class RiskRewardOptimizer:
    """リスク・リワード比最適化"""
    
    def __init__(self, target_risk_reward: float = 2.0):
        """
        Args:
            target_risk_reward: 目標リスク・リワード比
        """
        self.target_risk_reward = target_risk_reward
    
    def calculate_optimal_levels(self, 
                                entry_price: float, 
                                stop_loss_distance: float,
                                risk_reward_ratio: Optional[float] = None) -> Tuple[float, float]:
        """
        最適なストップロス・利確レベルを計算
        
        Args:
            entry_price: エントリー価格
            stop_loss_distance: 損切り距離（価格単位）
            risk_reward_ratio: リスク・リワード比（指定なければtargetを使用）
            
        Returns:
            (ストップロス価格, 利確価格)
        """
        if risk_reward_ratio is None:
            risk_reward_ratio = self.target_risk_reward
        
        # ストップロス価格（ロングの場合）
        stop_loss_price = entry_price - stop_loss_distance
        
        # 目標リターン（損切り距離×リスク・リワード比）
        target_return = stop_loss_distance * risk_reward_ratio
        
        # 利確価格
        take_profit_price = entry_price + target_return
        
        return stop_loss_price, take_profit_price


class MultipleProfitTaking:
    """分割利確モジュール"""
    
    def __init__(self, levels: List[Tuple[float, float]] = [(0.5, 0.30), (0.3, 0.60), (0.2, 1.00)]):
        """
        Args:
            levels: 分割利確レベル (割合, 利確率) のリスト
                   例: [(0.5, 0.30), (0.3, 0.60), (0.2, 1.00)] 
                   -> 総額の50%を利益30%で、30%を利益60%で、20%を利益100%で利確
        """
        self.levels = levels  # (割合, 利確率) のリスト
    
    def calculate_exit_prices(self, entry_price: float, base_target: float) -> List[Tuple[float, int]]:
        """
        分割利確の価格を計算
        
        Args:
            entry_price: エントリー価格
            base_target: 基準目標価格
            
        Returns:
            [(利確価格, 売却割合), ...] のリスト
        """
        exits = []
        
        for portion, profit_ratio in self.levels:
            take_profit_price = entry_price + (base_target - entry_price) * profit_ratio
            exits.append((take_profit_price, portion))
        
        return exits


class SmartExitStrategy:
    """スマート出口戦略統合モジュール"""
    
    def __init__(self, 
                 atr_period: int = 14, 
                 atr_multiplier: float = 2.0,
                 trail_percentage: float = 0.10,
                 target_risk_reward: float = 2.0,
                 enable_multiple_taking: bool = True):
        """
        Args:
            atr_period: ATR計算期間
            atr_multiplier: ATR乗数
            trail_percentage: トレーリング割合
            target_risk_reward: 目標リスク・リワード比
            enable_multiple_taking: 分割利確を有効にするか
        """
        self.atr_stop = ATRBasedStopLoss(atr_period=atr_period, multiplier=atr_multiplier)
        self.trailing_stop = TrailingStop(trail_percentage=trail_percentage)
        self.risk_reward_optimizer = RiskRewardOptimizer(target_risk_reward=target_risk_reward)
        self.multiple_taking = MultipleProfitTaking() if enable_multiple_taking else None
        
        # 市場状態に基づく調整用パラメータ
        self.market_volatility_factor = 1.0
    
    def calculate_exit_strategy(self, 
                               entry_price: float, 
                               df: pd.DataFrame,
                               direction: str = "long",
                               initial_stop_distance: Optional[float] = None,
                               market_condition: str = "normal") -> ExitStrategyResult:
        """
        総合的な出口戦略を計算
        
        Args:
            entry_price: エントリー価格
            df: OHLCデータ
            direction: "long" or "short"
            initial_stop_distance: 初期ストップ距離（指定があればそれを使用）
            market_condition: "high_volatility", "normal", "low_volatility"
            
        Returns:
            ExitStrategyResult
        """
        # 市場状況に基づく調整
        if market_condition == "high_volatility":
            self.market_volatility_factor = 1.3
        elif market_condition == "low_volatility":
            self.market_volatility_factor = 0.8
        else:
            self.market_volatility_factor = 1.0
        
        # ATRベースのストップロス
        if initial_stop_distance is None:
            atr_stop_price, current_atr = self.atr_stop.calculate_stop_loss(entry_price, df, direction)
            stop_distance = abs(entry_price - atr_stop_price)
        else:
            stop_distance = initial_stop_distance
            current_atr = stop_distance / self.atr_stop.multiplier
            # ATRベースのストップ価格を再計算
            if direction == "long":
                atr_stop_price = entry_price - stop_distance
            else:
                atr_stop_price = entry_price + stop_distance
        
        # リスク・リワード比に基づく利確価格
        _, take_profit_price = self.risk_reward_optimizer.calculate_optimal_levels(
            entry_price, 
            stop_distance * self.market_volatility_factor
        )
        
        # トレーリングストップ設定
        trailing_enabled = True
        trailing_distance = current_atr * self.atr_stop.multiplier if current_atr > 0 else stop_distance * 0.5
        
        # 戦略タイプの決定
        strategy_type = "atr_trailing"
        if market_condition == "high_volatility":
            strategy_type = "volatility_adaptive"
        elif market_condition == "low_volatility":
            strategy_type = "tight_stop"
        
        return ExitStrategyResult(
            stop_loss_price=atr_stop_price,
            take_profit_price=take_profit_price,
            trailing_stop_enabled=trailing_enabled,
            trailing_distance=trailing_distance * self.market_volatility_factor,
            exit_strategy_type=strategy_type
        )
    
    def evaluate_exit_conditions(self,
                                 current_price: float,
                                 exit_strategy: ExitStrategyResult,
                                 direction: str = "long") -> Dict[str, bool]:
        """
        退出条件を評価
        
        Args:
            current_price: 現在価格
            exit_strategy: 出口戦略
            direction: "long" or "short"
            
        Returns:
            どの退出条件が満たされているかの辞書
        """
        direction = direction.lower()
        
        # 損切り条件
        stop_loss_hit = (direction == "long" and current_price <= exit_strategy.stop_loss_price) or \
                       (direction == "short" and current_price >= exit_strategy.stop_loss_price)
        
        # 利確条件
        take_profit_hit = (direction == "long" and current_price >= exit_strategy.take_profit_price) or \
                         (direction == "short" and current_price <= exit_strategy.take_profit_price)
        
        # トレーリングストップが有効な場合の評価
        if exit_strategy.trailing_stop_enabled:
            trailing_stop_price = self.trailing_stop.update_price(current_price, direction)
            trailing_stop_hit = (direction == "long" and current_price <= trailing_stop_price) or \
                               (direction == "short" and current_price >= trailing_stop_price)
        else:
            trailing_stop_hit = False
        
        return {
            'stop_loss': stop_loss_hit,
            'take_profit': take_profit_hit,
            'trailing_stop': trailing_stop_hit
        }


# 使用例とテスト
if __name__ == "__main__":
    print("ストップロス・利確最適化モジュールのテスト")
    
    # テスト用のダミーデータ生成
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    np.random.seed(42)
    
    # OHLCデータ（仮の価格）
    prices = 3000 + np.cumsum(np.random.normal(0, 20, 30))
    high_prices = prices + np.random.uniform(10, 30, 30)
    low_prices = prices - np.random.uniform(10, 30, 30)
    
    df = pd.DataFrame({
        'Close': prices,
        'High': high_prices,
        'Low': low_prices
    }, index=dates)
    
    # ATRベースストップロスのテスト
    atr_stop = ATRBasedStopLoss(atr_period=14, multiplier=2.0)
    entry_price = df['Close'].iloc[-1]
    stop_price, atr_value = atr_stop.calculate_stop_loss(entry_price, df, "long")
    print(f"ATRベースストップロス: エントリー価格 {entry_price:.0f}円, ストップ価格 {stop_price:.0f}円, ATR {atr_value:.2f}")
    
    # トレーリングストップのテスト
    trailing_stop = TrailingStop(trail_percentage=0.05)  # 5%トレーリング
    sample_prices = [3000, 3050, 3100, 3080, 3020]
    for price in sample_prices:
        stop_price = trailing_stop.update_price(price, "long")
        print(f"価格 {price}, トレーリングストップ {stop_price:.0f}")
    
    # リスク・リワード比最適化のテスト
    risk_reward = RiskRewardOptimizer(target_risk_reward=2.0)
    stop_price, profit_price = risk_reward.calculate_optimal_levels(entry_price=3000, stop_loss_distance=100)
    print(f"リスク・リワード最適化: ストップ {stop_price:.0f}円, 利確 {profit_price:.0f}円")
    
    # 分割利確のテスト
    multi_taking = MultipleProfitTaking()
    exit_levels = multi_taking.calculate_exit_prices(entry_price=3000, base_target=3300)
    print(f"分割利確レベル: {[(f'{price:.0f}円', f'{portion:.0%}') for price, portion in exit_levels]}")
    
    # スマート出口戦略のテスト
    smart_exit = SmartExitStrategy()
    exit_strategy = smart_exit.calculate_exit_strategy(
        entry_price=3000,
        df=df,
        direction="long",
        market_condition="normal"
    )
    print(f"スマート出口戦略:")
    print(f"  ストップロス価格: {exit_strategy.stop_loss_price:.0f}円")
    print(f"  利確価格: {exit_strategy.take_profit_price:.0f}円")
    print(f"  トレーリング有効: {exit_strategy.trailing_stop_enabled}")
    print(f"  戦略タイプ: {exit_strategy.exit_strategy_type}")
    
    # 退出条件評価のテスト
    conditions = smart_exit.evaluate_exit_conditions(current_price=3050, exit_strategy=exit_strategy, direction="long")
    print(f"退出条件評価 (価格3050): {conditions}")