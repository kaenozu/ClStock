"""
動的ポジションサイジングモジュール
Kelly Criterion、リスクベースサイジング、ボラティリティ調整などを実装
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class PositionSizeResult:
    """ポジションサイズ計算結果"""
    shares: int  # 株数
    position_value: float  # ポジション価値
    risk_percentage: float  # リスク率
    confidence_adjustment: float  # 信頼度調整
    volatility_adjustment: float  # ボラティリティ調整


class KellyCriterionSizer:
    """ケリー基準ポジションサイジング"""
    
    def __init__(self, max_kelly_fraction: float = 0.25):
        """
        Args:
            max_kelly_fraction: ケリー基準の最大割合（過剰なリスクを避けるため）
        """
        self.max_kelly_fraction = max_kelly_fraction
        
    def calculate_kelly_fraction(self, win_rate: float, avg_win_rate: float, avg_loss_rate: float) -> float:
        """
        ケリー基準の資金比率を計算
        
        Args:
            win_rate: 勝率
            avg_win_rate: 平均利益率
            avg_loss_rate: 平均損失率
            
        Returns:
            資金比率
        """
        if avg_loss_rate == 0 or avg_win_rate <= 0:
            return 0.0
            
        # ケリー基準の式: f = (bp - q) / b
        # b = 勝ち時の利益率 / トゥーロス率（損失率）
        b = avg_win_rate / abs(avg_loss_rate)
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b if b != 0 else 0.0
        
        # 最大ケリー割合を超えないように制限
        return max(0.0, min(kelly_fraction, self.max_kelly_fraction))
    
    def calculate_position_size(self, 
                              capital: float, 
                              price: float, 
                              win_rate: float, 
                              avg_win_rate: float, 
                              avg_loss_rate: float,
                              max_position_ratio: float = 0.10) -> PositionSizeResult:
        """
        ポジションサイズを計算
        
        Args:
            capital: 総資本
            price: 現在価格
            win_rate: 勝率
            avg_win_rate: 平均利益率
            avg_loss_rate: 平均損失率
            max_position_ratio: 最大ポジション比率
            
        Returns:
            PositionSizeResult
        """
        kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win_rate, avg_loss_rate)
        
        # ケリー基準による資金
        kelly_capital = capital * kelly_fraction
        
        # ポジション上限をかける
        max_capital_for_position = capital * max_position_ratio
        
        # 保守的な方を採用
        actual_capital = min(kelly_capital, max_capital_for_position)
        
        # 株数を計算
        shares = int(actual_capital / price) if price > 0 else 0
        position_value = shares * price
        
        risk_percentage = avg_loss_rate if avg_loss_rate != 0 else 0.02  # デフォルト2%
        
        return PositionSizeResult(
            shares=shares,
            position_value=position_value,
            risk_percentage=risk_percentage,
            confidence_adjustment=1.0,  # ケリー基準では1.0（調整なし）
            volatility_adjustment=1.0   # ケリー基準では1.0（調整なし）
        )


class RiskBasedSizer:
    """リスクベースポジションサイジング"""
    
    def __init__(self, max_risk_percentage: float = 0.02):
        """
        Args:
            max_risk_percentage: 最大リスク率（例: 0.02で2%）
        """
        self.max_risk_percentage = max_risk_percentage
        
    def calculate_position_size(self,
                                capital: float,
                                price: float,
                                stop_loss_price: Optional[float] = None,
                                risk_percentage: Optional[float] = None,
                                volatility: Optional[float] = None,
                                max_position_ratio: float = 0.10) -> PositionSizeResult:
        """
        リスクベースでポジションサイズを計算
        
        Args:
            capital: 総資本
            price: 現在価格
            stop_loss_price: 損切り価格
            risk_percentage: リスク率（指定がなければmax_risk_percentageを使用）
            volatility: ボラティリティ
            max_position_ratio: 最大ポジション比率
            
        Returns:
            PositionSizeResult
        """
        if risk_percentage is None:
            risk_percentage = self.max_risk_percentage
            
        # 損切り価格が指定されている場合、損失リスクを計算
        if stop_loss_price is not None and price > stop_loss_price:
            risk_per_share = price - stop_loss_price
        else:
            # 損切り価格がない場合、ボラティリティまたはデフォルトリスクを使用
            if volatility is not None:
                risk_per_share = price * volatility * 2  # 2σとしてリスクを計算
            else:
                risk_per_share = price * risk_percentage  # リスク率に基づく
        
        # リスク金額を計算
        risk_amount = capital * risk_percentage
        
        # 株数を計算
        shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        
        # 最大ポジション制限をかける
        max_shares = int(capital * max_position_ratio / price) if price > 0 else 0
        shares = min(shares, max_shares)
        
        position_value = shares * price
        actual_risk_percentage = (risk_per_share * shares) / capital if capital > 0 else 0
        
        return PositionSizeResult(
            shares=shares,
            position_value=position_value,
            risk_percentage=actual_risk_percentage,
            confidence_adjustment=1.0,
            volatility_adjustment=1.0
        )


class VolatilityAdjustedSizer:
    """ボラティリティ調整ポジションサイジング"""
    
    def __init__(self, base_risk_percentage: float = 0.02, volatility_target: float = 0.20):
        """
        Args:
            base_risk_percentage: 基準リスク率
            volatility_target: 目標ボラティリティ（年率）
        """
        self.base_risk_percentage = base_risk_percentage
        self.volatility_target = volatility_target
    
    def calculate_position_size(self,
                                capital: float,
                                price: float,
                                volatility: float,  # 252日ボラティリティ
                                max_position_ratio: float = 0.10) -> PositionSizeResult:
        """
        ボラティリティ調整でポジションサイズを計算
        
        Args:
            capital: 総資本
            price: 現在価格
            volatility: ボラティリティ（年率）
            max_position_ratio: 最大ポジション比率
            
        Returns:
            PositionSizeResult
        """
        # ボラティリティが目標より高い場合はポジションを小さく、低い場合は大きく
        if volatility > 0:
            adjustment_factor = self.volatility_target / volatility
        else:
            adjustment_factor = 1.0  # ボラティリティが0の場合は調整なし
        
        # 調整されたリスク率
        adjusted_risk_percentage = self.base_risk_percentage * adjustment_factor
        
        # 最大リスク率を制限
        adjusted_risk_percentage = min(adjusted_risk_percentage, self.base_risk_percentage * 2)
        
        # リスク金額
        risk_amount = capital * adjusted_risk_percentage
        
        # リスクを価格の変動幅として計算（1σとして）
        risk_per_share = price * (volatility / np.sqrt(252))  # 日次ボラティリティ
        
        # 株数を計算
        shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        
        # 最大ポジション制限
        max_shares = int(capital * max_position_ratio / price) if price > 0 else 0
        shares = min(shares, max_shares)
        
        position_value = shares * price
        actual_risk_percentage = (risk_per_share * shares) / capital if capital > 0 else 0
        
        return PositionSizeResult(
            shares=shares,
            position_value=position_value,
            risk_percentage=actual_risk_percentage,
            confidence_adjustment=1.0,
            volatility_adjustment=adjustment_factor
        )


class CorrelationAdjustedSizer:
    """相関調整ポジションサイジング"""
    
    def __init__(self, base_sizer: Optional[object] = None):
        """
        Args:
            base_sizer: 基本のポジションサイジングクラス（RiskBasedSizer等）
        """
        self.base_sizer = base_sizer or RiskBasedSizer()
    
    def calculate_position_size(self,
                                capital: float,
                                price: float,
                                correlation_with_portfolio: float,
                                **kwargs) -> PositionSizeResult:
        """
        相関調整でポジションサイズを計算
        
        Args:
            capital: 総資本
            price: 現在価格
            correlation_with_portfolio: ポートフォリオとの相関
            **kwargs: base_sizerに渡すその他のパラメータ
            
        Returns:
            PositionSizeResult
        """
        # 基本のポジションサイズを計算
        base_result = self.base_sizer.calculate_position_size(capital, price, **kwargs)
        
        # 相関が高いほどポジションを小さく、低いほど大きく
        correlation_factor = 1.0 - max(0.0, correlation_with_portfolio)  # 相関が1.0のとき0、0のとき1
        
        # 調整後の株数
        adjusted_shares = int(base_result.shares * correlation_factor)
        
        # 最終的なポジション価値
        final_position_value = adjusted_shares * price
        
        return PositionSizeResult(
            shares=adjusted_shares,
            position_value=final_position_value,
            risk_percentage=base_result.risk_percentage * correlation_factor,
            confidence_adjustment=base_result.confidence_adjustment,
            volatility_adjustment=base_result.volatility_adjustment
        )


class AdaptivePositionSizer:
    """適応型ポジションサイジング"""
    
    def __init__(self, 
                 initial_capital: float = 1000000,
                 max_position_ratio: float = 0.10,
                 max_risk_percentage: float = 0.02):
        """
        Args:
            initial_capital: 初期資本
            max_position_ratio: 最大ポジション比率
            max_risk_percentage: 最大リスク率
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_ratio = max_position_ratio
        self.max_risk_percentage = max_risk_percentage
        
        # 各種サイジング戦略
        self.kelly_sizer = KellyCriterionSizer()
        self.risk_sizer = RiskBasedSizer(max_risk_percentage=max_risk_percentage)
        self.volatility_sizer = VolatilityAdjustedSizer()
        self.correlation_sizer = CorrelationAdjustedSizer(self.risk_sizer)
        
        # パフォーマンス履歴
        self.performance_history: List[Dict] = []
    
    def calculate_optimal_position_size(self,
                                      symbol: str,
                                      price: float,
                                      win_rate: Optional[float] = None,
                                      avg_win_rate: Optional[float] = None,
                                      avg_loss_rate: Optional[float] = None,
                                      stop_loss_price: Optional[float] = None,
                                      volatility: Optional[float] = None,
                                      correlation_with_portfolio: Optional[float] = None,
                                      strategy_confidence: Optional[float] = None) -> PositionSizeResult:
        """
        総合的に最適なポジションサイズを計算
        
        Args:
            symbol: 銘柄コード
            price: 現在価格
            win_rate: 勝率
            avg_win_rate: 平均利益率
            avg_loss_rate: 平均損失率
            stop_loss_price: 損切り価格
            volatility: ボラティリティ
            correlation_with_portfolio: ポートフォリオとの相関
            strategy_confidence: 戦略信頼度
            
        Returns:
            PositionSizeResult
        """
        # ケリー基準による計算（勝率情報がある場合）
        kelly_result = None
        if all(x is not None for x in [win_rate, avg_win_rate, avg_loss_rate]):
            kelly_result = self.kelly_sizer.calculate_position_size(
                self.current_capital, price, win_rate, avg_win_rate, avg_loss_rate, self.max_position_ratio
            )
        
        # リスクベース計算
        risk_result = self.risk_sizer.calculate_position_size(
            self.current_capital, price, stop_loss_price, 
            max_position_ratio=self.max_position_ratio
        )
        
        # ボラティリティ調整計算（ボラティリティ情報がある場合）
        volatility_result = None
        if volatility is not None:
            volatility_result = self.volatility_sizer.calculate_position_size(
                self.current_capital, price, volatility, self.max_position_ratio
            )
        
        # 相関調整計算（相関情報がある場合）
        correlation_result = None
        if correlation_with_portfolio is not None:
            correlation_result = self.correlation_sizer.calculate_position_size(
                self.current_capital, price, correlation_with_portfolio,
                stop_loss_price=stop_loss_price
            )
        
        # 複数の計算結果から最も保守的なものを選択
        results = [r for r in [kelly_result, risk_result, volatility_result, correlation_result] if r is not None]
        
        if not results:
            # どれも計算できない場合はデフォルト値
            return PositionSizeResult(
                shares=int(self.current_capital * 0.01 / price) if price > 0 else 0,
                position_value=0.0,
                risk_percentage=0.01,
                confidence_adjustment=1.0,
                volatility_adjustment=1.0
            )
        
        # 最も株数が少ない保守的な方を選択
        final_result = min(results, key=lambda x: x.shares if x is not None else float('inf'))
        
        # 戦略信頼度がある場合、それに応じて調整
        if strategy_confidence is not None and final_result:
            adjusted_shares = int(final_result.shares * strategy_confidence)
            final_result = PositionSizeResult(
                shares=adjusted_shares,
                position_value=adjusted_shares * price,
                risk_percentage=final_result.risk_percentage * strategy_confidence,
                confidence_adjustment=strategy_confidence,
                volatility_adjustment=final_result.volatility_adjustment
            )
        
        return final_result or results[0]  # 最終的に最初の結果を返す
    
    def update_capital(self, new_capital: float):
        """資本を更新"""
        self.current_capital = new_capital
    
    def record_performance(self, symbol: str, result: Dict):
        """パフォーマンスを記録"""
        record = {
            'symbol': symbol,
            'timestamp': pd.Timestamp.now(),
            'result': result
        }
        self.performance_history.append(record)
        
        # 履史が100件を超えたら古いものを削除
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]


# 使用例とテスト
if __name__ == "__main__":
    print("動的ポジションサイジングモジュールのテスト")
    
    # ケリー基準のテスト
    kelly_sizer = KellyCriterionSizer()
    kelly_result = kelly_sizer.calculate_position_size(
        capital=1000000,
        price=3000,
        win_rate=0.6,  # 勝率60%
        avg_win_rate=0.05,  # 平均利益率5%
        avg_loss_rate=-0.03  # 平均損失率3%
    )
    print(f"ケリー基準ポジション: {kelly_result.shares}株, 価値: {kelly_result.position_value:,.0f}円")
    
    # リスクベースのテスト
    risk_sizer = RiskBasedSizer()
    risk_result = risk_sizer.calculate_position_size(
        capital=1000000,
        price=3000,
        stop_loss_price=2900,  # 損切り価格2900円
        max_position_ratio=0.10
    )
    print(f"リスクベースポジション: {risk_result.shares}株, 価値: {risk_result.position_value:,.0f}円")
    
    # ボラティリティ調整のテスト
    vol_sizer = VolatilityAdjustedSizer()
    vol_result = vol_sizer.calculate_position_size(
        capital=1000000,
        price=3000,
        volatility=0.25,  # 25%ボラティリティ
        max_position_ratio=0.10
    )
    print(f"ボラティリティ調整ポジション: {vol_result.shares}株, 価値: {vol_result.position_value:,.0f}円")
    
    # 適応型サイジングのテスト
    adaptive_sizer = AdaptivePositionSizer()
    adaptive_result = adaptive_sizer.calculate_optimal_position_size(
        symbol="7203",
        price=3000,
        win_rate=0.6,
        avg_win_rate=0.05,
        avg_loss_rate=-0.03,
        stop_loss_price=2900,
        volatility=0.25,
        correlation_with_portfolio=0.3,
        strategy_confidence=0.8
    )
    print(f"適応型ポジション: {adaptive_result.shares}株, 価値: {adaptive_result.position_value:,.0f}円")
    print(f"リスク率: {adaptive_result.risk_percentage:.4f}, 信頼度調整: {adaptive_result.confidence_adjustment:.2f}")