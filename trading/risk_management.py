"""
リスク管理システムモジュール
VaR、ES、動的ポジショニングなど高度なリスク管理機能を提供
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class RiskMetrics:
    """リスク指標データクラス"""
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    expected_shortfall_95: float  # 95% ES
    expected_shortfall_99: float  # 99% ES
    volatility: float  # ボラティリティ
    sharpe_ratio: float  # シャープレシオ
    max_drawdown: float  # 最大ドローダウン
    current_pnl: float  # 現在の損益
    portfolio_value: float  # ポートフォリオ価値


class ValueAtRiskCalculator:
    """VaR計算器"""
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        Args:
            confidence_levels: 信頼区間のリスト (例: [0.95, 0.99])
        """
        self.confidence_levels = confidence_levels
        
    def calculate_historical_var(self, returns: pd.Series) -> Dict[float, float]:
        """
        ヒストリカルVaRを計算
        
        Args:
            returns: 資産リターンのSeries
            
        Returns:
            信頼区間に対応したVaR値の辞書
        """
        if len(returns) == 0:
            return {level: 0.0 for level in self.confidence_levels}
        
        var_values = {}
        for level in self.confidence_levels:
            # 逆CDF（左側）を使ってVaRを計算
            var_percentile = (1 - level) * 100
            var_value = np.percentile(returns.dropna(), var_percentile)
            var_values[level] = var_value
            
        return var_values
    
    def calculate_parametric_var(self, returns: pd.Series) -> Dict[float, float]:
        """
        パラメトリックVaR（正規分布仮定）を計算
        
        Args:
            returns: 資産リターンのSeries
            
        Returns:
            信頼区間に対応したVaR値の辞書
        """
        if len(returns) == 0:
            return {level: 0.0 for level in self.confidence_levels}
        
        returns_clean = returns.dropna()
        mean_return = returns_clean.mean()
        std_return = returns_clean.std()
        
        var_values = {}
        for level in self.confidence_levels:
            # 正規分布の逆CDFを使ってVaRを計算
            from scipy.stats import norm
            z_score = norm.ppf(1 - level)  # 例: 95% -> 1.645
            var_value = mean_return - z_score * std_return
            var_values[level] = var_value
            
        return var_values


class ExpectedShortfallCalculator:
    """期待ショートフォール計算器"""
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        Args:
            confidence_levels: 信頼区間のリスト (例: [0.95, 0.99])
        """
        self.confidence_levels = confidence_levels
        self.var_calculator = ValueAtRiskCalculator(confidence_levels=confidence_levels)
    
    def calculate_es(self, returns: pd.Series) -> Dict[float, float]:
        """
        期待ショートフォール(ES)を計算
        
        Args:
            returns: 資産リターンのSeries
            
        Returns:
            信頼区間に対応したES値の辞書
        """
        if len(returns) == 0:
            return {level: 0.0 for level in self.confidence_levels}
        
        returns_clean = returns.dropna()
        var_values = self.var_calculator.calculate_historical_var(returns)
        
        es_values = {}
        for level in self.confidence_levels:
            var_level = var_values[level]
            # VaRよりも悪いリターンの平均を計算
            worst_returns = returns_clean[returns_clean <= var_level]
            if len(worst_returns) > 0:
                es_values[level] = worst_returns.mean()
            else:
                # VaRに一致する値がなかった場合、近似計算
                es_values[level] = var_level
        
        return es_values


class PortfolioRiskManager:
    """ポートフォリオリスク管理クラス"""
    
    def __init__(self, initial_capital: float = 1000000):
        """
        Args:
            initial_capital: 初期資本
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict] = {}  # 銘柄別の保有状況
        self.historical_pnl: List[float] = []  # 履歴損益
        self.historical_portfolio_values: List[float] = []  # 履歴ポートフォリオ価値
        self.var_calculator = ValueAtRiskCalculator()
        self.es_calculator = ExpectedShortfallCalculator()
        
    def update_position(self, symbol: str, quantity: int, price: float, cash_flow: float = 0.0):
        """
        保有状況を更新し、現金残高も更新
        
        Args:
            symbol: 銘柄コード
            quantity: 保有数量
            price: 現在価格
            cash_flow: 現金の増減額（購入時は負、売却時は正）
        """
        if quantity > 0:
            self.positions[symbol] = {
                'quantity': quantity,
                'price': price,
                'value': quantity * price
            }
        elif symbol in self.positions:
            del self.positions[symbol]
        
        # 現金残高を更新
        self.current_capital += cash_flow
    
    def calculate_portfolio_value(self) -> float:
        """現在のポートフォリオ価値を計算"""
        total_value = self.current_capital
        for symbol, pos_data in self.positions.items():
            total_value += pos_data['value']
        return total_value
    
    def calculate_portfolio_returns(self) -> pd.Series:
        """ポートフォリオリターンを計算"""
        if len(self.historical_portfolio_values) < 2:
            return pd.Series([], dtype=float)
        
        portfolio_values = pd.Series(self.historical_portfolio_values)
        returns = portfolio_values.pct_change().dropna()
        return returns
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """リスク指標を計算"""
        portfolio_returns = self.calculate_portfolio_returns()
        
        # VaRの計算
        var_values = self.var_calculator.calculate_historical_var(portfolio_returns)
        var_95 = var_values.get(0.95, 0.0)
        var_99 = var_values.get(0.99, 0.0)
        
        # ESの計算
        es_values = self.es_calculator.calculate_es(portfolio_returns)
        es_95 = es_values.get(0.95, 0.0)
        es_99 = es_values.get(0.99, 0.0)
        
        # ボラティリティ
        volatility = portfolio_returns.std() if not portfolio_returns.empty else 0.0
        
        # シャープレシオ (無リスクレートを2%と仮定)
        if not portfolio_returns.empty and volatility != 0:
            sharpe_ratio = (portfolio_returns.mean() - 0.02) / volatility
        else:
            sharpe_ratio = 0.0
        
        # 最大ドローダウン
        portfolio_values = pd.Series(self.historical_portfolio_values)
        if not portfolio_values.empty:
            running_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - running_max) / running_max
            max_drawdown = drawdown.min() if not drawdown.empty else 0.0
        else:
            max_drawdown = 0.0
        
        # 現在の損益とポートフォリオ価値
        current_portfolio_value = self.calculate_portfolio_value()
        current_pnl = current_portfolio_value - self.initial_capital
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            current_pnl=current_pnl,
            portfolio_value=current_portfolio_value
        )
    
    def update_portfolio_history(self):
        """ポートフォリオ履歴を更新"""
        current_value = self.calculate_portfolio_value()
        self.historical_portfolio_values.append(current_value)
    
    def is_within_risk_limits(self, max_drawdown_limit: float = -0.15) -> bool:
        """
        リスク上限内かどうかチェック
        
        Args:
            max_drawdown_limit: 最大ドローダウン制限 (例: -0.15 で15%)
            
        Returns:
            True if within limits, False otherwise
        """
        metrics = self.calculate_risk_metrics()
        return metrics.max_drawdown >= max_drawdown_limit


class KellyCriterionPositionSizer:
    """ケリー基準によるポジショニング"""
    
    def __init__(self):
        pass
    
    def calculate_kelly_fraction(self, win_rate: float, avg_win_rate: float, avg_loss_rate: float) -> float:
        """
        ケリー基準に基づく資金比率を計算
        
        Args:
            win_rate: 勝率
            avg_win_rate: 平均利益率
            avg_loss_rate: 平均損失率
            
        Returns:
            資金比率
        """
        # ケリー基準の式: f = (bp - q) / b
        # b = 勝ち時の配当率 (利益率)
        # p = 勝率
        # q = 1 - 勝率
        if avg_loss_rate == 0:
            return 0.0
            
        b = avg_win_rate / abs(avg_loss_rate) if avg_loss_rate != 0 else 0
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b if b != 0 else 0.0
        
        # 実際にはケリー基準の一部(例: 50%)だけ使用して過剰なリスクを避ける
        return max(0.0, min(kelly_fraction * 0.5, 0.2))  # 最大20%までに制限
    
    def calculate_position_size(self, 
                              current_capital: float, 
                              win_rate: float, 
                              avg_win_rate: float, 
                              avg_loss_rate: float,
                              price: float,
                              risk_percentage: float = 0.02) -> int:
        """
        ポジションサイズを計算
        
        Args:
            current_capital: 現在の資本
            win_rate: 勝率
            avg_win_rate: 平均利益率
            avg_loss_rate: 平均損失率
            price: 現在価格
            risk_percentage: 許容リスク率 (例: 0.02 で2%)
            
        Returns:
            株数
        """
        # ケリー基準による資金比率
        kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win_rate, avg_loss_rate)
        
        # またはリスクベースでの計算
        risk_amount = current_capital * risk_percentage
        position_size = risk_amount / abs(avg_loss_rate) / price if avg_loss_rate != 0 else 0
        
        # ケリー基準でも計算
        kelly_position_value = current_capital * kelly_fraction
        kelly_position_size = kelly_position_value / price if price > 0 else 0
        
        # 両方の計算結果を考慮して保守的な方を採用
        final_position_size = min(position_size, kelly_position_size)
        
        return int(final_position_size)


class DynamicRiskManager:
    """動的风险管理システム"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.risk_manager = PortfolioRiskManager(initial_capital)
        self.position_sizer = KellyCriterionPositionSizer()
        self.max_position_size = 0.1  # 最大10%まで
        self.risk_thresholds = {
            'var_95': -0.05,  # 95%VaRが-5%を超えるとリスク警告
            'var_99': -0.10,  # 99%VaRが-10%を超えるとリスク警告
            'max_drawdown': -0.15  # 最大ドローダウンが-15%を超えるとリスク警告
        }
    
    def should_reduce_position(self, symbol: str, current_price: float) -> bool:
        """
        ポジション縮小が必要かどうか判断
        
        Args:
            symbol: 銘柄コード
            current_price: 現在価格
            
        Returns:
            True if position should be reduced, False otherwise
        """
        metrics = self.risk_manager.calculate_risk_metrics()
        
        # リスク閾値を超えていたら縮小を検討
        if (metrics.var_95 < self.risk_thresholds['var_95'] or 
            metrics.var_99 < self.risk_thresholds['var_99'] or 
            metrics.max_drawdown < self.risk_thresholds['max_drawdown']):
            return True
            
        return False
    
    def calculate_optimal_position_size(self, 
                                      symbol: str, 
                                      current_price: float,
                                      win_rate: float,
                                      avg_win_rate: float,
                                      avg_loss_rate: float) -> int:
        """
        最適なポジションサイズを計算
        
        Args:
            symbol: 銘柄コード
            current_price: 現在価格
            win_rate: 勝率
            avg_win_rate: 平均利益率
            avg_loss_rate: 平均損失率
            
        Returns:
            株数
        """
        current_capital = self.risk_manager.current_capital
        
        # ケリー基準とリスクベースで計算
        position_size = self.position_sizer.calculate_position_size(
            current_capital, win_rate, avg_win_rate, avg_loss_rate, current_price
        )
        
        # 最大ポジション制限をかける
        max_position_value = current_capital * self.max_position_size
        max_position_size = max_position_value / current_price if current_price > 0 else 0
        
        return min(position_size, int(max_position_size))
    
    def can_open_position(self, 
                         symbol: str, 
                         current_price: float,
                         predicted_return: float,
                         win_rate: float) -> bool:
        """
        ポジション開設が可能かどうか判断
        
        Args:
            symbol: 銘柄コード
            current_price: 現在価格
            predicted_return: 予測リターン
            win_rate: 勝率
            
        Returns:
            True if position can be opened, False otherwise
        """
        # リスク指標を確認
        metrics = self.risk_manager.calculate_risk_metrics()
        
        # リスクが上限を超えている場合は新規ポジションを制限
        if (metrics.var_95 < self.risk_thresholds['var_95'] or 
            metrics.var_99 < self.risk_thresholds['var_99'] or 
            metrics.max_drawdown < self.risk_thresholds['max_drawdown']):
            return False
        
        # 勝率が一定以上かつ予測リターンが正である場合にのみ開設
        return win_rate > 0.55 and predicted_return > 0.02  # 2%以上期待リターン
    
    def get_risk_adjusted_confidence(self, 
                                   base_confidence: float, 
                                   market_volatility: float,
                                   portfolio_concentration: float) -> float:
        """
        リスク調整された信頼度を計算
        
        Args:
            base_confidence: 基本信頼度
            market_volatility: 市場ボラティリティ
            portfolio_concentration: ポートフォリオ集中度
            
        Returns:
            リスク調整済み信頼度
        """
        # 市場ボラティリティが高いと信頼度を下げる
        volatility_factor = 1.0 / (1.0 + market_volatility)
        
        # ポートフォリオ集中度が高いと信頼度を下げる
        concentration_factor = 1.0 / (1.0 + portfolio_concentration)
        
        # リスク調整後の信頼度
        adjusted_confidence = base_confidence * volatility_factor * concentration_factor
        
        return max(0.0, min(1.0, adjusted_confidence))


# 使用例
if __name__ == "__main__":
    # サンプルデータでテスト
    print("リスク管理システムのテスト")
    
    # 仮のリターンデータを生成
    np.random.seed(42)
    sample_returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 年間252日分のリターン
    
    # VaR計算
    var_calc = ValueAtRiskCalculator()
    var_values = var_calc.calculate_historical_var(sample_returns)
    print(f"ヒストリカルVaR (95%): {var_values[0.95]:.4f}")
    print(f"ヒストリカルVaR (99%): {var_values[0.99]:.4f}")
    
    # ES計算
    es_calc = ExpectedShortfallCalculator()
    es_values = es_calc.calculate_es(sample_returns)
    print(f"期待ショートフォール (95%): {es_values[0.95]:.4f}")
    print(f"期待ショートフォール (99%): {es_values[0.99]:.4f}")
    
    # ポートフォリオリスク管理
    risk_manager = PortfolioRiskManager(initial_capital=1000000)
    
    # サンプルのポートフォリオ価値履歴を追加
    for i in range(30):
        # 仮のポートフォリオ価値（多少の変動を含む）
        value = 1000000 * (1 + np.random.normal(0.001, 0.015, 1)[0] * i)
        risk_manager.historical_portfolio_values.append(value)
    
    # リスク指標を計算
    metrics = risk_manager.calculate_risk_metrics()
    print(f"\nリスク指標:")
    print(f"VaR 95%: {metrics.var_95:.4f}")
    print(f"VaR 99%: {metrics.var_99:.4f}")
    print(f"ES 95%: {metrics.expected_shortfall_95:.4f}")
    print(f"ES 99%: {metrics.expected_shortfall_99:.4f}")
    print(f"ボラティリティ: {metrics.volatility:.4f}")
    print(f"シャープレシオ: {metrics.sharpe_ratio:.4f}")
    print(f"最大ドローダウン: {metrics.max_drawdown:.4f}")
    print(f"現在の損益: {metrics.current_pnl:,.0f}円")
    print(f"ポートフォリオ価値: {metrics.portfolio_value:,.0f}円")