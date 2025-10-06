"""
高度なリスク管理とVaR計算モジュール
GARCHモデル、ストレステスト、シナリオ分析などを含む
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
from scipy import stats

# ARCHモデルはオプションの依存関係であるため、エラーを処理
try:
    from arch import arch_model  # ARCHモデル用
    ARCH_AVAILABLE = True
except ImportError:
    arch_model = None
    ARCH_AVAILABLE = False

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class AdvancedRiskMetrics:
    """高度なリスク指標データクラス"""
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    current_pnl: float
    portfolio_value: float
    volatility_forecast: float  # 予測ボラティリティ
    tail_loss: float  # テールリスク
    stress_test_loss: float  # ストレステスト損失


class GARCHVaRModel:
    """GARCHモデルベースのVaR計算"""
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Args:
            p: GARCH(p,q)モデルのpパラメータ
            q: GARCH(p,q)モデルのqパラメータ
        """
        if not ARCH_AVAILABLE:
            print("警告: archモジュールが利用できません。GARCHモデルは使用できません。")
            print("インストールするには: pip install arch")
        
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
        
    def fit_model(self, returns: pd.Series) -> None:
        """GARCHモデルを学習"""
        if not ARCH_AVAILABLE:
            print("archモジュールが利用できないため、GARCHモデルを学習できません。")
            return
        
        if len(returns) < 100:  # 最低限のデータ数が必要
            print("警告: GARCHモデルの学習に十分なデータがありません")
            return
            
        try:
            self.model = arch_model(returns.dropna() * 100, p=self.p, q=self.q, dist='Normal')  # 標準化したリターン
            self.fitted_model = self.model.fit(update_freq=5, disp='off')
        except Exception as e:
            print(f"GARCHモデルの学習に失敗しました: {e}")
            self.fitted_model = None
    
    def forecast_volatility(self, steps: int = 1) -> float:
        """ボラティリティを予測"""
        if not ARCH_AVAILABLE or self.fitted_model is None:
            return 0.0
            
        try:
            forecast = self.fitted_model.forecast(horizon=steps)
            # 最新の予測されたボラティリティを返す
            var_forecast = forecast.variance.iloc[-1, 0]
            return np.sqrt(var_forecast) / 100  # 元のスケールに戻す
        except Exception:
            return 0.0
    
    def calculate_garch_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """GARCHモデルベースのVaRを計算"""
        if not ARCH_AVAILABLE or self.fitted_model is None:
            return 0.0
            
        try:
            # 予測されたボラティリティを取得
            forecast_vol = self.forecast_volatility()
            
            # リターンの平均を取得
            mean_return = returns.mean() if not returns.empty else 0.0
            
            # 正規分布仮定でVaRを計算
            z_score = stats.norm.ppf(1 - confidence)
            var_value = mean_return - z_score * forecast_vol
            
            return var_value
        except Exception:
            return 0.0


class StressTester:
    """ストレステストモジュール"""
    
    def __init__(self):
        pass
    
    def historical_stress_test(self, returns: pd.Series, scenarios: Optional[List[str]] = None) -> Dict[str, float]:
        """
        過去の極端な市場状況をシミュレーション
        
        Args:
            returns: 歴史的リターン
            scenarios: シナリオのリスト（指定がなければデフォルト）
            
        Returns:
            シナリオ名と損失の辞書
        """
        if scenarios is None:
            scenarios = [
                "2008年リーマンショック", 
                "2011年東日本大震災", 
                "2020年コロナショック"
            ]
        
        # 実際には特定の市場イベントの期間を定義し、それに対応するリターンを分析
        # ここでは簡略的に過去の最大損失期間をシミュレーション
        stress_results = {}
        
        if len(returns) == 0:
            return {scenario: 0.0 for scenario in scenarios}
        
        # 最悪の5日間リターンをシミュレーション（例として）
        window_size = 5
        if len(returns) >= window_size:
            rolling_returns = returns.rolling(window=window_size).sum()
            worst_periods = rolling_returns.nsmallest(3)  # 最悪3期間
            
            for i, scenario in enumerate(scenarios):
                if i < len(worst_periods):
                    stress_results[scenario] = worst_periods.iloc[i]
                else:
                    stress_results[scenario] = worst_periods.iloc[-1]  # 同じ値を繰り返す
        
        return stress_results
    
    def scenario_stress_test(self, base_returns: pd.Series, scenario_impact: float = -0.10) -> float:
        """
        シナリオベースのストレステスト
        
        Args:
            base_returns: 基本リターン
            scenario_impact: シナリオの影響度（例: -0.10 で10%下落）
            
        Returns:
            シナリオ下での損失
        """
        # 基本リターンにシナリオ影響を加える
        stressed_returns = base_returns + scenario_impact
        
        # シナリオ下でのVaRを計算
        if len(stressed_returns) > 0:
            var_99 = np.percentile(stressed_returns.dropna(), 1)  # 99% VaR
        else:
            var_99 = 0.0
        
        return var_99


class CorrelationRiskManager:
    """相関リスク管理"""
    
    def __init__(self):
        self.correlation_matrix = None
    
    def calculate_portfolio_correlation_risk(self, 
                                           returns_dict: Dict[str, pd.Series],
                                           weights: Dict[str, float]) -> float:
        """
        ポートフォリオ相関リスクを計算
        
        Args:
            returns_dict: 銘柄別のリターンの辞書
            weights: 銘柄別のウェイトの辞書
            
        Returns:
            相関リスクスコア
        """
        if len(returns_dict) < 2:
            return 0.0
        
        # リターンのDataFrameを作成
        df_returns = pd.DataFrame(returns_dict)
        
        # 欠損値を処理
        df_returns = df_returns.dropna()
        
        if df_returns.empty or len(df_returns.columns) < 2:
            return 0.0
        
        # 相関行列を計算
        corr_matrix = df_returns.corr()
        
        # ウェイトベクトル
        weight_vector = np.array([weights.get(col, 0.0) for col in df_returns.columns])
        
        # 分散が0の資産を除外
        valid_indices = weight_vector != 0
        if not np.any(valid_indices):
            return 0.0
        
        # 有効な相関行列とウェイトを取得
        corr_matrix = corr_matrix.iloc[valid_indices, valid_indices]
        weight_vector = weight_vector[valid_indices]
        
        # ウェイトの正規化
        weight_sum = np.sum(weight_vector)
        if weight_sum != 0:
            weight_vector = weight_vector / weight_sum
        else:
            return 0.0
        
        # ポートフォリオの相関リスク（相関が高いとリスクが増加）
        portfolio_corr_risk = 1 - np.dot(np.dot(weight_vector, corr_matrix), weight_vector)
        
        return portfolio_corr_risk


class AdvancedRiskManager:
    """高度なリスク管理システム"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.historical_pnl: List[float] = []
        self.historical_portfolio_values: List[float] = []
        self.garch_model = GARCHVaRModel()
        self.stress_tester = StressTester()
        self.correlation_manager = CorrelationRiskManager()
        
        # リスク上限
        self.risk_limits = {
            'var_99': -0.08,  # 99%VaRが-8%を超えると警告
            'max_drawdown': -0.20,  # 最大ドローダウンが-20%を超えると警告
            'position_concentration': 0.30  # 単一銘柄への集中度が30%を超えると警告
        }
    
    def update_position(self, symbol: str, quantity: int, price: float, cash_flow: float = 0.0):
        """保有状況を更新し、現金残高も更新"""
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
        """ポートフォリオ価値を計算"""
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
    
    def calculate_diversification_ratio(self, returns_dict: Dict[str, pd.Series]) -> float:
        """分散化レシオを計算"""
        if len(returns_dict) < 2:
            return 0.0
        
        # 全体のポートフォリオリターン
        portfolio_returns = self.calculate_portfolio_returns()
        
        if portfolio_returns.empty:
            return 0.0
        
        portfolio_vol = portfolio_returns.std()
        if portfolio_vol == 0:
            return 1.0
        
        # 各資産のウェイト付きボラティリティ
        weighted_asset_vol = 0.0
        total_value = self.calculate_portfolio_value()
        
        if total_value == 0:
            return 0.0
        
        for symbol, returns in returns_dict.items():
            if symbol in self.positions:
                asset_weight = self.positions[symbol]['value'] / total_value
                asset_vol = returns.std() if not returns.empty else 0.0
                weighted_asset_vol += asset_weight * asset_vol
        
        if weighted_asset_vol == 0:
            return 1.0
        
        # 分散化レシオ = 資産加重平均ボラティリティ / ポートフォリオボラティリティ
        diversification_ratio = weighted_asset_vol / portfolio_vol
        
        # 分散化レシオが1より大きいと非分散化、1に近いと高度に分散化
        return min(diversification_ratio, 2.0)  # 最大2.0に制限
    
    def calculate_advanced_risk_metrics(self, 
                                      returns_dict: Optional[Dict[str, pd.Series]] = None) -> AdvancedRiskMetrics:
        """高度なリスク指標を計算"""
        portfolio_returns = self.calculate_portfolio_returns()
        
        # 基本的なリスク指標
        if not portfolio_returns.empty:
            var_95 = np.percentile(portfolio_returns.dropna(), 5)
            var_99 = np.percentile(portfolio_returns.dropna(), 1)
        else:
            var_95 = var_99 = 0.0
        
        # 期待ショートフォール
        if not portfolio_returns.empty:
            worst_5_returns = portfolio_returns.nsmallest(int(len(portfolio_returns) * 0.05))
            worst_1_returns = portfolio_returns.nsmallest(int(len(portfolio_returns) * 0.01))
            expected_shortfall_95 = worst_5_returns.mean() if not worst_5_returns.empty else 0.0
            expected_shortfall_99 = worst_1_returns.mean() if not worst_1_returns.empty else 0.0
        else:
            expected_shortfall_95 = expected_shortfall_99 = 0.0
        
        # ボラティリティ
        volatility = portfolio_returns.std() if not portfolio_returns.empty else 0.0
        
        # シャープレシオ
        if not portfolio_returns.empty and volatility != 0:
            sharpe_ratio = (portfolio_returns.mean() - 0.02) / volatility  # 無リスクレート2%と仮定
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
        
        # GARCHボラティリティ予測
        garch_vol = 0.0
        if not portfolio_returns.empty:
            self.garch_model.fit_model(portfolio_returns)
            garch_vol = self.garch_model.forecast_volatility()
        
        # テールリスク (最悪5%の損失の平均)
        tail_loss = expected_shortfall_95
        
        # ストレステスト損失
        stress_losses = self.stress_tester.historical_stress_test(portfolio_returns)
        stress_test_loss = min(stress_losses.values()) if stress_losses else 0.0
        
        # 現在の損益とポートフォリオ価値
        current_portfolio_value = self.calculate_portfolio_value()
        current_pnl = current_portfolio_value - self.initial_capital
        
        return AdvancedRiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=expected_shortfall_95,
            expected_shortfall_99=expected_shortfall_99,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            current_pnl=current_pnl,
            portfolio_value=current_portfolio_value,
            volatility_forecast=garch_vol,
            tail_loss=tail_loss,
            stress_test_loss=stress_test_loss
        )
    
    def check_risk_limits(self) -> Dict[str, Tuple[bool, float]]:
        """リスク上限チェック"""
        metrics = self.calculate_advanced_risk_metrics()
        
        checks = {}
        
        # VaRチェック
        checks['var_99'] = (metrics.var_99 >= self.risk_limits['var_99'], metrics.var_99)
        
        # 最大ドローダウンチェック
        checks['max_drawdown'] = (metrics.max_drawdown >= self.risk_limits['max_drawdown'], metrics.max_drawdown)
        
        # ポジション集中度チェック
        total_value = metrics.portfolio_value
        if total_value > 0:
            max_position_ratio = 0.0
            for symbol, pos_data in self.positions.items():
                ratio = pos_data['value'] / total_value
                max_position_ratio = max(max_position_ratio, ratio)
            checks['position_concentration'] = (max_position_ratio <= self.risk_limits['position_concentration'], max_position_ratio)
        else:
            checks['position_concentration'] = (True, 0.0)
        
        return checks
    
    def generate_risk_report(self) -> str:
        """リスクレポート生成"""
        metrics = self.calculate_advanced_risk_metrics()
        checks = self.check_risk_limits()
        
        report = []
        report.append("=" * 60)
        report.append("高度リスク管理レポート")
        report.append("=" * 60)
        report.append(f"ポートフォリオ価値: {metrics.portfolio_value:,.0f}円")
        report.append(f"現在の損益: {metrics.current_pnl:,.0f}円")
        report.append("")
        report.append("[リスク指標]")
        report.append(f"VaR 95%: {metrics.var_95:.4f} ({metrics.var_95*100:.2f}%)")
        report.append(f"VaR 99%: {metrics.var_99:.4f} ({metrics.var_99*100:.2f}%)")
        report.append(f"期待ショートフォール 95%: {metrics.expected_shortfall_95:.4f} ({metrics.expected_shortfall_95*100:.2f}%)")
        report.append(f"期待ショートフォール 99%: {metrics.expected_shortfall_99:.4f} ({metrics.expected_shortfall_99*100:.2f}%)")
        report.append(f"ボラティリティ: {metrics.volatility:.4f} ({metrics.volatility*100:.2f}%)")
        report.append(f"予測ボラティリティ: {metrics.volatility_forecast:.4f} ({metrics.volatility_forecast*100:.2f}%)")
        report.append(f"シャープレシオ: {metrics.sharpe_ratio:.4f}")
        report.append(f"最大ドローダウン: {metrics.max_drawdown:.4f} ({metrics.max_drawdown*100:.2f}%)")
        report.append(f"テールリスク: {metrics.tail_loss:.4f} ({metrics.tail_loss*100:.2f}%)")
        report.append(f"ストレステスト損失: {metrics.stress_test_loss:.4f} ({metrics.stress_test_loss*100:.2f}%)")
        report.append("")
        report.append("[リスク上限チェック]")
        for check_name, (passed, value) in checks.items():
            status = "✅ OK" if passed else "❌ 超過"
            report.append(f"{check_name}: {status} (値: {value:.4f})")
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


# 使用例とテスト
if __name__ == "__main__":
    print("高度リスク管理システムのテスト")
    
    # 仮のリターンデータ生成
    np.random.seed(42)
    sample_returns = pd.Series(np.random.normal(0.0005, 0.025, 500))  # 500日分のリターン
    
    # GARCHモデルのテスト
    garch_model = GARCHVaRModel()
    garch_model.fit_model(sample_returns)
    garch_var_95 = garch_model.calculate_garch_var(sample_returns, confidence=0.95)
    garch_var_99 = garch_model.calculate_garch_var(sample_returns, confidence=0.99)
    forecast_vol = garch_model.forecast_volatility()
    
    print(f"GARCHモデル VaR 95%: {garch_var_95:.4f}")
    print(f"GARCHモデル VaR 99%: {garch_var_99:.4f}")
    print(f"GARCH予測ボラティリティ: {forecast_vol:.4f}")
    
    # ストレステストのテスト
    stress_tester = StressTester()
    stress_results = stress_tester.historical_stress_test(sample_returns)
    print(f"\nストレステスト結果:")
    for scenario, loss in stress_results.items():
        print(f"  {scenario}: {loss:.4f}")
    
    # 高度リスク管理のテスト
    advanced_risk_manager = AdvancedRiskManager(initial_capital=1000000)
    
    # サンプルのポートフォリオ価値履歴を追加
    for i in range(60):
        value = 1000000 * (1 + np.random.normal(0.0005, 0.02, 1)[0] * i + np.random.normal(0, 0.015, 1)[0])
        advanced_risk_manager.historical_portfolio_values.append(value)
    
    # 銘柄保有状況を追加
    advanced_risk_manager.update_position("7203", 100, 3000)
    advanced_risk_manager.update_position("6758", 50, 12000)
    
    # 高度リスク指標計算
    advanced_metrics = advanced_risk_manager.calculate_advanced_risk_metrics()
    print(f"\n高度リスク指標:")
    print(f"VaR 95%: {advanced_metrics.var_95:.4f}")
    print(f"VaR 99%: {advanced_metrics.var_99:.4f}")
    print(f"ES 95%: {advanced_metrics.expected_shortfall_95:.4f}")
    print(f"ES 99%: {advanced_metrics.expected_shortfall_99:.4f}")
    print(f"ボラティリティ: {advanced_metrics.volatility:.4f}")
    print(f"予測ボラティリティ: {advanced_metrics.volatility_forecast:.4f}")
    print(f"シャープレシオ: {advanced_metrics.sharpe_ratio:.4f}")
    print(f"最大ドローダウン: {advanced_metrics.max_drawdown:.4f}")
    print(f"テールリスク: {advanced_metrics.tail_loss:.4f}")
    print(f"ストレステスト損失: {advanced_metrics.stress_test_loss:.4f}")
    print(f"ポートフォリオ価値: {advanced_metrics.portfolio_value:,.0f}円")
    
    # リスク上限チェック
    checks = advanced_risk_manager.check_risk_limits()
    print(f"\nリスク上限チェック:")
    for check_name, (passed, value) in checks.items():
        status = "✅ OK" if passed else "❌ 超過"
        print(f"  {check_name}: {status} (値: {value:.4f})")
    
    # リスクレポート生成
    report = advanced_risk_manager.generate_risk_report()
    print(f"\n{report}")