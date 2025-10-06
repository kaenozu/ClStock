"""
リスク管理コンポーネントの包括的テスト
"""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.risk_management import (
    ValueAtRiskCalculator, 
    ExpectedShortfallCalculator, 
    PortfolioRiskManager, 
    KellyCriterionPositionSizer,
    DynamicRiskManager
)
from trading.advanced_risk_management import (
    GARCHVaRModel,
    StressTester,
    CorrelationRiskManager,
    AdvancedRiskManager
)
from trading.position_sizing import (
    KellyCriterionSizer,
    RiskBasedSizer,
    VolatilityAdjustedSizer,
    CorrelationAdjustedSizer,
    AdaptivePositionSizer,
    PositionSizeResult
)
from trading.stop_loss_taking import (
    ATRBasedStopLoss,
    TrailingStop,
    RiskRewardOptimizer,
    MultipleProfitTaking,
    SmartExitStrategy,
    ExitStrategyResult
)
from trading.comprehensive_risk_system import (
    ComprehensiveRiskManager,
    RiskManagedPortfolio,
    ComprehensiveRiskAssessment
)


class TestValueAtRiskCalculator(unittest.TestCase):
    """VaR計算器のテスト"""
    
    def setUp(self):
        self.var_calc = ValueAtRiskCalculator()
        # テスト用のリターンデータ
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    
    def test_historical_var_calculation(self):
        """ヒストリカルVaR計算のテスト"""
        var_values = self.var_calc.calculate_historical_var(self.returns)
        self.assertIn(0.95, var_values)
        self.assertIn(0.99, var_values)
        # VaRは通常負の値
        self.assertLessEqual(var_values[0.95], 0)
        self.assertLessEqual(var_values[0.99], 0)
        # 99%VaRは95%VaRより絶対値が大きい
        self.assertLessEqual(var_values[0.99], var_values[0.95])
    
    def test_parametric_var_calculation(self):
        """パラメトリックVaR計算のテスト"""
        var_values = self.var_calc.calculate_parametric_var(self.returns)
        self.assertIn(0.95, var_values)
        self.assertIn(0.99, var_values)
        # VaRは通常負の値
        self.assertLessEqual(var_values[0.95], 0)
        self.assertLessEqual(var_values[0.99], 0)
        # 99%VaRは95%VaRより絶対値が大きい
        self.assertLessEqual(var_values[0.99], var_values[0.95])


class TestExpectedShortfallCalculator(unittest.TestCase):
    """期待ショートフォール計算器のテスト"""
    
    def setUp(self):
        self.es_calc = ExpectedShortfallCalculator()
        # テスト用のリターンデータ
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    
    def test_es_calculation(self):
        """期待ショートフォール計算のテスト"""
        es_values = self.es_calc.calculate_es(self.returns)
        self.assertIn(0.95, es_values)
        self.assertIn(0.99, es_values)
        # ESは通常VaRより絶対値が大きい（リスクの期待値）
        var_calc = ValueAtRiskCalculator()
        var_values = var_calc.calculate_historical_var(self.returns)
        self.assertLessEqual(es_values[0.95], var_values[0.95])
        self.assertLessEqual(es_values[0.99], var_values[0.99])


class TestPortfolioRiskManager(unittest.TestCase):
    """ポートフォリオリスクマネージャーのテスト"""
    
    def setUp(self):
        self.risk_manager = PortfolioRiskManager(initial_capital=1000000)
        # サンプルのポートフォリオ価値履歴を追加
        for i in range(30):
            value = 1000000 * (1 + np.random.normal(0.001, 0.015, 1)[0] * i)
            self.risk_manager.historical_portfolio_values.append(value)
    
    def test_position_management(self):
        """保有状況管理のテスト"""
        self.risk_manager.update_position("7203", 100, 3000)
        self.assertIn("7203", self.risk_manager.positions)
        self.assertEqual(self.risk_manager.positions["7203"]["quantity"], 100)
        self.assertEqual(self.risk_manager.positions["7203"]["price"], 3000)
        self.assertEqual(self.risk_manager.positions["7203"]["value"], 300000)
        
        # 保有数0で削除
        self.risk_manager.update_position("7203", 0, 3000)
        self.assertNotIn("7203", self.risk_manager.positions)
    
    def test_portfolio_value_calculation(self):
        """ポートフォリオ価値計算のテスト"""
        # 株式購入: 100株 × 3000円 = 300,000円
        # 現金は300,000円減少し、ポジション価値は300,000円増加
        # 総ポートフォリオ価値はほぼ変化しないはず
        initial_capital = self.risk_manager.current_capital
        purchase_amount = 100 * 3000  # 300,000円
        
        self.risk_manager.update_position("7203", 100, 3000, cash_flow=-purchase_amount)
        portfolio_value = self.risk_manager.calculate_portfolio_value()
        
        # ポートフォリオ価値は初期資本と同等（現金減少と資産増加で相殺）
        expected_value = initial_capital  # 1,000,000円
        self.assertEqual(portfolio_value, expected_value)
    
    def test_risk_metrics_calculation(self):
        """リスク指標計算のテスト"""
        metrics = self.risk_manager.calculate_risk_metrics()
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics.var_95, float)
        self.assertIsInstance(metrics.var_99, float)
        self.assertIsInstance(metrics.sharpe_ratio, float)


class TestKellyCriterionSizer(unittest.TestCase):
    """ケリー基準サイザーのテスト"""
    
    def setUp(self):
        self.sizer = KellyCriterionSizer()
    
    def test_kelly_fraction_calculation(self):
        """ケリー分数計算のテスト"""
        # 勝率60%、平均利益5%、平均損失3%のケース
        fraction = self.sizer.calculate_kelly_fraction(0.6, 0.05, -0.03)
        self.assertGreaterEqual(fraction, 0)
        self.assertLessEqual(fraction, 0.25)  # max_kelly_fractionのデフォルト値
    
    def test_position_size_calculation(self):
        """ポジションサイズ計算のテスト"""
        result = self.sizer.calculate_position_size(
            capital=1000000,
            price=3000,
            win_rate=0.6,
            avg_win_rate=0.05,
            avg_loss_rate=-0.03
        )
        self.assertIsInstance(result, PositionSizeResult)
        self.assertGreaterEqual(result.shares, 0)
        self.assertGreaterEqual(result.position_value, 0)


class TestATRBasedStopLoss(unittest.TestCase):
    """ATRベースストップロスのテスト"""
    
    def setUp(self):
        self.atr_stop = ATRBasedStopLoss()
        # テスト用OHLCデータ
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        prices = 3000 + np.cumsum(np.random.normal(0, 20, 30))
        self.df = pd.DataFrame({
            'Close': prices,
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.99 for p in prices]
        }, index=dates)
    
    def test_atr_calculation(self):
        """ATR計算のテスト"""
        atr_values = self.atr_stop.calculate_atr(self.df)
        self.assertIsInstance(atr_values, pd.Series)
        self.assertEqual(len(atr_values), len(self.df))
        self.assertGreater(atr_values.iloc[-1], 0)
    
    def test_stop_loss_calculation(self):
        """ストップロス価格計算のテスト"""
        current_price = self.df['Close'].iloc[-1]
        stop_price, atr_value = self.atr_stop.calculate_stop_loss(current_price, self.df, "long")
        self.assertLess(stop_price, current_price)  # ロングの場合は価格より下
        self.assertGreater(atr_value, 0)


class TestSmartExitStrategy(unittest.TestCase):
    """スマート出口戦略のテスト"""
    
    def setUp(self):
        self.exit_strategy = SmartExitStrategy()
        # テスト用OHLCデータ
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        prices = 3000 + np.cumsum(np.random.normal(0, 20, 30))
        self.df = pd.DataFrame({
            'Close': prices,
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.99 for p in prices]
        }, index=dates)
    
    def test_exit_strategy_calculation(self):
        """出口戦略計算のテスト"""
        result = self.exit_strategy.calculate_exit_strategy(
            entry_price=3000,
            df=self.df,
            direction="long",
            market_condition="normal"
        )
        self.assertIsInstance(result, ExitStrategyResult)
        self.assertLess(result.stop_loss_price, 3000)  # ストップはエントリー価格より低い
        self.assertGreater(result.take_profit_price, 3000)  # 利確はエントリー価格より高い
        self.assertEqual(result.trailing_stop_enabled, True)


class TestComprehensiveRiskManager(unittest.TestCase):
    """包括的リスクマネージャーのテスト"""
    
    def setUp(self):
        self.comp_risk_manager = ComprehensiveRiskManager(initial_capital=1000000)
        # テスト用市場データ
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        prices = 3000 + np.cumsum(np.random.normal(0, 20, 30))
        self.market_data = {
            'Close': prices,
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.99 for p in prices]
        }
    
    def test_comprehensive_risk_calculation(self):
        """包括的リスク計算のテスト"""
        assessment = self.comp_risk_manager.calculate_comprehensive_risk(
            symbol="7203",
            price=3200,
            market_data=self.market_data
        )
        self.assertIsInstance(assessment, ComprehensiveRiskAssessment)
        self.assertIsNotNone(assessment.basic_metrics)
        self.assertIsNotNone(assessment.advanced_metrics)
        self.assertIsNotNone(assessment.position_size)
        self.assertIsNotNone(assessment.exit_strategy)
        self.assertIn(assessment.risk_level, ['low', 'medium', 'high', 'very_high'])
        self.assertGreaterEqual(assessment.risk_score, 0)
        self.assertLessEqual(assessment.risk_score, 1)


class TestRiskManagedPortfolio(unittest.TestCase):
    """リスク管理型ポートフォリオのテスト"""
    
    def setUp(self):
        self.portfolio = RiskManagedPortfolio(initial_capital=1000000)
        # テスト用市場データ
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        prices = 3000 + np.cumsum(np.random.normal(0, 20, 30))
        self.market_data = {
            'Close': prices,
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.99 for p in prices]
        }
    
    def test_safe_trade_execution(self):
        """安全な取引実行のテスト"""
        # 購入取引
        result = self.portfolio.execute_trade_safely(
            "7203",
            "buy",
            3200,
            self.market_data
        )
        if result:  # これはNoneになる可能性あり（リスク制限で）
            self.assertIn(result['symbol'], ['7203', 'missing'])
            if result['action'] == 'buy':
                self.assertGreaterEqual(result['shares'], 0)
                self.assertIn(result['risk_level'], ['low', 'medium', 'high', 'very_high'])
        
        # 売却取引
        result = self.portfolio.execute_trade_safely(
            "7203",
            "sell",
            3250,
            self.market_data
        )
        if result:
            self.assertEqual(result['action'], 'sell')
            self.assertGreaterEqual(result['shares'], 0)


def run_all_tests():
    """すべてのテストを実行"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    print("リスク管理コンポーネントのテストを実行します...")
    run_all_tests()