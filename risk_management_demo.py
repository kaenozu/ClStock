"""
ClStock リスク管理システム デモンストレーション
VaR、動的ポジショニング、ストップロスなど高度なリスク管理機能のデモ
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.risk_management import DynamicRiskManager
from trading.position_sizing import AdaptivePositionSizer
from trading.stop_loss_taking import SmartExitStrategy
from trading.comprehensive_risk_system import RiskManagedPortfolio


def demo_basic_risk_management():
    """基本リスク管理機能のデモ"""
    print("="*60)
    print("1. 基本リスク管理機能デモ")
    print("="*60)
    
    # リスクマネージャーの初期化
    risk_manager = DynamicRiskManager(initial_capital=1000000)
    
    # サンプル市場データ（価格リターン）
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.02, 252))  # 年間252日分のリターン
    
    # ポートフォリオ価値履歴をシミュレーション
    for i in range(30):
        value = 1000000 * (1 + np.random.normal(0.001, 0.015, 1)[0] * i)
        risk_manager.risk_manager.historical_portfolio_values.append(value)
    
    # 保有銘柄を追加
    risk_manager.risk_manager.update_position("7203", 100, 3000)
    
    # リスク指標を計算
    metrics = risk_manager.risk_manager.calculate_risk_metrics()
    print(f"ポートフォリオ価値: {metrics.portfolio_value:,.0f}円")
    print(f"現在の損益: {metrics.current_pnl:,.0f}円")
    print(f"VaR 95%: {metrics.var_95:.4f} ({metrics.var_95*100:.2f}%)")
    print(f"VaR 99%: {metrics.var_99:.4f} ({metrics.var_99*100:.2f}%)")
    print(f"最大ドローダウン: {metrics.max_drawdown:.4f} ({metrics.max_drawdown*100:.2f}%)")
    print(f"シャープレシオ: {metrics.sharpe_ratio:.4f}")
    
    # ポジションサイズ計算のデモ
    print(f"\nポジションサイズ計算デモ:")
    position_size = risk_manager.position_sizer.calculate_position_size(
        current_capital=metrics.portfolio_value,
        win_rate=0.6,  # 勝率60%
        avg_win_rate=0.05,  # 平均利益率5%
        avg_loss_rate=-0.03,  # 平均損失率3%
        price=3100  # 現在価格
    )
    print(f"推奨株数: {position_size}株")


def demo_position_sizing():
    """動的ポジショニングのデモ"""
    print("\n" + "="*60)
    print("2. 動的ポジショニングデモ")
    print("="*60)
    
    sizer = AdaptivePositionSizer(initial_capital=1000000)
    
    # 異なる市場条件でのテスト
    conditions = [
        {"name": "高勝率条件", "win_rate": 0.7, "avg_win": 0.06, "avg_loss": -0.02},
        {"name": "普通条件", "win_rate": 0.55, "avg_win": 0.04, "avg_loss": -0.03},
        {"name": "低勝率条件", "win_rate": 0.4, "avg_win": 0.03, "avg_loss": -0.04}
    ]
    
    for condition in conditions:
        result = sizer.calculate_optimal_position_size(
            symbol="7203",
            price=3000,
            win_rate=condition["win_rate"],
            avg_win_rate=condition["avg_win"],
            avg_loss_rate=condition["avg_loss"],
            volatility=0.25,
            correlation_with_portfolio=0.2,
            strategy_confidence=0.8
        )
        print(f"{condition['name']}: {result.shares}株, 価値: {result.position_value:,.0f}円")


def demo_exit_strategy():
    """出口戦略のデモ"""
    print("\n" + "="*60)
    print("3. スマート出口戦略デモ")
    print("="*60)
    
    exit_strategy = SmartExitStrategy()
    
    # テスト用のOHLCデータを作成
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D')
    base_price = 3000
    prices = [base_price * (1 + np.random.normal(0, 0.01)) for _ in range(30)]
    df = pd.DataFrame({
        'Close': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    }, index=dates)
    
    # 長期市場条件での出口戦略計算
    entry_price = df['Close'].iloc[-1]
    result = exit_strategy.calculate_exit_strategy(
        entry_price=entry_price,
        df=df,
        direction="long",
        market_condition="normal"
    )
    
    print(f"エントリー価格: {entry_price:.0f}円")
    print(f"損切り価格: {result.stop_loss_price:.0f}円")
    print(f"利確価格: {result.take_profit_price:.0f}円")
    print(f"トレーリング有効: {result.trailing_stop_enabled}")
    print(f"戦略タイプ: {result.exit_strategy_type}")
    
    # 短期トレードでの出口戦略計算
    result_short = exit_strategy.calculate_exit_strategy(
        entry_price=entry_price,
        df=df,
        direction="long",
        market_condition="high_volatility"
    )
    
    print(f"\n高ボラティリティ条件:")
    print(f"損切り価格: {result_short.stop_loss_price:.0f}円")
    print(f"利確価格: {result_short.take_profit_price:.0f}円")


def demo_comprehensive_system():
    """包括的リスク管理システムのデモ"""
    print("\n" + "="*60)
    print("4. 包括的リスク管理システムデモ")
    print("="*60)
    
    # リスク管理型ポートフォリオの初期化
    portfolio = RiskManagedPortfolio(initial_capital=1000000)
    
    # テスト用市場データ
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D')
    prices = [3000 * (1 + np.random.normal(0, 0.015)) for _ in range(30)]
    market_data = {
        'Close': np.array(prices),
        'High': np.array([p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]),
        'Low': np.array([p * (1 - abs(np.random.normal(0, 0.01))) for p in prices])
    }
    
    print(f"初期資本: {portfolio.risk_manager.current_capital:,.0f}円")
    
    # 購入取引シミュレーション
    buy_result = portfolio.execute_trade_safely(
        symbol="7203",
        action="buy",
        price=3100,  # 現在価格
        market_data=market_data
    )
    
    if buy_result:
        print(f"購入成功: {buy_result['shares']}株, 価値: {buy_result['value']:,.0f}円")
        print(f"リスクレベル: {buy_result['risk_level']}, スコア: {buy_result['risk_score']:.3f}")
        print(f"提案損切り: {buy_result['stop_loss']:.0f}円, 利確: {buy_result['take_profit']:.0f}円")
    else:
        print("購入はリスク制限により制限されました")
    
    # ポートフォリオのリスク評価を取得
    risk_assessment = portfolio.risk_manager.calculate_comprehensive_risk(
        "7203", 3100, market_data=market_data
    )
    
    print(f"\n包括的リスク評価:")
    print(f"基本VaR 95%: {risk_assessment.basic_metrics.var_95:.4f}")
    print(f"高度VaR 99%: {risk_assessment.advanced_metrics.var_99:.4f}")
    print(f"最大ドローダウン: {risk_assessment.basic_metrics.max_drawdown:.4f}")
    print(f"リスクレベル: {risk_assessment.risk_level}, スコア: {risk_assessment.risk_score:.3f}")


def run_full_demo():
    """完全なデモを実行"""
    print("ClStock リスク管理システム デモンストレーション")
    print("高度なリスク管理アルゴリズムの実装と機能を紹介します")
    print()
    
    # 個別のデモを実行
    demo_basic_risk_management()
    demo_position_sizing()
    demo_exit_strategy()
    demo_comprehensive_system()
    
    print("\n" + "="*60)
    print("デモ完了！")
    print("ClStockのリスク管理システムは、")
    print("- VaR (Value at Risk) と ES (Expected Shortfall) 計算")
    print("- Kelly Criterion とリスクベースの動的ポジショニング")
    print("- ATRベース損切りとトレーリングストップ")
    print("- 相関リスク管理とストレステスト")
    print("- 市場状況に基づくリスクパラメータ調整")
    print("を実装しています。")
    print("="*60)


if __name__ == "__main__":
    run_full_demo()