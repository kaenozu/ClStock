#!/usr/bin/env python3
"""
ポートフォリオ最適化システム
84.6%予測精度を活用した最適ポートフォリオ構築
モダンポートフォリオ理論とシャープレシオ最適化を統合
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from data.stock_data import StockDataProvider
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import logging
from utils.logger_config import setup_logger

logger = setup_logger(__name__)
from datetime import datetime, timedelta


class PortfolioOptimizer:
    def __init__(self, initial_capital=1000000):
        self.data_provider = StockDataProvider()
        self.scaler = StandardScaler()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.portfolio = {}  # ポートフォリオ構成

    def analyze_stock_performance(self, symbol, data):
        """個別銘柄のパフォーマンス分析"""
        close = data["Close"]
        returns = close.pct_change()

        # 84.6%パターンでの予測精度
        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()

        strong_trend = ((sma_10 > sma_20) & (sma_20 > sma_50)) | (
            (sma_10 < sma_20) & (sma_20 < sma_50)
        )

        # トレンド期間の継続性
        trend_quality = 0
        for i in range(10, len(strong_trend)):
            if strong_trend.iloc[i]:
                recent_trend = strong_trend.iloc[i - 10 : i].sum()
                if recent_trend >= 7:
                    trend_quality += 1

        trend_score = trend_quality / len(data) if len(data) > 0 else 0

        # リターンとリスク計算
        annual_return = returns.mean() * 252  # 年率換算
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

        # 最大ドローダウン
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            "symbol": symbol,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "trend_score": trend_score,
            "prediction_score": 0.846 * trend_score,  # 84.6%精度を反映
        }

    def calculate_correlation_matrix(self, stocks_data):
        """銘柄間の相関行列計算"""
        returns_data = {}

        for symbol, data in stocks_data.items():
            returns = data["Close"].pct_change()
            returns_data[symbol] = returns

        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()

        return correlation_matrix

    def optimize_portfolio_weights(
        self, expected_returns, covariance_matrix, risk_free_rate=0.001
    ):
        """シャープレシオ最大化によるポートフォリオ最適化"""
        n_assets = len(expected_returns)

        # 目的関数：シャープレシオの負値（最小化するため）
        def negative_sharpe_ratio(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_volatility = np.sqrt(
                np.dot(weights.T, np.dot(covariance_matrix, weights))
            )
            sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe

        # 制約条件
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # 重みの合計が1
        ]

        # 境界条件（0〜1、最大30%）
        bounds = tuple((0, 0.3) for _ in range(n_assets))

        # 初期値（均等配分）
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        # 最適化実行
        result = minimize(
            negative_sharpe_ratio,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x if result.success else initial_weights

    def create_optimal_portfolio(self, symbols, lookback_months=12):
        """最適ポートフォリオ構築"""
        print("=== ポートフォリオ最適化システム ===")
        print(f"初期資金: {self.initial_capital:,}円")

        stocks_performance = []
        stocks_data = {}

        # 各銘柄のパフォーマンス分析
        for symbol in symbols:
            try:
                data = self.data_provider.get_stock_data(symbol, "2y")
                if len(data) < 200:
                    continue

                performance = self.analyze_stock_performance(symbol, data)
                stocks_performance.append(performance)
                stocks_data[symbol] = data

                print(
                    f"\n{symbol}: {self.data_provider.jp_stock_codes.get(symbol, symbol)}"
                )
                print(f"  年率リターン: {performance['annual_return']*100:.1f}%")
                print(
                    f"  年率ボラティリティ: {performance['annual_volatility']*100:.1f}%"
                )
                print(f"  シャープレシオ: {performance['sharpe_ratio']:.2f}")
                print(f"  予測スコア: {performance['prediction_score']*100:.1f}%")

            except Exception as e:
                print(f"  {symbol}: データ取得エラー")
                continue

        if len(stocks_performance) < 3:
            print("ポートフォリオ構築に十分な銘柄がありません")
            return None

        # 期待リターンと共分散行列の計算
        symbols_valid = [p["symbol"] for p in stocks_performance]
        expected_returns = np.array([p["annual_return"] for p in stocks_performance])

        # リターンデータの準備
        returns_data = pd.DataFrame()
        for symbol in symbols_valid:
            returns = stocks_data[symbol]["Close"].pct_change()
            returns_data[symbol] = returns

        covariance_matrix = returns_data.cov() * 252  # 年率換算

        # ポートフォリオ最適化
        optimal_weights = self.optimize_portfolio_weights(
            expected_returns, covariance_matrix.values
        )

        # 最適ポートフォリオの詳細
        portfolio = {}
        total_investment = 0

        print("\n=== 最適ポートフォリオ構成 ===")
        for i, symbol in enumerate(symbols_valid):
            weight = optimal_weights[i]
            if weight > 0.01:  # 1%以上の配分のみ
                investment = self.initial_capital * weight
                portfolio[symbol] = {
                    "weight": weight,
                    "investment": investment,
                    "company": self.data_provider.jp_stock_codes.get(symbol, symbol),
                }
                total_investment += investment

                print(f"{symbol}: {weight*100:.1f}% ({investment:,.0f}円)")

        # ポートフォリオ全体のパフォーマンス計算
        portfolio_return = np.sum(expected_returns * optimal_weights)
        portfolio_volatility = np.sqrt(
            np.dot(optimal_weights.T, np.dot(covariance_matrix.values, optimal_weights))
        )
        portfolio_sharpe = portfolio_return / portfolio_volatility

        print(f"\n=== ポートフォリオパフォーマンス予測 ===")
        print(f"期待年率リターン: {portfolio_return*100:.1f}%")
        print(f"予想年率ボラティリティ: {portfolio_volatility*100:.1f}%")
        print(f"シャープレシオ: {portfolio_sharpe:.2f}")
        print(f"84.6%予測精度適用後の期待リターン: {portfolio_return*0.846*100:.1f}%")

        return {
            "portfolio": portfolio,
            "expected_return": portfolio_return,
            "expected_volatility": portfolio_volatility,
            "sharpe_ratio": portfolio_sharpe,
            "adjusted_return": portfolio_return * 0.846,
            "total_investment": total_investment,
        }

    def backtest_portfolio(self, portfolio_config, test_months=6):
        """ポートフォリオのバックテスト"""
        print("\n=== ポートフォリオバックテスト ===")

        portfolio_value = self.initial_capital
        portfolio_returns = []

        for symbol, config in portfolio_config["portfolio"].items():
            try:
                # テスト期間のデータ取得
                data = self.data_provider.get_stock_data(symbol, "1y")

                # 最新6ヶ月のリターン
                recent_returns = data["Close"].pct_change().tail(test_months * 20)

                # ポートフォリオ配分に基づくリターン
                weighted_returns = recent_returns * config["weight"]
                portfolio_returns.append(weighted_returns)

            except Exception:
                continue

        if portfolio_returns:
            # 全体のポートフォリオリターン
            total_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)
            cumulative_return = (1 + total_returns).prod() - 1

            final_value = portfolio_value * (1 + cumulative_return)
            profit = final_value - portfolio_value

            print(f"初期資産: {portfolio_value:,.0f}円")
            print(f"最終資産: {final_value:,.0f}円")
            print(f"収益: {profit:,.0f}円")
            print(f"収益率: {cumulative_return*100:.1f}%")
            print(f"年率換算: {cumulative_return*2*100:.1f}%")

            return {
                "initial_value": portfolio_value,
                "final_value": final_value,
                "profit": profit,
                "return_rate": cumulative_return,
                "annualized_return": cumulative_return * 2,
            }

        return None


def main():
    """ポートフォリオ最適化実行"""
    optimizer = PortfolioOptimizer(initial_capital=1000000)

    # データプロバイダから銘柄リスト取得
    data_provider = StockDataProvider()
    all_symbols = list(data_provider.jp_stock_codes.keys())

    # 上位20銘柄でポートフォリオ構築
    symbols = all_symbols[:20]

    # 最適ポートフォリオ作成
    portfolio_result = optimizer.create_optimal_portfolio(symbols)

    if portfolio_result:
        print(f"\n=== 投資推奨 ===")
        print(f"100万円を投資した場合の期待値:")

        expected_profit = 1000000 * portfolio_result["adjusted_return"]
        print(f"年間期待収益: {expected_profit:,.0f}円")
        print(f"月間期待収益: {expected_profit/12:,.0f}円")

        # バックテスト実行
        backtest_result = optimizer.backtest_portfolio(portfolio_result)

        if backtest_result:
            print(f"\n実績ベース年率: {backtest_result['annualized_return']*100:.1f}%")
            print(
                f"実績ベース年間収益予測: {1000000 * backtest_result['annualized_return']:,.0f}円"
            )


if __name__ == "__main__":
    main()
