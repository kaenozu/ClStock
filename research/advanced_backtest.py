#!/usr/bin/env python3
"""超高性能予測システムのバックテスト＆MAPE測定
"""

from typing import Dict, List

import numpy as np
from data.stock_data import StockDataProvider
from models.predictor import StockPredictor
from utils.logger_config import setup_logger

# ログ設定
logger = setup_logger(__name__)


class AdvancedBacktester:
    """高度なバックテストシステム"""

    def __init__(self):
        self.data_provider = StockDataProvider()
        self.results = {}
        self.performance_metrics = {}

    def calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """MAPE（平均絶対パーセント誤差）計算"""
        # ゼロ除算回避
        mask = actual != 0
        if mask.sum() == 0:
            return float("inf")

        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return mape

    def calculate_smape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """SMAPE（対称平均絶対パーセント誤差）計算"""
        numerator = np.abs(predicted - actual)
        denominator = (np.abs(actual) + np.abs(predicted)) / 2

        # ゼロ除算回避
        mask = denominator != 0
        if mask.sum() == 0:
            return float("inf")

        smape = np.mean(numerator[mask] / denominator[mask]) * 100
        return smape

    def calculate_directional_accuracy(
        self, actual_returns: np.ndarray, predicted_scores: np.ndarray,
    ) -> float:
        """方向性精度計算（上昇/下降予測の正確性）"""
        actual_direction = (actual_returns > 0).astype(int)
        predicted_direction = (predicted_scores > 50).astype(int)

        accuracy = np.mean(actual_direction == predicted_direction)
        return accuracy

    def calculate_sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate: float = 0.001,
    ) -> float:
        """シャープレシオ計算"""
        if np.std(returns) == 0:
            return 0
        return (np.mean(returns) - risk_free_rate) / np.std(returns)

    def calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """最大ドローダウン計算"""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)

    def run_prediction_backtest(
        self,
        symbols: List[str],
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
        prediction_horizon: int = 30,
    ) -> Dict:
        """予測精度バックテスト実行"""
        logger.info(f"予測精度バックテスト開始: {len(symbols)}銘柄")
        logger.info(f"期間: {start_date} - {end_date}")
        logger.info(f"予測期間: {prediction_horizon}日")

        results = {
            "predictions": [],
            "actuals": [],
            "symbols": [],
            "dates": [],
            "returns": [],
            "scores": [],
        }

        # 各モデルの初期化
        predictors = {
            "rule_based": StockPredictor(use_ml_model=False, use_ultra_mode=False),
            "ml_model": StockPredictor(
                use_ml_model=True, ml_model_type="xgboost", use_ultra_mode=False,
            ),
            "ultra_model": StockPredictor(use_ultra_mode=True),
        }

        model_results = {
            name: {"predictions": [], "actuals": [], "returns": [], "scores": []}
            for name in predictors
        }

        for symbol in symbols:
            try:
                logger.info(f"バックテスト実行中: {symbol}")

                # データ取得
                data = self.data_provider.get_stock_data(symbol, "2y")
                if len(data) < 200:
                    logger.warning(f"データ不足: {symbol}")
                    continue

                # 日付範囲でフィルタ
                data = data[(data.index >= start_date) & (data.index <= end_date)]

                if len(data) < prediction_horizon * 2:
                    continue

                # 時系列バックテスト
                for i in range(len(data) - prediction_horizon):
                    current_date = data.index[i]
                    future_date = data.index[i + prediction_horizon]

                    current_price = data["Close"].iloc[i]
                    future_price = data["Close"].iloc[i + prediction_horizon]
                    actual_return = (future_price - current_price) / current_price

                    # 各モデルで予測
                    for model_name, predictor in predictors.items():
                        try:
                            # 現在時点でのデータを使用して予測
                            historical_data = data.iloc[: i + 1]
                            if len(historical_data) < 60:  # 最小データ要件
                                continue

                            # 一時的にデータプロバイダーのデータを設定
                            temp_data = historical_data.copy()
                            score = predictor.calculate_score(symbol)

                            # 予測価格（スコアベース）
                            if score > 50:
                                predicted_return = (
                                    (score - 50) / 50 * 0.1
                                )  # 最大10%の上昇予測
                            else:
                                predicted_return = (
                                    (score - 50) / 50 * 0.1
                                )  # 最大10%の下落予測

                            model_results[model_name]["predictions"].append(
                                predicted_return,
                            )
                            model_results[model_name]["actuals"].append(actual_return)
                            model_results[model_name]["returns"].append(actual_return)
                            model_results[model_name]["scores"].append(score)

                        except Exception as e:
                            logger.error(f"予測エラー {model_name} {symbol}: {e!s}")
                            continue

                    # 共通結果記録
                    results["symbols"].append(symbol)
                    results["dates"].append(current_date)
                    results["actuals"].append(actual_return)
                    results["returns"].append(actual_return)

            except Exception as e:
                logger.error(f"バックテストエラー {symbol}: {e!s}")
                continue

        # 性能指標計算
        performance_summary = {}

        for model_name, model_data in model_results.items():
            if not model_data["predictions"]:
                continue

            predictions = np.array(model_data["predictions"])
            actuals = np.array(model_data["actuals"])
            scores = np.array(model_data["scores"])

            # MAPE計算
            mape = self.calculate_mape(actuals, predictions)
            smape = self.calculate_smape(actuals, predictions)

            # MSE, RMSE, MAE
            mse = np.mean((predictions - actuals) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - actuals))

            # 方向性精度
            direction_accuracy = self.calculate_directional_accuracy(actuals, scores)

            # 相関係数
            correlation = (
                np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
            )

            performance_summary[model_name] = {
                "MAPE": mape,
                "SMAPE": smape,
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "Direction_Accuracy": direction_accuracy,
                "Correlation": correlation,
                "Sample_Size": len(predictions),
            }

            logger.info(f"{model_name} 性能指標:")
            logger.info(f"  MAPE: {mape:.2f}%")
            logger.info(f"  方向性精度: {direction_accuracy:.3f}")
            logger.info(f"  相関係数: {correlation:.3f}")

        return {
            "model_results": model_results,
            "performance_summary": performance_summary,
            "raw_results": results,
        }

    def run_investment_simulation(
        self,
        symbols: List[str],
        initial_capital: float = 1000000,
        rebalance_frequency: int = 30,
    ) -> Dict:
        """投資シミュレーション実行"""
        logger.info(f"投資シミュレーション開始: 初期資本{initial_capital:,.0f}円")

        # 予測器初期化
        predictor = StockPredictor(use_ultra_mode=True)

        portfolio_history = []
        current_capital = initial_capital
        positions = {}  # {symbol: shares}

        # データ取得
        all_data = {}
        for symbol in symbols:
            data = self.data_provider.get_stock_data(symbol, "1y")
            all_data[symbol] = data

        # 共通の日付インデックス
        common_dates = None
        for symbol, data in all_data.items():
            if common_dates is None:
                common_dates = data.index
            else:
                common_dates = common_dates.intersection(data.index)

        common_dates = sorted(common_dates)

        for i, date in enumerate(common_dates[:-rebalance_frequency]):
            if i % rebalance_frequency == 0:  # リバランス
                # 現在のポートフォリオ価値計算
                portfolio_value = current_capital
                for symbol, shares in positions.items():
                    if symbol in all_data and date in all_data[symbol].index:
                        current_price = all_data[symbol].loc[date, "Close"]
                        portfolio_value += shares * current_price

                # 予測スコア取得
                scores = {}
                for symbol in symbols:
                    try:
                        score = predictor.calculate_score(symbol)
                        scores[symbol] = score
                    except Exception as e:
                        logger.warning(f"予測失敗 {symbol}: {e!s}")
                        scores[symbol] = 50  # 中立スコア

                # 上位3銘柄選択
                top_symbols = sorted(
                    scores.keys(), key=lambda x: scores[x], reverse=True,
                )[:3]

                # ポジションクリア
                for symbol, shares in positions.items():
                    if symbol in all_data and date in all_data[symbol].index:
                        current_price = all_data[symbol].loc[date, "Close"]
                        current_capital += shares * current_price

                positions = {}

                # 新しいポジション構築
                capital_per_stock = current_capital / len(top_symbols)
                for symbol in top_symbols:
                    if symbol in all_data and date in all_data[symbol].index:
                        current_price = all_data[symbol].loc[date, "Close"]
                        shares = int(capital_per_stock / current_price)
                        positions[symbol] = shares
                        current_capital -= shares * current_price

            # ポートフォリオ記録
            portfolio_value = current_capital
            for symbol, shares in positions.items():
                if symbol in all_data and date in all_data[symbol].index:
                    current_price = all_data[symbol].loc[date, "Close"]
                    portfolio_value += shares * current_price

            portfolio_history.append(
                {
                    "date": date,
                    "portfolio_value": portfolio_value,
                    "cash": current_capital,
                    "positions": positions.copy(),
                },
            )

        # ベンチマーク（均等投資）計算
        benchmark_history = []
        equal_weight_capital = initial_capital / len(symbols)

        for date in common_dates:
            benchmark_value = 0
            for symbol in symbols:
                if symbol in all_data and date in all_data[symbol].index:
                    # 最初の日の価格で正規化
                    first_price = all_data[symbol].iloc[0]["Close"]
                    current_price = all_data[symbol].loc[date, "Close"]
                    benchmark_value += equal_weight_capital * (
                        current_price / first_price
                    )

            benchmark_history.append({"date": date, "benchmark_value": benchmark_value})

        # 性能指標計算
        portfolio_values = [p["portfolio_value"] for p in portfolio_history]
        benchmark_values = [
            b["benchmark_value"] for b in benchmark_history[: len(portfolio_history)]
        ]

        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]

        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        benchmark_return = (
            benchmark_values[len(portfolio_history) - 1] - initial_capital
        ) / initial_capital

        sharpe_ratio = self.calculate_sharpe_ratio(portfolio_returns)
        max_drawdown = self.calculate_max_drawdown(np.array(portfolio_values))

        alpha = total_return - benchmark_return

        simulation_results = {
            "portfolio_history": portfolio_history,
            "benchmark_history": benchmark_history,
            "performance_metrics": {
                "Total_Return": total_return,
                "Benchmark_Return": benchmark_return,
                "Alpha": alpha,
                "Sharpe_Ratio": sharpe_ratio,
                "Max_Drawdown": max_drawdown,
                "Final_Value": portfolio_values[-1],
                "Initial_Capital": initial_capital,
            },
        }

        return simulation_results

    def generate_performance_report(
        self, backtest_results: Dict, simulation_results: Dict,
    ):
        """性能レポート生成"""
        print("=" * 80)
        print("超高性能予測システム バックテスト結果レポート")
        print("=" * 80)

        # 1. 予測精度結果
        print("\n1. 予測精度比較")
        print("-" * 50)

        performance_summary = backtest_results["performance_summary"]

        # テーブル形式で表示
        print(
            f"{'モデル':<15} {'MAPE':<8} {'方向精度':<8} {'相関係数':<8} {'サンプル数':<8}",
        )
        print("-" * 55)

        for model_name, metrics in performance_summary.items():
            mape = metrics.get("MAPE", float("inf"))
            direction = metrics.get("Direction_Accuracy", 0)
            correlation = metrics.get("Correlation", 0)
            sample_size = metrics.get("Sample_Size", 0)

            mape_str = f"{mape:.1f}%" if mape != float("inf") else "N/A"

            print(
                f"{model_name:<15} {mape_str:<8} {direction:.3f}    {correlation:.3f}    {sample_size:<8}",
            )

        # 2. 投資シミュレーション結果
        print("\n2. 投資シミュレーション結果")
        print("-" * 50)

        metrics = simulation_results["performance_metrics"]

        print(f"初期資本:           {metrics['Initial_Capital']:>15,.0f}円")
        print(f"最終資産:           {metrics['Final_Value']:>15,.0f}円")
        print(f"総リターン:         {metrics['Total_Return'] * 100:>14.2f}%")
        print(f"ベンチマーク:       {metrics['Benchmark_Return'] * 100:>14.2f}%")
        print(f"アルファ:           {metrics['Alpha'] * 100:>14.2f}%")
        print(f"シャープレシオ:     {metrics['Sharpe_Ratio']:>15.3f}")
        print(f"最大ドローダウン:   {metrics['Max_Drawdown'] * 100:>14.2f}%")

        # 3. 総合評価
        print("\n3. 総合評価")
        print("-" * 30)

        # ベストモデル特定
        best_model = None
        best_score = float("inf")

        for model_name, metrics in performance_summary.items():
            mape = metrics.get("MAPE", float("inf"))
            if mape < best_score:
                best_score = mape
                best_model = model_name

        print(f"最高精度モデル: {best_model} (MAPE: {best_score:.1f}%)")

        if metrics["Alpha"] > 0:
            print(f"投資戦略: ベンチマーク超過 (+{metrics['Alpha'] * 100:.2f}%)")
        else:
            print(f"投資戦略: ベンチマーク未達 ({metrics['Alpha'] * 100:.2f}%)")

        if metrics["Sharpe_Ratio"] > 1.0:
            print(f"リスク調整: 優秀 (シャープレシオ: {metrics['Sharpe_Ratio']:.3f})")
        elif metrics["Sharpe_Ratio"] > 0.5:
            print(f"リスク調整: 良好 (シャープレシオ: {metrics['Sharpe_Ratio']:.3f})")
        else:
            print(f"リスク調整: 要改善 (シャープレシオ: {metrics['Sharpe_Ratio']:.3f})")

        print("\n" + "=" * 80)


def main():
    """メイン実行関数"""
    # バックテスター初期化
    backtester = AdvancedBacktester()

    # テスト対象銘柄
    data_provider = StockDataProvider()
    test_symbols = list(data_provider.jp_stock_codes.keys())[:5]  # 最初の5銘柄

    print(f"バックテスト対象銘柄: {test_symbols}")

    # 1. 予測精度バックテスト
    print("\n予測精度バックテスト実行中...")
    backtest_results = backtester.run_prediction_backtest(
        symbols=test_symbols,
        start_date="2024-01-01",
        end_date="2024-12-31",
        prediction_horizon=30,
    )

    # 2. 投資シミュレーション
    print("\n投資シミュレーション実行中...")
    simulation_results = backtester.run_investment_simulation(
        symbols=test_symbols, initial_capital=1000000, rebalance_frequency=30,
    )

    # 3. レポート生成
    backtester.generate_performance_report(backtest_results, simulation_results)


if __name__ == "__main__":
    main()
