#!/usr/bin/env python3
"""改善されたリターン率予測バックテスト（MAPE最適化）
"""

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from data.stock_data import StockDataProvider
from models.predictor import StockPredictor
from utils.logger_config import setup_logger

# ログ設定
logger = setup_logger(__name__)


class ReturnRateBacktester:
    """改善されたリターン率予測バックテストシステム"""

    def __init__(self, prediction_days: int = 5):
        self.data_provider = StockDataProvider()
        self.prediction_days = prediction_days
        self.results = {}
        self.performance_metrics = {}

    def calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """MAPE（平均絶対パーセント誤差）計算"""
        # リターン率の場合、分母が0になる可能性を考慮
        mask = np.abs(actual) > 0.001  # 0.1%以上の変動のみ
        if mask.sum() == 0:
            return float("inf")

        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return mape

    def calculate_smape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """SMAPE（対称平均絶対パーセント誤差）計算"""
        numerator = np.abs(predicted - actual)
        denominator = (np.abs(actual) + np.abs(predicted)) / 2

        # ゼロ除算回避
        mask = denominator > 0.0001
        if mask.sum() == 0:
            return float("inf")

        smape = np.mean(numerator[mask] / denominator[mask]) * 100
        return smape

    def calculate_directional_accuracy(
        self, actual: np.ndarray, predicted: np.ndarray,
    ) -> float:
        """方向性精度計算（上昇/下降の予測精度）"""
        actual_direction = np.sign(actual)
        predicted_direction = np.sign(predicted)

        # 中立（±0.2%以内）を除外
        significant_moves = np.abs(actual) > 0.002
        if significant_moves.sum() == 0:
            return 0.0

        correct_direction = (
            actual_direction[significant_moves]
            == predicted_direction[significant_moves]
        ).mean()
        return correct_direction * 100

    def walk_forward_backtest(
        self,
        symbols: List[str],
        start_date: str = "2023-01-01",
        end_date: str = "2024-01-01",
    ) -> Dict:
        """ウォークフォワード分析による時系列バックテスト"""
        results = {
            "predictions": [],
            "actuals": [],
            "symbols": [],
            "dates": [],
            "errors": [],
        }

        # 日付範囲設定
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # 予測器初期化
        predictor = StockPredictor(prediction_days=self.prediction_days)

        current_date = start_dt
        total_predictions = 0
        successful_predictions = 0

        print(f"ウォークフォワードバックテスト開始: {len(symbols)}銘柄")
        print(f"期間: {start_date} → {end_date}")
        print(f"予測期間: {self.prediction_days}日")
        print("-" * 50)

        while current_date < end_dt - timedelta(days=self.prediction_days + 5):
            # 各銘柄について予測
            for symbol in symbols[:10]:  # 最初の10銘柄でテスト
                try:
                    # 予測実行
                    predicted_return = predictor.predict_return_rate(symbol)

                    # 実際のリターン取得
                    actual_return = self._get_actual_return(
                        symbol, current_date, self.prediction_days,
                    )

                    if actual_return is not None:
                        results["predictions"].append(predicted_return)
                        results["actuals"].append(actual_return)
                        results["symbols"].append(symbol)
                        results["dates"].append(current_date)
                        results["errors"].append(abs(predicted_return - actual_return))

                        successful_predictions += 1

                    total_predictions += 1

                except Exception as e:
                    logger.warning(
                        f"Prediction failed for {symbol} at {current_date}: {e!s}",
                    )
                    continue

            # 次の週に進む（週次バックテスト）
            current_date += timedelta(days=7)

            if successful_predictions > 0 and successful_predictions % 50 == 0:
                print(f"進捗: {successful_predictions}件の予測完了")

        print(f"バックテスト完了: {successful_predictions}/{total_predictions}件成功")
        return results

    def _get_actual_return(self, symbol: str, start_date: datetime, days: int) -> float:
        """実際のリターン取得"""
        try:
            # 十分な期間のデータを取得
            data = self.data_provider.get_stock_data(symbol, "1y")
            if data.empty:
                return None

            # タイムゾーンを統一（UTC）
            data.index = pd.to_datetime(data.index).tz_localize(None)
            start_date_utc = pd.to_datetime(start_date).tz_localize(None)

            # 開始日に最も近いデータを見つける
            start_price_idx = data.index.get_indexer(
                [start_date_utc], method="nearest",
            )[0]

            # 終了日のインデックス計算
            end_idx = min(start_price_idx + days, len(data) - 1)

            if start_price_idx >= len(data) - 1 or end_idx <= start_price_idx:
                return None

            start_price = data.iloc[start_price_idx]["Close"]
            end_price = data.iloc[end_idx]["Close"]

            actual_return = (end_price - start_price) / start_price
            return actual_return

        except Exception as e:
            logger.error(f"Error getting actual return for {symbol}: {e!s}")
            return None

    def analyze_results(self, results: Dict) -> Dict:
        """結果分析とメトリクス計算"""
        if not results["predictions"] or len(results["predictions"]) == 0:
            return {
                "error": "No valid predictions",
                "mape": float("inf"),
                "smape": float("inf"),
                "directional_accuracy": 0.0,
                "mae": float("inf"),
                "rmse": float("inf"),
                "correlation": 0.0,
                "total_predictions": 0,
                "predictions_range": {"min": 0, "max": 0, "mean": 0, "std": 0},
                "actuals_range": {"min": 0, "max": 0, "mean": 0, "std": 0},
            }

        predictions = np.array(results["predictions"])
        actuals = np.array(results["actuals"])

        # 基本メトリクス
        mape = self.calculate_mape(actuals, predictions)
        smape = self.calculate_smape(actuals, predictions)
        directional_accuracy = self.calculate_directional_accuracy(actuals, predictions)

        # 統計情報
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

        # 相関係数
        correlation = (
            np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
        )

        analysis = {
            "mape": mape,
            "smape": smape,
            "directional_accuracy": directional_accuracy,
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation,
            "total_predictions": len(predictions),
            "predictions_range": {
                "min": float(predictions.min()),
                "max": float(predictions.max()),
                "mean": float(predictions.mean()),
                "std": float(predictions.std()),
            },
            "actuals_range": {
                "min": float(actuals.min()),
                "max": float(actuals.max()),
                "mean": float(actuals.mean()),
                "std": float(actuals.std()),
            },
        }

        return analysis

    def generate_report(self, analysis: Dict) -> str:
        """分析結果レポート生成"""
        if "error" in analysis:
            return f"""
========================================
改善されたリターン率予測バックテスト結果
========================================

エラー: {analysis["error"]}
有効な予測データが取得できませんでした。
タイムゾーンやデータ取得の問題を確認してください。
========================================
"""

        report = f"""
========================================
改善されたリターン率予測バックテスト結果
========================================

基本メトリクス:
  MAPE (平均絶対パーセント誤差): {analysis["mape"]:.2f}%
  SMAPE (対称MAPE): {analysis["smape"]:.2f}%
  方向性精度: {analysis["directional_accuracy"]:.2f}%
  MAE (平均絶対誤差): {analysis["mae"]:.4f}
  RMSE (二乗平均平方根誤差): {analysis["rmse"]:.4f}
  相関係数: {analysis["correlation"]:.4f}

予測統計:
  総予測数: {analysis["total_predictions"]}
  予測リターン範囲: {analysis["predictions_range"]["min"]:.3f} ～ {analysis["predictions_range"]["max"]:.3f}
  予測平均: {analysis["predictions_range"]["mean"]:.3f} ± {analysis["predictions_range"]["std"]:.3f}

実績統計:
  実績リターン範囲: {analysis["actuals_range"]["min"]:.3f} ～ {analysis["actuals_range"]["max"]:.3f}
  実績平均: {analysis["actuals_range"]["mean"]:.3f} ± {analysis["actuals_range"]["std"]:.3f}

評価:
"""

        # MAPE評価
        if analysis["mape"] < 8:
            report += "  プロフェッショナル使用レベル (MAPE < 8%)\n"
        elif analysis["mape"] < 15:
            report += "  実用レベル (MAPE < 15%)\n"
        elif analysis["mape"] < 25:
            report += "  改善が必要 (MAPE < 25%)\n"
        else:
            report += "  大幅な改善が必要 (MAPE >= 25%)\n"

        # 方向性精度評価
        if analysis["directional_accuracy"] > 60:
            report += "  優秀な方向性予測精度\n"
        elif analysis["directional_accuracy"] > 52:
            report += "  良好な方向性予測精度\n"
        else:
            report += "  方向性予測精度の改善が必要\n"

        report += "\n========================================\n"

        return report


def main():
    """メイン実行関数"""
    print("改善されたリターン率予測バックテスト開始")

    # バックテスター初期化
    backtester = ReturnRateBacktester(prediction_days=5)

    # テスト銘柄
    data_provider = StockDataProvider()
    test_symbols = list(data_provider.jp_stock_codes.keys())[:15]  # 最初の15銘柄

    print(f"テスト銘柄数: {len(test_symbols)}")

    # バックテスト実行
    results = backtester.walk_forward_backtest(
        symbols=test_symbols, start_date="2023-06-01", end_date="2023-12-01",
    )

    # 結果分析
    analysis = backtester.analyze_results(results)

    # レポート生成・表示
    report = backtester.generate_report(analysis)
    print(report)


if __name__ == "__main__":
    main()
