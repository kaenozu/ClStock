#!/usr/bin/env python3
"""
ClStock MAPE計測システム
84.6%精度ベースシステムの正確なMAPE値を計測

MAPE = Mean Absolute Percentage Error
価格予測精度の標準的な評価指標
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging
from utils.logger_config import setup_logger
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = setup_logger(__name__)

class MAPEMeasurementSystem:
    """MAPE計測システム"""

    def __init__(self):
        self.test_symbols = ['7203', '6758', '9984', '8306', '6861', '4661', '9433', '4519', '6367', '8035']
        self.results = {}

    def calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """MAPE計算"""
        # ゼロ除算を避ける
        mask = actual != 0
        if not np.any(mask):
            return float('inf')

        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return mape

    def get_stock_data_and_predict(self, symbol: str) -> Dict[str, Any]:
        """株価データ取得と予測実行"""
        try:
            import yfinance as yf
            from trend_following_predictor import TrendFollowingPredictor

            # データ取得
            ticker = yf.Ticker(f"{symbol}.T")
            data = ticker.history(period="1y")

            if len(data) < 60:
                return {'error': 'Insufficient data'}

            # 84.6%精度予測システム
            predictor = TrendFollowingPredictor()

            # 予測実行（過去30日間）
            test_period = 30
            actual_prices = []
            predicted_prices = []

            for i in range(test_period):
                # バックテスト用データ分割
                end_date = len(data) - test_period + i
                if end_date < 60:
                    continue

                historical_data = data.iloc[:end_date]
                actual_price = data.iloc[end_date]['Close']

                # 一時的に古いデータで予測
                try:
                    # trend_following_predictorを古いデータで実行
                    prediction_result = predictor.predict_stock(symbol, data=historical_data)

                    # 価格予測（簡易版：現在価格からの変化率予測）
                    current_price = historical_data['Close'].iloc[-1]
                    if prediction_result['direction'] == 1:
                        # 上昇予測：信頼度に応じた上昇率
                        price_change = current_price * 0.02 * prediction_result['confidence']
                        predicted_price = current_price + price_change
                    else:
                        # 下降予測
                        price_change = current_price * 0.02 * prediction_result['confidence']
                        predicted_price = current_price - price_change

                    actual_prices.append(actual_price)
                    predicted_prices.append(predicted_price)

                except Exception as e:
                    logger.warning(f"Prediction error for {symbol} at day {i}: {e}")
                    continue

            if len(actual_prices) < 10:
                return {'error': 'Insufficient predictions'}

            # MAPE計算
            actual_array = np.array(actual_prices)
            predicted_array = np.array(predicted_prices)

            mape = self.calculate_mape(actual_array, predicted_array)

            # 追加統計
            mae = np.mean(np.abs(actual_array - predicted_array))
            rmse = np.sqrt(np.mean((actual_array - predicted_array) ** 2))

            return {
                'symbol': symbol,
                'mape': mape,
                'mae': mae,
                'rmse': rmse,
                'actual_prices': actual_prices,
                'predicted_prices': predicted_prices,
                'predictions_count': len(actual_prices),
                'data_points': len(data)
            }

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return {'error': str(e)}

    def run_comprehensive_mape_test(self) -> Dict[str, Any]:
        """包括的MAPE計測"""
        logger.info("=" * 60)
        logger.info("ClStock MAPE計測システム")
        logger.info("84.6%精度ベースシステムの正確なMAPE測定")
        logger.info("=" * 60)

        all_results = {}
        successful_measurements = 0
        total_mape_values = []

        for symbol in self.test_symbols:
            logger.info(f"\n測定中: {symbol}")

            result = self.get_stock_data_and_predict(symbol)

            if 'error' in result:
                logger.error(f"{symbol}: {result['error']}")
                all_results[symbol] = result
                continue

            mape = result['mape']
            mae = result['mae']
            rmse = result['rmse']
            predictions_count = result['predictions_count']

            logger.info(f"{symbol}: MAPE={mape:.2f}%, MAE={mae:.2f}, RMSE={rmse:.2f}, 予測数={predictions_count}")

            all_results[symbol] = result
            successful_measurements += 1
            total_mape_values.append(mape)

        # 総合統計
        if total_mape_values:
            average_mape = np.mean(total_mape_values)
            median_mape = np.median(total_mape_values)
            min_mape = np.min(total_mape_values)
            max_mape = np.max(total_mape_values)
            std_mape = np.std(total_mape_values)
        else:
            average_mape = median_mape = min_mape = max_mape = std_mape = 0.0

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols_tested': len(self.test_symbols),
            'successful_measurements': successful_measurements,
            'success_rate': successful_measurements / len(self.test_symbols) * 100,
            'average_mape': average_mape,
            'median_mape': median_mape,
            'min_mape': min_mape,
            'max_mape': max_mape,
            'std_mape': std_mape,
            'individual_results': all_results,
            'mape_distribution': self._analyze_mape_distribution(total_mape_values)
        }

        # 結果表示
        logger.info("\n" + "=" * 60)
        logger.info("MAPE計測結果サマリー")
        logger.info("=" * 60)
        logger.info(f"測定銘柄数: {len(self.test_symbols)}")
        logger.info(f"成功測定数: {successful_measurements}")
        logger.info(f"成功率: {summary['success_rate']:.1f}%")
        logger.info(f"")
        logger.info(f"MAPE統計:")
        logger.info(f"  平均MAPE: {average_mape:.2f}%")
        logger.info(f"  中央値MAPE: {median_mape:.2f}%")
        logger.info(f"  最小MAPE: {min_mape:.2f}%")
        logger.info(f"  最大MAPE: {max_mape:.2f}%")
        logger.info(f"  標準偏差: {std_mape:.2f}%")

        # MAPE評価
        mape_grade = self._evaluate_mape_performance(average_mape)
        logger.info(f"")
        logger.info(f"MAPE評価: {mape_grade}")

        return summary

    def _analyze_mape_distribution(self, mape_values: List[float]) -> Dict[str, Any]:
        """MAPE分布分析"""
        if not mape_values:
            return {}

        # MAPE範囲別分析
        excellent = sum(1 for x in mape_values if x <= 5.0)    # 5%以下
        good = sum(1 for x in mape_values if 5.0 < x <= 10.0)  # 5-10%
        acceptable = sum(1 for x in mape_values if 10.0 < x <= 20.0)  # 10-20%
        poor = sum(1 for x in mape_values if x > 20.0)         # 20%超

        return {
            'excellent_5pct': excellent,
            'good_5_10pct': good,
            'acceptable_10_20pct': acceptable,
            'poor_20pct_plus': poor,
            'percentile_25': np.percentile(mape_values, 25),
            'percentile_75': np.percentile(mape_values, 75),
            'percentile_90': np.percentile(mape_values, 90)
        }

    def _evaluate_mape_performance(self, average_mape: float) -> str:
        """MAPE性能評価"""
        if average_mape <= 5.0:
            return "優秀 (5%以下) - 極めて高精度"
        elif average_mape <= 10.0:
            return "良好 (5-10%) - 高精度"
        elif average_mape <= 15.0:
            return "標準 (10-15%) - 実用的精度"
        elif average_mape <= 20.0:
            return "許容 (15-20%) - 一般的精度"
        else:
            return "要改善 (20%超) - 精度向上が必要"

def main():
    """メイン実行関数"""
    print("=" * 70)
    print("ClStock MAPE計測システム")
    print("84.6%精度ベースシステムの価格予測精度測定")
    print("=" * 70)

    # MAPE計測システム初期化
    mape_system = MAPEMeasurementSystem()

    try:
        # 包括的MAPE計測実行
        results = mape_system.run_comprehensive_mape_test()

        # 結果保存
        import json
        output_file = f"mape_measurement_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # JSON serializable変換
        serializable_results = {}
        for key, value in results.items():
            try:
                json.dumps(value)
                serializable_results[key] = value
            except:
                if isinstance(value, (np.ndarray, list)):
                    serializable_results[key] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in value]
                else:
                    serializable_results[key] = str(value)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"\n📊 詳細結果を保存: {output_file}")

        # 最終評価
        avg_mape = results.get('average_mape', 0)
        print(f"\n🎯 最終MAPE評価")
        print(f"平均MAPE: {avg_mape:.2f}%")

        if avg_mape <= 10.0:
            print("✅ 優秀な価格予測精度です！")
        elif avg_mape <= 15.0:
            print("👍 実用的な価格予測精度です")
        else:
            print("⚠️ 価格予測精度の改善が推奨されます")

        return 0

    except KeyboardInterrupt:
        print("\n⏸️ MAPE計測中断")
        return 130
    except Exception as e:
        logger.error(f"MAPE計測エラー: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())