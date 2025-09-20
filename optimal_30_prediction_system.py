#!/usr/bin/env python3
"""
最適30銘柄予測システム
TSE 4000最適化で発見した30銘柄での統合予測
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
from utils.logger_config import setup_logger
logger = setup_logger(__name__)

warnings.filterwarnings("ignore")

from data.stock_data import StockDataProvider
from models.ml_models import UltraHighPerformancePredictor


# 予測システム定数
NEUTRAL_SCORE = 50.0
SCORE_TO_CHANGE_MULTIPLIER = 0.1
CONFIDENCE_MULTIPLIER = 2.0
MAX_CONFIDENCE = 100.0

# パフォーマンス定数
EXPECTED_RETURN_RATE = 17.32  # TSE4000最適化結果
PORTFOLIO_SIZE = 30

# 表示フォーマット定数
SEPARATOR_LINE = "=" * 80
SUBSECTION_LINE = "-" * 60

class Optimal30PredictionSystem:
    def __init__(self):
        self.data_provider = StockDataProvider()
        self.predictor = UltraHighPerformancePredictor()
        self.optimal_30_symbols = self._initialize_optimal_symbols()
        
        self._print_initialization_info()

    def _initialize_optimal_symbols(self) -> List[str]:
        """最適30銘柄の初期化（セクター別分類）"""
        return [
            # CHEMICALS（化学）- 最高収益セクター
            '4004.T',  # スコア27.2, +45.0%期待
            '4005.T',  # スコア19.9, +26.9%期待
            '4188.T',  # スコア13.3, +13.0%期待

            # TECH（テクノロジー）
            '9984.T',  # ソフトバンクG スコア26.9, +44.7%期待
            '8035.T',  # 東京エレクトロン スコア12.9, +16.9%期待

            # ENERGY（エネルギー）
            '1803.T',  # スコア16.9, +22.3%期待
            '5101.T',  # スコア16.5, +21.4%期待
            '1605.T',  # スコア16.5, +20.3%期待
            '1332.T',  # 日本水産 スコア14.0, +11.6%期待
            '5020.T',  # スコア13.7, +15.3%期待

            # FINANCE（金融）
            '8031.T',  # 三井物産 スコア16.6, +20.8%期待
            '8058.T',  # 三菱商事 スコア16.3, +19.2%期待
            '8002.T',  # 丸紅 スコア15.1, +17.9%期待
            '8001.T',  # 伊藤忠商事 スコア13.0, +12.6%期待

            # AUTOMOTIVE（自動車）
            '9022.T',  # スコア16.8, +16.3%期待
            '7269.T',  # スズキ スコア15.0, +18.5%期待
            '7261.T',  # スコア14.7, +19.6%期待
            '5401.T',  # スコア14.2, +12.4%期待

            # CONSUMER（消費）
            '4523.T',  # スコア17.4, +21.4%期待
            '3099.T',  # スコア17.2, +24.9%期待

            # REALESTATE（不動産）
            '1808.T',  # スコア13.9, +7.7%期待
            '1893.T',  # スコア13.8, +13.3%期待
            '8802.T',  # スコア13.6, +13.8%期待
            '1812.T',  # スコア12.7, +12.5%期待

            # MANUFACTURING（製造）
            '6770.T',  # スコア16.1, +19.4%期待
            '6504.T',  # スコア14.3, +18.7%期待

            # その他
            '4324.T',  # TELECOM スコア15.5, +19.0%期待
            '2282.T',  # FOOD スコア13.8, +10.1%期待
            '9101.T',  # TRANSPORT スコア9.1, +6.3%期待
            '4503.T',  # HEALTHCARE スコア10.6, +4.9%期待
        ]

    def _print_initialization_info(self):
        """初期化情報を表示"""
        print(f"最適30銘柄予測システム初期化完了")
        print(f"対象銘柄数: {len(self.optimal_30_symbols)}銘柄")

    def run_comprehensive_prediction(self):
        """最適30銘柄の包括的予測分析"""
        self._print_header()
        
        predictions = []
        success_count = 0

        for i, symbol in enumerate(self.optimal_30_symbols, 1):
            print(f"\n[{i:2d}/30] 予測中: {symbol}")
            
            prediction_result = self._process_single_prediction(symbol)
            if prediction_result:
                predictions.append(prediction_result)
                success_count += 1
                self._print_success_message(prediction_result)

        # 結果分析
        self.analyze_prediction_results(predictions)
        return predictions

    def _print_header(self):
        """ヘッダー情報を表示"""
        print("\n" + SEPARATOR_LINE)
        print("最適30銘柄 統合予測システム")
        print(SEPARATOR_LINE)
        print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"基準: TSE 4000最適化結果（+{EXPECTED_RETURN_RATE}%利益率組み合わせ）")

    def _process_single_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """個別銘柄の予測処理"""
        try:
            stock_data = self._get_stock_data(symbol)
            if stock_data.empty:
                print(f"  ❌ データ取得失敗")
                return None

            prediction_score = self._get_prediction_score(symbol)
            if prediction_score is None or prediction_score <= 0:
                print(f"  [失敗] 予測失敗")
                return None

            return self._create_prediction_result(symbol, stock_data, prediction_score)

        except Exception as e:
            print(f"  [エラー] エラー: {str(e)}")
            return None

    def _get_stock_data(self, symbol: str) -> pd.DataFrame:
        """株価データ取得"""
        return self.data_provider.get_stock_data(symbol, "1y")

    def _get_prediction_score(self, symbol: str) -> Optional[float]:
        """予測スコア取得"""
        return self.predictor.ultra_predict(symbol)

    def _create_prediction_result(self, symbol: str, stock_data: pd.DataFrame, prediction_score: float) -> Dict[str, Any]:
        """予測結果の作成"""
        current_price = stock_data['Close'].iloc[-1]
        change_rate = self._convert_score_to_change_rate(prediction_score)
        predicted_price = current_price * (1 + change_rate / 100)
        confidence = self._calculate_confidence(prediction_score)

        return {
            'symbol': symbol,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'change_rate': change_rate,
            'confidence': confidence,
            'prediction_score': prediction_score
        }

    def _convert_score_to_change_rate(self, score: float) -> float:
        """スコアを変化率に変換"""
        return (score - NEUTRAL_SCORE) * SCORE_TO_CHANGE_MULTIPLIER

    def _calculate_confidence(self, score: float) -> float:
        """信頼度計算"""
        return min(MAX_CONFIDENCE, abs(score - NEUTRAL_SCORE) * CONFIDENCE_MULTIPLIER)

    def _print_success_message(self, result: Dict[str, Any]):
        """成功メッセージの表示"""
        trend = "[上昇]" if result['change_rate'] > 0 else "[下降]"
        print(f"  [成功] {trend} {result['change_rate']:+.2f}% "
              f"(スコア: {result['prediction_score']:.1f}, 信頼度: {result['confidence']:.1f}%)")
        print(f"     現在価格: ¥{result['current_price']:.0f} → 予測価格: ¥{result['predicted_price']:.0f}")

    def analyze_prediction_results(self, predictions):
        """予測結果の分析とランキング"""
        if not predictions:
            print("\n[エラー] 予測可能な銘柄がありませんでした")
            return

        self._print_analysis_header(predictions)
        self._display_bullish_predictions(predictions)
        self._display_bearish_predictions(predictions)
        self._display_statistics(predictions)
        self._display_investment_recommendations(predictions)

    def _print_analysis_header(self, predictions):
        """分析ヘッダーの表示"""
        print("\n" + SEPARATOR_LINE)
        print("最適30銘柄 予測結果分析")
        print(SEPARATOR_LINE)
        print(f"予測成功率: {len(predictions)}/30 ({len(predictions)/30*100:.1f}%)")

    def _display_bullish_predictions(self, predictions):
        """上昇予測銘柄の表示"""
        bullish_predictions = [p for p in predictions if p['change_rate'] > 0]
        
        print(f"\n[上昇予測] 銘柄数: {len(bullish_predictions)}銘柄")
        print(SUBSECTION_LINE)
        print("順位  銘柄     現在価格   予測価格   変化率    信頼度")
        print(SUBSECTION_LINE)

        bullish_sorted = sorted(bullish_predictions, key=lambda x: x['change_rate'], reverse=True)
        for i, pred in enumerate(bullish_sorted[:10], 1):
            print(f"{i:2d}.  {pred['symbol']}  ¥{pred['current_price']:6.0f}  ¥{pred['predicted_price']:6.0f}  "
                  f"{pred['change_rate']:+6.2f}%  {pred['confidence']:5.1f}%")

    def _display_bearish_predictions(self, predictions):
        """下降予測銘柄の表示"""
        bearish_predictions = [p for p in predictions if p['change_rate'] <= 0]
        
        print(f"\n[下降予測] 銘柄数: {len(bearish_predictions)}銘柄")
        if bearish_predictions:
            print(SUBSECTION_LINE)
            bearish_sorted = sorted(bearish_predictions, key=lambda x: x['change_rate'])
            for i, pred in enumerate(bearish_sorted[:5], 1):
                print(f"{i:2d}.  {pred['symbol']}  ¥{pred['current_price']:6.0f}  ¥{pred['predicted_price']:6.0f}  "
                      f"{pred['change_rate']:+6.2f}%  {pred['confidence']:5.1f}%")

    def _display_statistics(self, predictions):
        """統計サマリーの表示"""
        if not predictions:
            return

        avg_change = np.mean([p['change_rate'] for p in predictions])
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        max_gain = max(p['change_rate'] for p in predictions)
        min_gain = min(p['change_rate'] for p in predictions)

        print(f"\n[統計サマリー]")
        print("-" * 40)
        print(f"平均変化率: {avg_change:+.2f}%")
        print(f"平均信頼度: {avg_confidence:.1f}%")
        print(f"最大上昇予測: {max_gain:+.2f}%")
        print(f"最大下降予測: {min_gain:+.2f}%")

    def _display_investment_recommendations(self, predictions):
        """投資推奨の表示"""
        print(f"\n[投資推奨] トップ5")
        print("-" * 40)
        
        top_5 = sorted(predictions, key=lambda x: x['change_rate'] * x['confidence'], reverse=True)[:5]
        for i, pred in enumerate(top_5, 1):
            score = pred['change_rate'] * pred['confidence'] / 100
            print(f"{i}. {pred['symbol']} - 変化率:{pred['change_rate']:+.2f}% "
                  f"信頼度:{pred['confidence']:.1f}% (スコア:{score:+.1f})")

    def run_single_prediction(self, symbol):
        """個別銘柄の詳細予測"""
        if symbol not in self.optimal_30_symbols:
            print(f"[エラー] {symbol}は最適30銘柄に含まれていません")
            return None

        print(f"\n[詳細予測] {symbol} 分析")
        print("-" * 50)

        try:
            stock_data = self._get_stock_data(symbol)
            if stock_data.empty:
                print(f"[エラー] データ取得失敗: {symbol}")
                return None

            return self._run_multi_period_prediction(symbol, stock_data)

        except Exception as e:
            print(f"[エラー] 予測エラー: {str(e)}")
            return None

    def _run_multi_period_prediction(self, symbol: str, stock_data: pd.DataFrame) -> Dict:
        """複数期間での予測"""
        periods = [1, 3, 5, 10]
        results = {}

        for days in periods:
            prediction_score = self._get_prediction_score(symbol)

            if prediction_score and prediction_score > 0:
                current_price = stock_data['Close'].iloc[-1]
                change_rate = self._convert_score_to_change_rate(prediction_score)
                predicted_price = current_price * (1 + change_rate / 100)
                confidence = self._calculate_confidence(prediction_score)

                results[days] = {
                    'predicted_price': predicted_price,
                    'change_rate': change_rate,
                    'confidence': confidence
                }

                print(f"{days:2d}日後予測: ¥{predicted_price:6.0f} ({change_rate:+6.2f}%) "
                      f"スコア:{prediction_score:5.1f} 信頼度:{confidence:5.1f}%")

        return results

def main():
    """メイン実行関数"""
    system = Optimal30PredictionSystem()

    print("\n選択してください:")
    print("1. 全30銘柄の統合予測")
    print("2. 個別銘柄の詳細予測")
    print("3. 上位10銘柄のクイック予測")

    try:
        choice = input("\n選択 (1-3): ").strip()

        if choice == "1":
            predictions = system.run_comprehensive_prediction()

        elif choice == "2":
            print("\n最適30銘柄:")
            for i, symbol in enumerate(system.optimal_30_symbols, 1):
                print(f"{i:2d}. {symbol}")

            symbol = input("\n銘柄コードを入力: ").strip().upper()
            if not symbol.endswith('.T'):
                symbol += '.T'

            system.run_single_prediction(symbol)

        elif choice == "3":
            # トップ10銘柄のクイック予測
            top_10 = system.optimal_30_symbols[:10]
            print(f"\n[クイック予測] トップ10銘柄")

            for symbol in top_10:
                try:
                    stock_data = system.data_provider.get_stock_data(symbol, "6mo")
                    if not stock_data.empty:
                        prediction_score = system.predictor.ultra_predict(symbol)
                        if prediction_score and prediction_score > 0:
                            current_price = stock_data['Close'].iloc[-1]
                            change_rate = (prediction_score - 50) * 0.1
                            predicted_price = current_price * (1 + change_rate / 100)
                            print(f"{symbol}: {change_rate:+.2f}% (¥{current_price:.0f}→¥{predicted_price:.0f}) スコア:{prediction_score:.1f}")
                except:
                    pass
        else:
            print("[エラー] 無効な選択です")

    except KeyboardInterrupt:
        print("\n\n予測システムを終了します")
    except Exception as e:
        print(f"\n[エラー] エラー: {str(e)}")

if __name__ == "__main__":
    main()