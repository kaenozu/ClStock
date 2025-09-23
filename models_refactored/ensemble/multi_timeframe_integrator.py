"""
マルチタイムフレーム統合システム - 複数時間軸の分析で精度向上
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any


class MultiTimeframeIntegrator:
    """マルチタイムフレーム統合システム - 複数時間軸の分析で精度向上"""

    def __init__(self):
        self.timeframes = {
            'short': '3mo',    # 短期：3ヶ月
            'medium': '1y',    # 中期：1年
            'long': '2y'       # 長期：2年
        }
        self.weights = {
            'short': 0.5,      # 短期重視
            'medium': 0.3,     # 中期
            'long': 0.2        # 長期
        }
        self.logger = logging.getLogger(__name__)

    def integrate_predictions(self, symbol: str, data_provider) -> Dict[str, Any]:
        """複数タイムフレームの予測を統合"""
        timeframe_results = {}

        for timeframe_name, period in self.timeframes.items():
            try:
                # 各タイムフレームでデータ取得
                data = data_provider.get_stock_data(symbol, period)
                if data.empty:
                    continue

                # タイムフレーム別分析
                analysis = self._analyze_timeframe(data, timeframe_name)
                timeframe_results[timeframe_name] = analysis

                self.logger.debug(f"Analyzed {timeframe_name} timeframe for {symbol}")

            except Exception as e:
                self.logger.error(f"Error analyzing {timeframe_name} timeframe for {symbol}: {str(e)}")

        # 統合予測の計算
        integrated_prediction = self._calculate_integrated_prediction(timeframe_results)

        return {
            'integrated_prediction': integrated_prediction,
            'timeframe_details': timeframe_results,
            'confidence_adjustment': self._calculate_confidence_adjustment(timeframe_results)
        }

    def _analyze_timeframe(self, data: pd.DataFrame, timeframe: str) -> Dict[str, float]:
        """特定タイムフレームの分析"""
        if data.empty or len(data) < 10:
            return {'trend': 0.0, 'momentum': 0.0, 'volatility': 0.0, 'strength': 0.0}

        # トレンド分析
        trend_score = self._calculate_trend_score(data)

        # モメンタム分析
        momentum_score = self._calculate_momentum_score(data)

        # ボラティリティ分析
        volatility_score = self._calculate_volatility_score(data)

        # 全体の強度スコア
        strength_score = (trend_score + momentum_score - abs(volatility_score)) / 3

        return {
            'trend': trend_score,
            'momentum': momentum_score,
            'volatility': volatility_score,
            'strength': strength_score
        }

    def _calculate_trend_score(self, data: pd.DataFrame) -> float:
        """トレンドスコア計算"""
        if len(data) < 20:
            return 0.0

        # 短期・長期移動平均の関係
        short_ma = data['Close'].rolling(window=10).mean().iloc[-1]
        long_ma = data['Close'].rolling(window=20).mean().iloc[-1]
        current_price = data['Close'].iloc[-1]

        # トレンド強度
        if short_ma > long_ma and current_price > short_ma:
            trend_strength = (current_price - long_ma) / long_ma
            return min(100, max(-100, trend_strength * 100))
        elif short_ma < long_ma and current_price < short_ma:
            trend_strength = (current_price - long_ma) / long_ma
            return min(100, max(-100, trend_strength * 100))
        else:
            return 0.0

    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """モメンタムスコア計算"""
        if len(data) < 5:
            return 0.0

        # 価格変化率
        price_changes = []
        for period in [1, 3, 5]:
            if len(data) > period:
                denominator = data['Close'].iloc[-1-period]
                if denominator != 0:
                    change = (data['Close'].iloc[-1] - denominator) / denominator
                price_changes.append(change)

        if price_changes:
            momentum = np.mean(price_changes) * 100
            return min(100, max(-100, momentum))
        return 0.0

    def _calculate_volatility_score(self, data: pd.DataFrame) -> float:
        """ボラティリティスコア計算"""
        if len(data) < 10:
            return 0.0

        # 価格変動率の標準偏差
        returns = data['Close'].pct_change().dropna()
        if len(returns) > 0:
            volatility = returns.std() * np.sqrt(252) * 100  # 年率化
            return min(100, volatility)
        return 0.0

    def _calculate_integrated_prediction(self, timeframe_results: Dict) -> float:
        """統合予測の計算"""
        if not timeframe_results:
            return 50.0

        weighted_score = 0.0
        total_weight = 0.0

        for timeframe, weight in self.weights.items():
            if timeframe in timeframe_results:
                strength = timeframe_results[timeframe]['strength']
                weighted_score += strength * weight * 50 + 50  # 0-100スケールに変換
                total_weight += weight

        if total_weight > 0:
            integrated = weighted_score / total_weight
            return max(0, min(100, integrated))
        return 50.0

    def _calculate_confidence_adjustment(self, timeframe_results: Dict) -> float:
        """信頼度調整値の計算"""
        if not timeframe_results:
            return 0.0

        # 各タイムフレームの強度の一致度
        strengths = [result['strength'] for result in timeframe_results.values()]
        if len(strengths) > 1:
            # 標準偏差が小さいほど一致度が高い
            consistency = 1.0 - (np.std(strengths) / 100.0)
            return max(0.0, min(1.0, consistency))
        return 0.5