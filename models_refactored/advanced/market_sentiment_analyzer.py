#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市場センチメント分析システム
ニュース、SNS、市場データから総合的なセンチメントを分析
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import re
import json

@dataclass
class SentimentData:
    """センチメントデータ構造"""
    symbol: str
    sentiment_score: float  # -1.0 (最も悲観的) から 1.0 (最も楽観的)
    confidence: float
    volume_indicator: float  # 取引量に基づく関心度
    momentum: float  # センチメントの変化速度
    sources: Dict[str, float]  # 情報源別のセンチメントスコア
    timestamp: datetime
    metadata: Dict[str, Any]

class NewsAnalyzer:
    """ニュース分析器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # センチメント用キーワード辞書
        self.positive_keywords = [
            '上昇', '好調', '増益', '回復', '成長', '新高値', '買い', '期待',
            '改善', '黒字', '最高益', '増配', '好決算', '突破', '急騰'
        ]

        self.negative_keywords = [
            '下落', '不調', '減益', '悪化', '縮小', '安値', '売り', '懸念',
            '赤字', '減配', '下方修正', '急落', '暴落', '不安', 'リスク'
        ]

        self.neutral_keywords = [
            '横ばい', '維持', '変わらず', '様子見', '中立', '保合い'
        ]

    def analyze_news_sentiment(self, news_texts: List[str]) -> float:
        """ニュースセンチメント分析"""
        if not news_texts:
            return 0.0

        total_sentiment = 0.0

        for text in news_texts:
            positive_count = sum(1 for keyword in self.positive_keywords if keyword in text)
            negative_count = sum(1 for keyword in self.negative_keywords if keyword in text)

            # センチメントスコア計算
            if positive_count + negative_count > 0:
                sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            else:
                sentiment = 0.0

            total_sentiment += sentiment

        return total_sentiment / len(news_texts)

class SocialMediaAnalyzer:
    """ソーシャルメディア分析器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # SNS特有の表現
        self.bullish_expressions = ['🚀', '📈', 'moon', 'buy', '買い', 'long', '爆益']
        self.bearish_expressions = ['📉', 'sell', '売り', 'short', '損切り', '暴落']

    def analyze_social_sentiment(self, social_posts: List[Dict[str, Any]]) -> Tuple[float, float]:
        """ソーシャルメディアセンチメント分析"""
        if not social_posts:
            return 0.0, 0.0

        sentiment_scores = []
        engagement_weights = []

        for post in social_posts:
            text = post.get('text', '')
            likes = post.get('likes', 0)
            retweets = post.get('retweets', 0)

            # エンゲージメント重み
            engagement = np.log1p(likes + retweets * 2)
            engagement_weights.append(engagement)

            # センチメント分析
            bullish_count = sum(1 for expr in self.bullish_expressions if expr in text.lower())
            bearish_count = sum(1 for expr in self.bearish_expressions if expr in text.lower())

            if bullish_count + bearish_count > 0:
                sentiment = (bullish_count - bearish_count) / (bullish_count + bearish_count)
            else:
                sentiment = 0.0

            sentiment_scores.append(sentiment)

        # 重み付き平均
        if sum(engagement_weights) > 0:
            weighted_sentiment = np.average(sentiment_scores, weights=engagement_weights)
            # エンゲージメントボリューム指標
            volume_indicator = np.log1p(sum(engagement_weights)) / 10
        else:
            weighted_sentiment = np.mean(sentiment_scores)
            volume_indicator = 0.1

        return weighted_sentiment, min(volume_indicator, 1.0)

class TechnicalSentimentAnalyzer:
    """技術的指標に基づくセンチメント分析"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_technical_sentiment(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """技術的センチメント分析"""
        if price_data.empty or len(price_data) < 20:
            return {
                'trend_sentiment': 0.0,
                'momentum_sentiment': 0.0,
                'volatility_sentiment': 0.0,
                'volume_sentiment': 0.0
            }

        try:
            # トレンドセンチメント（移動平均）
            ma_short = price_data['Close'].rolling(window=5).mean().iloc[-1]
            ma_long = price_data['Close'].rolling(window=20).mean().iloc[-1]
            current_price = price_data['Close'].iloc[-1]

            trend_sentiment = 0.0
            if ma_long > 0:
                trend_sentiment = (ma_short - ma_long) / ma_long
                trend_sentiment = max(min(trend_sentiment, 1.0), -1.0)

            # モメンタムセンチメント（RSI的な指標）
            price_changes = price_data['Close'].pct_change().dropna()
            gains = price_changes[price_changes > 0].mean()
            losses = abs(price_changes[price_changes < 0].mean())

            if losses > 0:
                rs = gains / losses
                rsi = 1 - (1 / (1 + rs))
                momentum_sentiment = (rsi - 0.5) * 2  # -1 to 1に正規化
            else:
                momentum_sentiment = 1.0 if gains > 0 else 0.0

            # ボラティリティセンチメント（低ボラティリティ = ポジティブ）
            volatility = price_data['Close'].pct_change().std()
            volatility_sentiment = 1.0 - min(volatility * 10, 1.0)  # 高ボラティリティはネガティブ

            # 出来高センチメント
            if 'Volume' in price_data.columns:
                recent_volume = price_data['Volume'].iloc[-5:].mean()
                historical_volume = price_data['Volume'].iloc[:-5].mean()

                if historical_volume > 0:
                    volume_ratio = recent_volume / historical_volume
                    volume_sentiment = np.tanh((volume_ratio - 1) * 2)  # -1 to 1に正規化
                else:
                    volume_sentiment = 0.0
            else:
                volume_sentiment = 0.0

            return {
                'trend_sentiment': trend_sentiment,
                'momentum_sentiment': momentum_sentiment,
                'volatility_sentiment': volatility_sentiment,
                'volume_sentiment': volume_sentiment
            }

        except Exception as e:
            self.logger.error(f"Technical sentiment analysis failed: {str(e)}")
            return {
                'trend_sentiment': 0.0,
                'momentum_sentiment': 0.0,
                'volatility_sentiment': 0.0,
                'volume_sentiment': 0.0
            }

class MarketSentimentAnalyzer:
    """
    市場センチメント総合分析システム

    特徴:
    - マルチソース分析（ニュース、SNS、技術指標）
    - リアルタイムセンチメント更新
    - センチメント予測モデル
    - 異常検知機能
    """

    def __init__(self, history_window: int = 100):
        self.logger = logging.getLogger(__name__)
        self.history_window = history_window

        # 分析器初期化
        self.news_analyzer = NewsAnalyzer()
        self.social_analyzer = SocialMediaAnalyzer()
        self.technical_analyzer = TechnicalSentimentAnalyzer()

        # センチメント履歴
        self.sentiment_history = defaultdict(list)

        # センチメント重み設定
        self.weights = {
            'news': 0.3,
            'social': 0.2,
            'technical': 0.5
        }

        self.logger.info("MarketSentimentAnalyzer initialized")

    def analyze_comprehensive_sentiment(self,
                                       symbol: str,
                                       news_data: Optional[List[str]] = None,
                                       social_data: Optional[List[Dict]] = None,
                                       price_data: Optional[pd.DataFrame] = None) -> SentimentData:
        """包括的センチメント分析"""

        sentiment_sources = {}

        # ニュースセンチメント
        if news_data:
            news_sentiment = self.news_analyzer.analyze_news_sentiment(news_data)
            sentiment_sources['news'] = news_sentiment
        else:
            sentiment_sources['news'] = 0.0

        # ソーシャルメディアセンチメント
        volume_indicator = 0.5
        if social_data:
            social_sentiment, volume_indicator = self.social_analyzer.analyze_social_sentiment(social_data)
            sentiment_sources['social'] = social_sentiment
        else:
            sentiment_sources['social'] = 0.0

        # 技術的センチメント
        if price_data is not None and not price_data.empty:
            technical_sentiments = self.technical_analyzer.analyze_technical_sentiment(price_data)
            sentiment_sources['technical'] = np.mean(list(technical_sentiments.values()))
            sentiment_sources.update(technical_sentiments)
        else:
            sentiment_sources['technical'] = 0.0

        # 総合センチメントスコア計算
        total_sentiment = 0.0
        total_weight = 0.0

        for source, weight in self.weights.items():
            if source in sentiment_sources:
                total_sentiment += sentiment_sources[source] * weight
                total_weight += weight

        if total_weight > 0:
            total_sentiment /= total_weight

        # モメンタム計算
        momentum = self._calculate_sentiment_momentum(symbol, total_sentiment)

        # 信頼度計算
        confidence = self._calculate_confidence(sentiment_sources)

        # センチメントデータ作成
        sentiment_data = SentimentData(
            symbol=symbol,
            sentiment_score=total_sentiment,
            confidence=confidence,
            volume_indicator=volume_indicator,
            momentum=momentum,
            sources=sentiment_sources,
            timestamp=datetime.now(),
            metadata={
                'data_available': {
                    'news': news_data is not None,
                    'social': social_data is not None,
                    'technical': price_data is not None
                }
            }
        )

        # 履歴更新
        self._update_history(symbol, sentiment_data)

        return sentiment_data

    def _calculate_sentiment_momentum(self, symbol: str, current_sentiment: float) -> float:
        """センチメントモメンタム計算"""
        if symbol not in self.sentiment_history or len(self.sentiment_history[symbol]) < 2:
            return 0.0

        recent_sentiments = [s.sentiment_score for s in self.sentiment_history[symbol][-10:]]
        recent_sentiments.append(current_sentiment)

        if len(recent_sentiments) >= 3:
            # 移動平均の変化率
            recent_ma = np.mean(recent_sentiments[-3:])
            previous_ma = np.mean(recent_sentiments[-6:-3]) if len(recent_sentiments) >= 6 else recent_sentiments[0]

            if abs(previous_ma) > 0.01:
                momentum = (recent_ma - previous_ma) / abs(previous_ma)
                return max(min(momentum, 1.0), -1.0)

        return 0.0

    def _calculate_confidence(self, sentiment_sources: Dict[str, float]) -> float:
        """信頼度計算"""
        # データソースの一致度に基づく信頼度
        sentiments = list(sentiment_sources.values())

        if len(sentiments) < 2:
            return 0.3  # データ不足時は低信頼度

        # センチメントの一致度（標準偏差が小さいほど高信頼）
        std_dev = np.std(sentiments)
        consistency_score = 1.0 - min(std_dev, 1.0)

        # データソース数に基づくボーナス
        source_bonus = len(sentiments) / 10

        confidence = min(consistency_score + source_bonus, 1.0)
        return max(confidence, 0.1)

    def _update_history(self, symbol: str, sentiment_data: SentimentData):
        """センチメント履歴更新"""
        self.sentiment_history[symbol].append(sentiment_data)

        # 履歴サイズ制限
        if len(self.sentiment_history[symbol]) > self.history_window:
            self.sentiment_history[symbol] = self.sentiment_history[symbol][-self.history_window:]

    def get_sentiment_trend(self, symbol: str, periods: int = 20) -> Dict[str, Any]:
        """センチメントトレンド取得"""
        if symbol not in self.sentiment_history or len(self.sentiment_history[symbol]) < 2:
            return {
                'trend': 'neutral',
                'strength': 0.0,
                'turning_points': []
            }

        recent_history = self.sentiment_history[symbol][-periods:]
        sentiments = [s.sentiment_score for s in recent_history]

        # トレンド判定
        if len(sentiments) >= 3:
            recent_avg = np.mean(sentiments[-3:])
            older_avg = np.mean(sentiments[:-3])

            trend_strength = recent_avg - older_avg

            if trend_strength > 0.1:
                trend = 'bullish'
            elif trend_strength < -0.1:
                trend = 'bearish'
            else:
                trend = 'neutral'

            # 転換点検出
            turning_points = self._detect_turning_points(sentiments)

            return {
                'trend': trend,
                'strength': abs(trend_strength),
                'turning_points': turning_points,
                'recent_sentiments': sentiments[-5:]
            }

        return {
            'trend': 'neutral',
            'strength': 0.0,
            'turning_points': []
        }

    def _detect_turning_points(self, sentiments: List[float]) -> List[int]:
        """転換点検出"""
        if len(sentiments) < 3:
            return []

        turning_points = []

        for i in range(1, len(sentiments) - 1):
            # ローカル最大値または最小値
            if (sentiments[i] > sentiments[i-1] and sentiments[i] > sentiments[i+1]) or \
               (sentiments[i] < sentiments[i-1] and sentiments[i] < sentiments[i+1]):
                turning_points.append(i)

        return turning_points

    def detect_sentiment_anomaly(self, symbol: str) -> Dict[str, Any]:
        """センチメント異常検知"""
        if symbol not in self.sentiment_history or len(self.sentiment_history[symbol]) < 20:
            return {'anomaly_detected': False}

        recent_sentiments = [s.sentiment_score for s in self.sentiment_history[symbol][-20:]]

        # 統計的異常検知
        mean = np.mean(recent_sentiments[:-1])
        std = np.std(recent_sentiments[:-1])
        current = recent_sentiments[-1]

        if std > 0:
            z_score = abs((current - mean) / std)

            if z_score > 2.5:  # 2.5σを超える変動
                return {
                    'anomaly_detected': True,
                    'z_score': z_score,
                    'direction': 'positive' if current > mean else 'negative',
                    'severity': 'high' if z_score > 3 else 'medium'
                }

        return {'anomaly_detected': False}

    def get_market_mood(self, symbols: List[str]) -> Dict[str, Any]:
        """市場全体のムード分析"""
        if not symbols:
            return {'mood': 'neutral', 'strength': 0.0}

        total_sentiment = 0.0
        valid_symbols = 0

        for symbol in symbols:
            if symbol in self.sentiment_history and self.sentiment_history[symbol]:
                latest = self.sentiment_history[symbol][-1]
                total_sentiment += latest.sentiment_score
                valid_symbols += 1

        if valid_symbols == 0:
            return {'mood': 'neutral', 'strength': 0.0}

        avg_sentiment = total_sentiment / valid_symbols

        if avg_sentiment > 0.3:
            mood = 'bullish'
        elif avg_sentiment < -0.3:
            mood = 'bearish'
        else:
            mood = 'neutral'

        return {
            'mood': mood,
            'strength': abs(avg_sentiment),
            'analyzed_symbols': valid_symbols,
            'average_sentiment': avg_sentiment
        }

    def generate_sentiment_report(self, symbol: str) -> Dict[str, Any]:
        """センチメントレポート生成"""
        if symbol not in self.sentiment_history or not self.sentiment_history[symbol]:
            return {'status': 'no_data'}

        latest = self.sentiment_history[symbol][-1]
        trend = self.get_sentiment_trend(symbol)
        anomaly = self.detect_sentiment_anomaly(symbol)

        report = {
            'symbol': symbol,
            'current_sentiment': {
                'score': latest.sentiment_score,
                'confidence': latest.confidence,
                'momentum': latest.momentum,
                'volume_indicator': latest.volume_indicator
            },
            'sources_breakdown': latest.sources,
            'trend': trend,
            'anomaly': anomaly,
            'recommendation': self._generate_recommendation(latest, trend, anomaly),
            'timestamp': latest.timestamp
        }

        return report

    def _generate_recommendation(self, latest: SentimentData,
                                trend: Dict[str, Any],
                                anomaly: Dict[str, Any]) -> str:
        """推奨事項生成"""
        score = latest.sentiment_score
        confidence = latest.confidence
        momentum = latest.momentum

        # 異常検知時
        if anomaly.get('anomaly_detected'):
            if anomaly['direction'] == 'positive':
                return "異常な楽観傾向検出 - 慎重な判断を推奨"
            else:
                return "異常な悲観傾向検出 - 逆張りの機会の可能性"

        # 通常時
        if score > 0.5 and confidence > 0.7 and momentum > 0:
            return "強い買いシグナル - ポジティブセンチメント継続"
        elif score > 0.3 and trend['trend'] == 'bullish':
            return "買い推奨 - 上昇トレンド確認"
        elif score < -0.5 and confidence > 0.7:
            return "売りシグナル - ネガティブセンチメント強い"
        elif score < -0.3 and trend['trend'] == 'bearish':
            return "売り推奨 - 下降トレンド継続"
        elif abs(score) < 0.2:
            return "中立 - 明確な方向性なし"
        else:
            return "様子見推奨 - センチメント不安定"