"""Data providers and analysis modules."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MacroEconomicDataProvider:
    """マクロ経済データプロバイダー"""

    def __init__(self):
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.cache_expiry = timedelta(hours=6)  # 6時間でキャッシュ期限切れ

    def get_economic_indicators(self, start_date: str, end_date: str) -> pd.DataFrame:
        """経済指標データを取得"""
        cache_key = f"economic_{start_date}_{end_date}"

        # キャッシュチェック
        if cache_key in self.data_cache:
            cached_data, cached_time = self.data_cache[cache_key]
            if datetime.now() - cached_time < self.cache_expiry:
                return cached_data

        try:
            # 実際のAPIコールの代わりにダミーデータを生成
            economic_data = self._generate_dummy_economic_data(start_date, end_date)

            # キャッシュに保存
            self.data_cache[cache_key] = (economic_data, datetime.now())

            return economic_data

        except Exception as e:
            logger.error(f"Economic data fetch failed: {e}")
            return pd.DataFrame()

    def _generate_dummy_economic_data(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """ダミーの経済データを生成"""
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # 経済指標のダミーデータ
        np.random.seed(42)
        data = {
            "gdp_growth": np.random.normal(2.0, 0.5, len(dates)),
            "inflation_rate": np.random.normal(2.5, 0.3, len(dates)),
            "unemployment_rate": np.random.normal(4.0, 0.2, len(dates)),
            "interest_rate": np.random.normal(1.5, 0.1, len(dates)),
            "consumer_confidence": np.random.normal(100, 10, len(dates)),
            "manufacturing_pmi": np.random.normal(52, 3, len(dates)),
            "services_pmi": np.random.normal(53, 3, len(dates)),
            "dollar_index": np.random.normal(95, 2, len(dates)),
            "vix_index": np.random.normal(20, 5, len(dates)),
            "oil_price": np.random.normal(70, 10, len(dates)),
        }

        return pd.DataFrame(data, index=dates)

    def get_sector_performance(self, symbols: List[str]) -> Dict[str, float]:
        """セクター別パフォーマンスを取得"""
        # セクター分類のダミーデータ
        sector_mapping = {
            "AAPL": "Technology",
            "GOOGL": "Technology",
            "MSFT": "Technology",
            "JPM": "Financial",
            "BAC": "Financial",
            "JNJ": "Healthcare",
            "PFE": "Healthcare",
            "XOM": "Energy",
            "CVX": "Energy",
        }

        sector_performance = {}
        for symbol in symbols:
            sector = sector_mapping.get(symbol, "Unknown")
            if sector not in sector_performance:
                # ランダムなパフォーマンス生成
                sector_performance[sector] = np.random.normal(0.02, 0.05)

        return sector_performance

    def get_market_sentiment_indicators(self) -> Dict[str, float]:
        """市場センチメント指標を取得"""
        return {
            "fear_greed_index": np.random.uniform(0, 100),
            "put_call_ratio": np.random.uniform(0.5, 1.5),
            "margin_debt": np.random.uniform(-0.1, 0.1),
            "insider_trading_ratio": np.random.uniform(-0.05, 0.05),
            "short_interest": np.random.uniform(0.1, 0.3),
        }


class SentimentAnalyzer:
    """センチメント分析器"""

    def __init__(self):
        self.sentiment_cache: Dict[str, Tuple[float, datetime]] = {}
        self.cache_expiry = timedelta(hours=1)  # 1時間でキャッシュ期限切れ

    def analyze_news_sentiment(self, symbol: str, articles: List[str]) -> float:
        """ニュース記事のセンチメント分析"""
        cache_key = f"news_{symbol}_{hash(str(articles))}"

        # キャッシュチェック
        if cache_key in self.sentiment_cache:
            sentiment, cached_time = self.sentiment_cache[cache_key]
            if datetime.now() - cached_time < self.cache_expiry:
                return sentiment

        try:
            # 実際のNLP処理の代わりにダミー分析
            sentiment_score = self._dummy_sentiment_analysis(articles)

            # キャッシュに保存
            self.sentiment_cache[cache_key] = (sentiment_score, datetime.now())

            return sentiment_score

        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return 0.0  # ニュートラル

    def _dummy_sentiment_analysis(self, articles: List[str]) -> float:
        """ダミーのセンチメント分析"""
        if not articles:
            return 0.0

        # 簡単なキーワードベースの分析
        positive_words = ["growth", "profit", "increase", "bull", "upgrade", "strong"]
        negative_words = ["loss", "decline", "bear", "downgrade", "weak", "crisis"]

        total_sentiment = 0.0
        for article in articles:
            article_lower = article.lower()

            positive_count = sum(1 for word in positive_words if word in article_lower)
            negative_count = sum(1 for word in negative_words if word in article_lower)

            # センチメントスコア計算 (-1 to 1)
            if positive_count + negative_count > 0:
                article_sentiment = (positive_count - negative_count) / (
                    positive_count + negative_count
                )
            else:
                article_sentiment = 0.0

            total_sentiment += article_sentiment

        # 平均センチメント
        avg_sentiment = total_sentiment / len(articles) if articles else 0.0
        return np.clip(avg_sentiment, -1.0, 1.0)

    def analyze_social_media_sentiment(self, symbol: str, posts: List[str]) -> float:
        """ソーシャルメディア投稿のセンチメント分析"""
        cache_key = f"social_{symbol}_{hash(str(posts))}"

        # キャッシュチェック
        if cache_key in self.sentiment_cache:
            sentiment, cached_time = self.sentiment_cache[cache_key]
            if datetime.now() - cached_time < self.cache_expiry:
                return sentiment

        try:
            # ソーシャルメディア特有の分析
            sentiment_score = self._analyze_social_posts(posts)

            # キャッシュに保存
            self.sentiment_cache[cache_key] = (sentiment_score, datetime.now())

            return sentiment_score

        except Exception as e:
            logger.error(f"Social media sentiment analysis failed for {symbol}: {e}")
            return 0.0

    def _analyze_social_posts(self, posts: List[str]) -> float:
        """ソーシャルメディア投稿の分析"""
        if not posts:
            return 0.0

        # ソーシャルメディア特有のキーワード
        positive_emojis = ["🚀", "📈", "💎", "🔥", "👍", "✨"]
        negative_emojis = ["📉", "😭", "💀", "👎", "😰", "🔻"]

        bullish_phrases = ["to the moon", "diamond hands", "buy the dip", "hodl"]
        bearish_phrases = ["paper hands", "dump it", "sell off", "crash"]

        total_sentiment = 0.0
        for post in posts:
            post_lower = post.lower()

            # 絵文字カウント
            positive_emoji_count = sum(1 for emoji in positive_emojis if emoji in post)
            negative_emoji_count = sum(1 for emoji in negative_emojis if emoji in post)

            # フレーズカウント
            bullish_count = sum(1 for phrase in bullish_phrases if phrase in post_lower)
            bearish_count = sum(1 for phrase in bearish_phrases if phrase in post_lower)

            # 総合カウント
            positive_total = positive_emoji_count + bullish_count
            negative_total = negative_emoji_count + bearish_count

            if positive_total + negative_total > 0:
                post_sentiment = (positive_total - negative_total) / (
                    positive_total + negative_total
                )
            else:
                post_sentiment = 0.0

            total_sentiment += post_sentiment

        avg_sentiment = total_sentiment / len(posts) if posts else 0.0
        return np.clip(avg_sentiment, -1.0, 1.0)

    def get_sentiment_summary(
        self,
        symbol: str,
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        """センチメントサマリーを取得"""
        # ダミーデータを生成
        base_sentiment = np.random.normal(0, 0.3)

        return {
            "overall_sentiment": np.clip(base_sentiment, -1.0, 1.0),
            "news_sentiment": np.clip(
                base_sentiment + np.random.normal(0, 0.1),
                -1.0,
                1.0,
            ),
            "social_sentiment": np.clip(
                base_sentiment + np.random.normal(0, 0.2),
                -1.0,
                1.0,
            ),
            "analyst_sentiment": np.clip(
                base_sentiment + np.random.normal(0, 0.15),
                -1.0,
                1.0,
            ),
            "sentiment_momentum": np.random.normal(0, 0.1),
            "sentiment_volume": np.random.uniform(0.1, 1.0),
        }

    def calculate_sentiment_impact(
        self,
        current_sentiment: float,
        historical_sentiment: List[float],
    ) -> float:
        """センチメントの影響度を計算"""
        if not historical_sentiment:
            return 0.0

        # センチメントの変化を計算
        avg_historical = np.mean(historical_sentiment)
        sentiment_change = current_sentiment - avg_historical

        # ボラティリティを考慮
        sentiment_volatility = (
            np.std(historical_sentiment) if len(historical_sentiment) > 1 else 0.1
        )

        # 影響度計算（標準化）
        impact = sentiment_change / (sentiment_volatility + 0.01)

        return np.clip(impact, -3.0, 3.0)  # -3から3の範囲に制限

    def clear_cache(self):
        """キャッシュをクリア"""
        self.sentiment_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """キャッシュ統計を取得"""
        now = datetime.now()
        active_entries = sum(
            1
            for _, cached_time in self.sentiment_cache.values()
            if now - cached_time < self.cache_expiry
        )

        return {
            "total_cached_items": len(self.sentiment_cache),
            "active_cached_items": active_entries,
            "expired_items": len(self.sentiment_cache) - active_entries,
        }
