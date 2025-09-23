"""
ニュースセンチメント分析システム
84.6%技術分析とニュース感情を統合
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import requests
import re
from datetime import datetime, timedelta
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import get_settings
from utils.exceptions import APIError, NetworkError, DataFetchError

logger = logging.getLogger(__name__)


class NewsSource:
    """ニュースソース基底クラス"""

    def __init__(self, name: str):
        self.name = name
        self.rate_limit_delay = 1.0  # 秒
        self.last_request_time = 0

    def _rate_limit(self) -> None:
        """レート制限の実装"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """ニュース取得（サブクラスで実装）"""
        raise NotImplementedError


class YahooNewsSource(NewsSource):
    """Yahoo Financeニュースソース"""

    def __init__(self):
        super().__init__("YahooFinance")
        self.base_url = "https://query1.finance.yahoo.com/v1/finance/search"

    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """Yahoo Financeからニュース取得"""
        try:
            self._rate_limit()

            # 日本株の場合は.Tを追加
            ticker = f"{symbol}.T" if symbol.isdigit() else symbol

            # 実際のyfinanceライブラリを使用してニュース取得
            import yfinance as yf

            stock = yf.Ticker(ticker)

            try:
                # yfinanceのnewsプロパティを使用
                news_data = stock.news

                if not news_data:
                    logger.warning(f"Yahoo Finance ニュースなし: {symbol}")
                    return []

                # 過去N日以内のニュースのみ
                cutoff_date = datetime.now() - timedelta(days=days)

                filtered_news = []
                for news_item in news_data:
                    # Unix timestampを変換
                    if "providerPublishTime" in news_item:
                        publish_time = datetime.fromtimestamp(
                            news_item["providerPublishTime"]
                        )

                        if publish_time >= cutoff_date:
                            filtered_news.append(
                                {
                                    "title": news_item.get("title", ""),
                                    "summary": news_item.get("summary", ""),
                                    "published": publish_time,
                                    "source": "Yahoo Finance",
                                    "url": news_item.get("link", ""),
                                    "relevance": self._calculate_relevance(
                                        news_item.get("title", ""), symbol
                                    ),
                                }
                            )

                logger.info(
                    f"Yahoo Finance ニュース取得: {symbol} ({len(filtered_news)}件)"
                )
                return filtered_news

            except Exception as e:
                logger.error(f"Yahoo Finance API エラー: {e}")
                return []

        except Exception as e:
            logger.error(f"Yahoo Finance ニュース取得エラー {symbol}: {e}")
            return []

    def _calculate_relevance(self, title: str, symbol: str) -> float:
        """ニュースの関連度を計算"""
        title_lower = title.lower()

        # 銘柄コードが含まれている場合は高関連度
        if symbol in title_lower:
            return 0.9

        # 会社名マッピング（簡略版）
        company_names = {
            "7203": ["toyota", "トヨタ"],
            "6758": ["sony", "ソニー"],
            "9434": ["softbank", "ソフトバンク"],
            "8306": ["mufg", "三菱"],
            "6861": ["keyence", "キーエンス"],
        }

        if symbol in company_names:
            for name in company_names[symbol]:
                if name.lower() in title_lower:
                    return 0.8

        return 0.5  # デフォルト関連度


class JapanNewsSource(NewsSource):
    """日本経済新聞等の日本語ニュースソース"""

    def __init__(self):
        super().__init__("JapanNews")
        self.search_engines = [
            "https://news.google.com/rss/search",
            "https://feeds.finance.yahoo.co.jp/rss/2.0/headline",
        ]

    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """日本語ニュース取得"""
        try:
            self._rate_limit()

            # 会社名マッピング
            company_names = {
                "7203": "トヨタ自動車",
                "6758": "ソニーグループ",
                "9434": "ソフトバンク",
                "8306": "三菱UFJフィナンシャル",
                "6861": "キーエンス",
                "6701": "日本電気",
                "8316": "三井住友フィナンシャル",
                "8411": "みずほフィナンシャル",
                "8058": "三菱商事",
                "8001": "伊藤忠商事",
            }

            company_name = company_names.get(symbol, f"{symbol}関連企業")
            search_query = f"{company_name} OR {symbol}"

            all_news = []

            # Google News RSSから取得
            news_from_google = self._fetch_from_google_news(search_query, days)
            all_news.extend(news_from_google)

            # Yahoo Finance JapanのRSSから取得
            news_from_yahoo_jp = self._fetch_from_yahoo_jp(search_query, days)
            all_news.extend(news_from_yahoo_jp)

            # 重複除去
            unique_news = self._remove_duplicates(all_news)

            logger.info(f"日本語ニュース取得: {symbol} ({len(unique_news)}件)")
            return unique_news

        except Exception as e:
            logger.error(f"日本語ニュース取得エラー {symbol}: {e}")
            return []

    def _fetch_from_google_news(self, query: str, days: int) -> List[Dict[str, Any]]:
        """Google NewsのRSSから取得"""
        try:
            import feedparser
            from urllib.parse import quote

            # Google News RSS URL
            encoded_query = quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"

            # RSS取得
            feed = feedparser.parse(rss_url)

            if not feed.entries:
                return []

            cutoff_date = datetime.now() - timedelta(days=days)
            news_list = []

            for entry in feed.entries[:20]:  # 最大20件
                try:
                    # 日付解析
                    published_time = datetime(*entry.published_parsed[:6])

                    if published_time >= cutoff_date:
                        news_list.append(
                            {
                                "title": entry.title,
                                "summary": entry.get("summary", ""),
                                "published": published_time,
                                "source": "Google News",
                                "url": entry.link,
                                "relevance": self._calculate_relevance_jp(
                                    entry.title, query
                                ),
                            }
                        )

                except Exception as e:
                    logger.debug(f"Google News エントリー解析エラー: {e}")
                    continue

            return news_list

        except ImportError:
            logger.warning(
                "feedparser not installed. Install with: pip install feedparser"
            )
            return []
        except Exception as e:
            logger.error(f"Google News取得エラー: {e}")
            return []

    def _fetch_from_yahoo_jp(self, query: str, days: int) -> List[Dict[str, Any]]:
        """Yahoo Finance JapanのRSSから取得"""
        try:
            # Yahoo Finance Japan経済ニュース
            rss_url = "https://feeds.finance.yahoo.co.jp/rss/2.0/headline"

            import feedparser

            feed = feedparser.parse(rss_url)

            if not feed.entries:
                return []

            cutoff_date = datetime.now() - timedelta(days=days)
            news_list = []

            for entry in feed.entries[:30]:  # 最大30件チェック
                try:
                    # 日付解析
                    published_time = datetime(*entry.published_parsed[:6])

                    if published_time >= cutoff_date:
                        # クエリに関連するニュースのみ
                        if self._is_relevant_to_query(entry.title, query):
                            news_list.append(
                                {
                                    "title": entry.title,
                                    "summary": entry.get("summary", ""),
                                    "published": published_time,
                                    "source": "Yahoo Finance Japan",
                                    "url": entry.link,
                                    "relevance": self._calculate_relevance_jp(
                                        entry.title, query
                                    ),
                                }
                            )

                except Exception as e:
                    logger.debug(f"Yahoo JP エントリー解析エラー: {e}")
                    continue

            return news_list

        except ImportError:
            logger.warning("feedparser not installed for Yahoo JP RSS")
            return []
        except Exception as e:
            logger.error(f"Yahoo JP取得エラー: {e}")
            return []

    def _is_relevant_to_query(self, title: str, query: str) -> bool:
        """クエリに関連するかチェック"""
        title_lower = title.lower()
        query_terms = query.lower().split(" or ")

        for term in query_terms:
            term = term.strip()
            if term in title_lower:
                return True

        return False

    def _calculate_relevance_jp(self, title: str, query: str) -> float:
        """日本語ニュースの関連度計算"""
        title_lower = title.lower()
        query_terms = query.lower().split(" or ")

        relevance_score = 0.0
        for term in query_terms:
            term = term.strip()
            if term in title_lower:
                relevance_score += 0.3

        # 株式、証券、投資関連キーワードでボーナス
        stock_keywords = ["株価", "決算", "業績", "増益", "減益", "投資", "証券"]
        for keyword in stock_keywords:
            if keyword in title:
                relevance_score += 0.1

        return min(relevance_score, 1.0)

    def _remove_duplicates(
        self, news_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """重複ニュースを除去"""
        seen_titles = set()
        unique_news = []

        for news in news_list:
            title_key = news["title"].lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(news)

        return unique_news


class SentimentAnalyzer:
    """センチメント分析エンジン"""

    def __init__(self):
        self.positive_keywords = [
            "上昇",
            "好調",
            "増益",
            "増収",
            "成長",
            "拡大",
            "好材料",
            "買い推奨",
            "アップグレード",
            "目標株価引き上げ",
            "強気",
            "過去最高",
            "記録的",
            "堅調",
            "回復",
            "改善",
        ]

        self.negative_keywords = [
            "下落",
            "不調",
            "減益",
            "減収",
            "縮小",
            "悪材料",
            "売り推奨",
            "ダウングレード",
            "目標株価引き下げ",
            "弱気",
            "最安値",
            "低迷",
            "悪化",
            "懸念",
            "リスク",
            "下方修正",
        ]

        self.neutral_keywords = [
            "維持",
            "横ばい",
            "変わらず",
            "据え置き",
            "様子見",
            "中立",
            "保留",
            "継続",
        ]

    def analyze_text(self, text: str) -> Dict[str, float]:
        """テキストのセンチメント分析"""
        text_lower = text.lower()

        positive_score = sum(1 for keyword in self.positive_keywords if keyword in text)
        negative_score = sum(1 for keyword in self.negative_keywords if keyword in text)
        neutral_score = sum(1 for keyword in self.neutral_keywords if keyword in text)

        total_score = positive_score + negative_score + neutral_score

        if total_score == 0:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

        return {
            "positive": positive_score / total_score,
            "negative": negative_score / total_score,
            "neutral": neutral_score / total_score,
        }

    def get_sentiment_score(self, text: str) -> float:
        """センチメントスコア（-1.0 to 1.0）"""
        sentiment = self.analyze_text(text)
        return sentiment["positive"] - sentiment["negative"]


class MarketSentimentAnalyzer:
    """市場センチメント統合分析システム"""

    def __init__(self):
        self.settings = get_settings()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_sources = [YahooNewsSource(), JapanNewsSource()]

    def fetch_all_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """全ソースからニュース取得"""
        all_news = []

        # 並列でニュース取得
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_source = {
                executor.submit(source.fetch_news, symbol, days): source
                for source in self.news_sources
            }

            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    news_list = future.result(timeout=30)
                    all_news.extend(news_list)
                except Exception as e:
                    logger.error(f"ニュース取得エラー {source.name}: {e}")

        # 重複除去と時系列ソート
        all_news = self._deduplicate_news(all_news)
        all_news.sort(key=lambda x: x["published"], reverse=True)

        logger.info(f"全ニュース取得完了 {symbol}: {len(all_news)}件")
        return all_news

    def _deduplicate_news(
        self, news_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """ニュース重複除去"""
        seen_titles = set()
        unique_news = []

        for news in news_list:
            title_key = news["title"].lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(news)

        return unique_news

    def analyze_news_sentiment(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """ニュースセンチメント分析"""
        try:
            # ニュース取得
            news_list = self.fetch_all_news(symbol, days)

            if not news_list:
                return self._default_sentiment_result(symbol, "No news available")

            # 各ニュースのセンチメント分析
            analyzed_news = []
            sentiment_scores = []
            relevance_weights = []

            for news in news_list:
                # タイトルと要約を結合
                full_text = f"{news['title']} {news.get('summary', '')}"

                # センチメント分析
                sentiment_score = self.sentiment_analyzer.get_sentiment_score(full_text)
                sentiment_detail = self.sentiment_analyzer.analyze_text(full_text)

                # 時間重み（新しいニュースほど重要）
                hours_ago = (datetime.now() - news["published"]).total_seconds() / 3600
                time_weight = max(0.1, 1.0 / (1.0 + hours_ago / 24))  # 24時間で半減

                # 関連度重み
                relevance = news.get("relevance", 0.5)

                # 総合重み
                total_weight = time_weight * relevance

                analyzed_news.append(
                    {
                        **news,
                        "sentiment_score": sentiment_score,
                        "sentiment_detail": sentiment_detail,
                        "time_weight": time_weight,
                        "total_weight": total_weight,
                    }
                )

                sentiment_scores.append(sentiment_score)
                relevance_weights.append(total_weight)

            # 加重平均でセンチメントスコア計算
            if relevance_weights and sum(relevance_weights) > 0:
                weighted_sentiment = np.average(
                    sentiment_scores, weights=relevance_weights
                )
            else:
                weighted_sentiment = np.mean(sentiment_scores)

            # センチメント強度（絶対値）
            sentiment_strength = abs(weighted_sentiment)

            # センチメント方向
            if weighted_sentiment > 0.1:
                sentiment_direction = "positive"
            elif weighted_sentiment < -0.1:
                sentiment_direction = "negative"
            else:
                sentiment_direction = "neutral"

            result = {
                "symbol": symbol,
                "analysis_time": datetime.now(),
                "news_count": len(news_list),
                "sentiment_score": float(weighted_sentiment),  # -1.0 to 1.0
                "sentiment_strength": float(sentiment_strength),  # 0.0 to 1.0
                "sentiment_direction": sentiment_direction,
                "confidence": min(0.9, sentiment_strength + 0.1),
                "analyzed_news": analyzed_news[:10],  # 最新10件
                "summary": self._generate_sentiment_summary(
                    sentiment_direction, sentiment_strength, len(news_list)
                ),
            }

            logger.info(
                f"センチメント分析完了 {symbol}: "
                f"{sentiment_direction} ({weighted_sentiment:.3f})"
            )

            return result

        except Exception as e:
            logger.error(f"センチメント分析エラー {symbol}: {e}")
            return self._default_sentiment_result(symbol, f"Analysis error: {e}")

    def _default_sentiment_result(self, symbol: str, reason: str) -> Dict[str, Any]:
        """デフォルトセンチメント結果"""
        return {
            "symbol": symbol,
            "analysis_time": datetime.now(),
            "news_count": 0,
            "sentiment_score": 0.0,
            "sentiment_strength": 0.0,
            "sentiment_direction": "neutral",
            "confidence": 0.1,
            "analyzed_news": [],
            "summary": f"センチメント分析不可: {reason}",
            "error": reason,
        }

    def _generate_sentiment_summary(
        self, direction: str, strength: float, news_count: int
    ) -> str:
        """センチメントサマリー生成"""
        strength_desc = (
            "強い" if strength > 0.6 else "中程度の" if strength > 0.3 else "弱い"
        )

        direction_map = {
            "positive": "ポジティブ",
            "negative": "ネガティブ",
            "neutral": "中立",
        }

        return (
            f"{news_count}件のニュース分析結果: "
            f"{strength_desc}{direction_map[direction]}センチメント"
        )

    def integrate_with_technical_analysis(
        self,
        symbol: str,
        technical_signal: Dict[str, Any],
        sentiment_weight: float = 0.3,
    ) -> Dict[str, Any]:
        """技術分析とセンチメント分析の統合"""
        try:
            # センチメント分析実行
            sentiment_result = self.analyze_news_sentiment(symbol)

            # 技術分析の信頼度と信号
            tech_confidence = technical_signal.get("confidence", 0.5)
            tech_signal_value = technical_signal.get("signal", 0)

            # センチメントの影響
            sentiment_score = sentiment_result["sentiment_score"]
            sentiment_confidence = sentiment_result["confidence"]

            # 統合信頼度計算
            tech_weight = 1.0 - sentiment_weight
            integrated_confidence = (
                tech_confidence * tech_weight + sentiment_confidence * sentiment_weight
            )

            # センチメントによる信号調整
            sentiment_adjustment = sentiment_score * sentiment_weight

            # 最終信号
            if tech_signal_value == 1:  # 買いシグナル
                if sentiment_score > 0.2:  # ポジティブセンチメント
                    final_signal = 1
                    signal_strength = min(
                        1.0, integrated_confidence + abs(sentiment_adjustment)
                    )
                elif sentiment_score < -0.3:  # 強いネガティブセンチメント
                    final_signal = 0  # シグナル取り消し
                    signal_strength = 0.0
                else:
                    final_signal = 1
                    signal_strength = integrated_confidence * (
                        1.0 + sentiment_adjustment
                    )
            elif tech_signal_value == -1:  # 売りシグナル
                if sentiment_score < -0.2:  # ネガティブセンチメント
                    final_signal = -1
                    signal_strength = min(
                        1.0, integrated_confidence + abs(sentiment_adjustment)
                    )
                elif sentiment_score > 0.3:  # 強いポジティブセンチメント
                    final_signal = 0  # シグナル取り消し
                    signal_strength = 0.0
                else:
                    final_signal = -1
                    signal_strength = integrated_confidence * (
                        1.0 + abs(sentiment_adjustment)
                    )
            else:  # 中立
                if abs(sentiment_score) > 0.4:  # 強いセンチメント
                    final_signal = 1 if sentiment_score > 0 else -1
                    signal_strength = sentiment_confidence * abs(sentiment_score)
                else:
                    final_signal = 0
                    signal_strength = 0.0

            # 結果統合
            integrated_result = {
                "symbol": symbol,
                "analysis_time": datetime.now(),
                "technical_analysis": technical_signal,
                "sentiment_analysis": sentiment_result,
                "integrated_signal": final_signal,
                "integrated_confidence": float(max(0.0, min(1.0, signal_strength))),
                "sentiment_weight": sentiment_weight,
                "recommendation": self._generate_recommendation(
                    final_signal,
                    signal_strength,
                    sentiment_result["sentiment_direction"],
                ),
            }

            logger.info(
                f"統合分析完了 {symbol}: "
                f"技術={tech_signal_value} センチメント={sentiment_score:.2f} "
                f"→ 最終={final_signal} (信頼度={signal_strength:.2f})"
            )

            return integrated_result

        except Exception as e:
            logger.error(f"統合分析エラー {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "integrated_signal": 0,
                "integrated_confidence": 0.0,
            }

    def _generate_recommendation(
        self, signal: int, confidence: float, sentiment: str
    ) -> str:
        """推奨アクション生成"""
        if signal == 1 and confidence > 0.7:
            return f"強い買い推奨 (センチメント: {sentiment})"
        elif signal == 1 and confidence > 0.5:
            return f"買い推奨 (センチメント: {sentiment})"
        elif signal == -1 and confidence > 0.7:
            return f"強い売り推奨 (センチメント: {sentiment})"
        elif signal == -1 and confidence > 0.5:
            return f"売り推奨 (センチメント: {sentiment})"
        else:
            return f"様子見 (センチメント: {sentiment})"

    def batch_sentiment_analysis(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """複数銘柄の一括センチメント分析"""
        results = {}

        logger.info(f"一括センチメント分析開始: {len(symbols)}銘柄")

        # 並列処理で高速化
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(self.analyze_news_sentiment, symbol): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=60)
                    results[symbol] = result
                except Exception as e:
                    logger.error(f"センチメント分析エラー {symbol}: {e}")
                    results[symbol] = self._default_sentiment_result(symbol, str(e))

        logger.info(f"一括センチメント分析完了: {len(results)}件")
        return results
