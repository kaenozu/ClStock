#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
フルオート投資システム
完全自動化：TSE4000最適化 → 学習・訓練 → 売買タイミング提示
"""

import logging
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

# 既存システムのインポート
from tse_4000_optimizer import TSE4000Optimizer
from models_new.hybrid.hybrid_predictor import HybridStockPredictor
from models_new.hybrid.prediction_modes import PredictionMode
from models_new.advanced.market_sentiment_analyzer import MarketSentimentAnalyzer
from models_new.advanced.trading_strategy_generator import (
    AutoTradingStrategyGenerator,
    ActionType,
)
from models_new.advanced.risk_management_framework import RiskManager
from data.stock_data import StockDataProvider


@dataclass
class AutoRecommendation:
    """自動投資推奨"""

    symbol: str
    company_name: str
    action: ActionType
    entry_price: float
    target_price: float
    stop_loss: float
    buy_date: datetime
    sell_date: datetime
    expected_return: float
    confidence: float
    reasoning: str
    risk_level: str


class FullAutoInvestmentSystem:
    """
    フルオート投資システム

    特徴:
    - 完全自動化（ユーザー判断不要）
    - TSE4000最適化自動実行
    - 学習・訓練自動実施
    - 売買タイミング自動算出
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # サブシステム初期化
        self.tse_optimizer = TSE4000Optimizer()
        self.hybrid_predictor = HybridStockPredictor(
            enable_cache=True,
            enable_adaptive_optimization=True,
            enable_streaming=True,
            enable_multi_gpu=True,
            enable_real_time_learning=True,
        )
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.strategy_generator = AutoTradingStrategyGenerator()
        self.risk_manager = RiskManager()
        self.data_provider = StockDataProvider()

        # 自動化設定
        self.auto_settings = {
            "portfolio_size": 10,  # 推奨銘柄数
            "investment_period_days": 30,  # 投資期間（日）
            "min_confidence": 0.7,  # 最小信頼度
            "max_risk_score": 2.5,  # 最大リスクスコア
            "rebalance_threshold": 0.1,  # リバランス閾値
        }

        self.logger.info("FullAutoInvestmentSystem initialized")

    async def run_full_auto_analysis(self) -> List[AutoRecommendation]:
        """フルオート分析実行"""
        self.logger.info("🚀 フルオート投資システム開始")

        # 市場時間チェック
        if not self._show_market_hours_warning():
            print("処理を中断しました。")
            return []

        # 進捗表示の初期化
        total_steps = 4
        current_step = 0

        def show_progress(step_name: str, step_num: int):
            nonlocal current_step
            current_step = step_num
            progress = (current_step / total_steps) * 100
            print(
                f"\n[進捗] [{current_step}/{total_steps}] ({progress:.0f}%) - {step_name}"
            )
            print("=" * 60)

        try:
            # Step 1: TSE4000最適化（必要に応じて）
            show_progress("TSE4000最適化実行中...", 1)
            optimized_symbols = await self._auto_tse4000_optimization()
            print(f"[完了] 最適化完了: {len(optimized_symbols)}銘柄選出")

            # Step 2: 学習・訓練自動実施
            show_progress("学習・訓練実行中...", 2)
            await self._auto_learning_and_training(optimized_symbols)
            print("[完了] 学習・訓練完了")

            # Step 3: 総合分析と推奨生成
            show_progress("総合分析・推奨生成中...", 3)
            recommendations = await self._generate_auto_recommendations(
                optimized_symbols
            )
            print(f"[完了] 推奨生成完了: {len(recommendations)}件")

            # Step 4: 結果表示
            show_progress("結果表示", 4)
            self._display_recommendations(recommendations)
            print("[完了] フルオート分析完了！")

            return recommendations

        except Exception as e:
            print(f"\n[エラー] フルオート分析失敗: {str(e)}")
            self.logger.error(f"フルオート分析失敗: {str(e)}")
            return []

    async def _auto_tse4000_optimization(self) -> List[str]:
        """TSE4000自動最適化（強化版エラーハンドリング）"""
        self.logger.info("📊 TSE4000最適化実行中...")
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # 前回最適化からの経過時間チェック
                need_optimization = self._check_optimization_necessity()

                if need_optimization:
                    print(
                        f"[実行] 最適化が必要です。実行中... (試行 {retry_count + 1}/{max_retries})"
                    )

                    # TSE4000最適化実行（タイムアウト付き）
                    try:
                        optimization_result = await asyncio.wait_for(
                            asyncio.create_task(self._run_tse_optimization_async()),
                            timeout=300,  # 5分タイムアウト
                        )
                    except asyncio.TimeoutError:
                        print("[タイムアウト] TSE4000最適化がタイムアウトしました")
                        raise Exception("TSE4000最適化タイムアウト")

                    if (
                        optimization_result
                        and "optimized_portfolio" in optimization_result
                    ):
                        symbols = [
                            stock.symbol
                            for stock in optimization_result["optimized_portfolio"]
                        ]
                        selected_symbols = symbols[
                            : self.auto_settings["portfolio_size"]
                        ]

                        if len(selected_symbols) >= 5:  # 最低5銘柄は必要
                            # 最適化履歴を記録
                            self._save_optimization_history(selected_symbols)

                            print(f"[完了] 最適化完了: {len(selected_symbols)}銘柄選出")
                            self.logger.info(
                                f"✅ 最適化完了: {len(selected_symbols)}銘柄選出"
                            )
                            return selected_symbols
                        else:
                            raise Exception(
                                f"選出銘柄数不足: {len(selected_symbols)}銘柄"
                            )
                    else:
                        raise Exception("最適化結果が無効または空です")

                else:
                    print("[使用] 前回の最適化結果を使用")
                    self.logger.info("📋 前回の最適化結果を使用")
                    # 前回の結果を読み込み
                    previous_symbols = self._load_previous_optimization()
                    if len(previous_symbols) >= 5:
                        return previous_symbols
                    else:
                        print("[警告] 前回結果も不足。デフォルト銘柄使用")
                        return self._get_default_symbols()

            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                print(
                    f"[エラー] 最適化エラー (試行 {retry_count}/{max_retries}): {error_msg}"
                )
                self.logger.warning(
                    f"TSE4000最適化エラー (試行 {retry_count}): {error_msg}"
                )

                if retry_count < max_retries:
                    print(f"[待機] {5}秒後に再試行します...")
                    await asyncio.sleep(5)
                else:
                    print("[安全] 最大試行回数到達。デフォルト銘柄を使用します")
                    self.logger.error(f"TSE4000最適化最終失敗: {error_msg}")
                    return self._get_default_symbols()

        return self._get_default_symbols()

    async def _run_tse_optimization_async(self):
        """TSE4000最適化の非同期実行"""
        # 同期的なTSE4000最適化を非同期で実行
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.tse_optimizer.run_comprehensive_optimization
        )

    def _check_market_hours(self) -> Tuple[bool, str]:
        """市場時間チェック"""
        try:
            current_time = datetime.now()
            weekday = current_time.weekday()  # 0=月曜, 6=日曜
            hour = current_time.hour
            minute = current_time.minute

            # 土日は市場休業
            if weekday >= 5:  # 土曜日・日曜日
                next_monday = current_time + timedelta(days=(7 - weekday))
                return (
                    False,
                    f"市場休業日です。次回開場: {next_monday.strftime('%m/%d（月）9:00')}",
                )

            # 平日の取引時間チェック
            # 東証: 9:00-11:30（前場）、12:30-15:00（後場）
            current_minutes = hour * 60 + minute

            # 前場: 9:00-11:30 (540-690分)
            morning_start = 9 * 60  # 540分
            morning_end = 11 * 60 + 30  # 690分

            # 後場: 12:30-15:00 (750-900分)
            afternoon_start = 12 * 60 + 30  # 750分
            afternoon_end = 15 * 60  # 900分

            if morning_start <= current_minutes <= morning_end:
                return True, "前場取引時間中"
            elif afternoon_start <= current_minutes <= afternoon_end:
                return True, "後場取引時間中"
            elif current_minutes < morning_start:
                return False, f"市場開場前です。開場時刻: 9:00"
            elif morning_end < current_minutes < afternoon_start:
                return False, f"昼休み時間です。後場開始: 12:30"
            else:  # current_minutes > afternoon_end
                next_day = current_time + timedelta(days=1)
                if next_day.weekday() >= 5:  # 翌日が土日
                    next_monday = current_time + timedelta(
                        days=(7 - current_time.weekday())
                    )
                    return (
                        False,
                        f"市場終了後です。次回開場: {next_monday.strftime('%m/%d（月）9:00')}",
                    )
                else:
                    return (
                        False,
                        f"市場終了後です。次回開場: {next_day.strftime('%m/%d 9:00')}",
                    )

        except Exception as e:
            self.logger.warning(f"市場時間チェックエラー: {e}")
            return True, "市場時間チェック無効（処理継続）"

    def _show_market_hours_warning(self) -> bool:
        """市場時間外の警告表示"""
        is_open, message = self._check_market_hours()

        if not is_open:
            print(f"\n[警告] {message}")
            print("[注意] 市場時間外でも分析は実行できますが、")
            print("       実際の取引は市場開場時間内に行ってください。")

            while True:
                choice = input("\n続行しますか？ (y/n): ").strip().lower()
                if choice in ["y", "yes", "はい"]:
                    return True
                elif choice in ["n", "no", "いいえ"]:
                    return False
                else:
                    print("yまたはnで入力してください。")
        else:
            print(f"[OK] {message}")
            return True

    def _check_optimization_necessity(self) -> bool:
        """最適化必要性判定"""
        try:
            # 最適化履歴ファイルを確認
            history_file = "tse4000_optimization_history.json"

            if not os.path.exists(history_file):
                # 履歴ファイルがない場合は最適化実行
                return True

            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)

            if not history or "last_optimization" not in history:
                return True

            # 前回最適化日時を取得
            last_optimization = datetime.fromisoformat(history["last_optimization"])
            current_time = datetime.now()

            # 前回最適化から経過日数
            days_since_last = (current_time - last_optimization).days

            # 3日以上経過している場合は最適化実行
            if days_since_last >= 3:
                return True

            # 月曜日または金曜日で前日以降に最適化していない場合
            weekday = current_time.weekday()
            if weekday in [0, 4]:  # 月曜・金曜
                # 前回最適化が昨日より前なら実行
                if days_since_last >= 1:
                    return True

            return False

        except Exception as e:
            self.logger.warning(f"最適化履歴確認エラー: {e}")
            # エラー時は安全のため最適化実行
            return True  # 月曜・金曜

    def _save_optimization_history(self, symbols: List[str]):
        """最適化履歴を保存"""
        try:
            history = {
                "last_optimization": datetime.now().isoformat(),
                "symbols": symbols,
                "symbol_count": len(symbols),
                "optimization_type": "tse4000_auto",
            }

            history_file = "tse4000_optimization_history.json"
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            self.logger.info(f"最適化履歴保存: {len(symbols)}銘柄")

        except Exception as e:
            self.logger.error(f"履歴保存エラー: {e}")

    def _load_previous_optimization(self) -> List[str]:
        """前回の最適化結果を読み込み"""
        try:
            history_file = "tse4000_optimization_history.json"

            if not os.path.exists(history_file):
                self.logger.info("履歴ファイルなし。デフォルト銘柄使用")
                return self._get_default_symbols()

            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)

            if "symbols" in history and history["symbols"]:
                symbols = history["symbols"]
                self.logger.info(f"前回最適化結果読み込み: {len(symbols)}銘柄")
                return symbols

        except Exception as e:
            self.logger.error(f"履歴読み込みエラー: {e}")

        return self._get_default_symbols()

    def _get_default_symbols(self) -> List[str]:
        """デフォルト銘柄リスト"""
        return [
            "6758.T",  # ソニーグループ
            "7203.T",  # トヨタ自動車
            "8306.T",  # 三菱UFJ銀行
            "4502.T",  # 武田薬品工業
            "9984.T",  # ソフトバンクグループ
            "6861.T",  # キーエンス
            "7974.T",  # 任天堂
            "4689.T",  # ヤフー
            "8035.T",  # 東京エレクトロン
            "6098.T",  # リクルートホールディングス
        ]

    async def _auto_learning_and_training(self, symbols: List[str]):
        """自動学習・訓練実施"""
        self.logger.info("🧠 学習・訓練自動実施中...")

        try:
            # 並列で各銘柄の学習実行
            learning_tasks = []

            for symbol in symbols:
                task = self._learn_single_symbol(symbol)
                learning_tasks.append(task)

            # 並列実行
            learning_results = await asyncio.gather(
                *learning_tasks, return_exceptions=True
            )

            successful_learning = len(
                [r for r in learning_results if not isinstance(r, Exception)]
            )
            self.logger.info(f"✅ 学習完了: {successful_learning}/{len(symbols)}銘柄")

        except Exception as e:
            self.logger.error(f"学習・訓練エラー: {str(e)}")

    async def _learn_single_symbol(self, symbol: str) -> Dict[str, Any]:
        """単一銘柄学習"""
        try:
            # 価格データ取得
            price_data = self.data_provider.get_stock_data(symbol)

            if price_data.empty:
                return {"symbol": symbol, "status": "no_data"}

            # 予測実行（学習効果込み）- 非同期で実行
            loop = asyncio.get_event_loop()
            prediction_result = await loop.run_in_executor(
                None,
                self.hybrid_predictor.predict,
                symbol,
                PredictionMode.RESEARCH_MODE,
            )

            # 実時間学習システムにデータ追加
            if (
                hasattr(self.hybrid_predictor, "real_time_learning_enabled")
                and self.hybrid_predictor.real_time_learning_enabled
            ):
                market_data = {
                    "symbol": symbol,
                    "price": (
                        price_data["Close"].iloc[-1] if "Close" in price_data else 1000
                    ),
                    "volume": (
                        price_data["Volume"].iloc[-1]
                        if "Volume" in price_data
                        else 100000
                    ),
                    "timestamp": datetime.now(),
                }
                # 実時間学習も非同期で実行
                if hasattr(self.hybrid_predictor, "process_real_time_market_data"):
                    try:
                        await loop.run_in_executor(
                            None,
                            self.hybrid_predictor.process_real_time_market_data,
                            market_data,
                        )
                    except Exception as rt_error:
                        self.logger.warning(f"実時間学習エラー {symbol}: {rt_error}")

            return {
                "symbol": symbol,
                "status": "success",
                "prediction": (
                    prediction_result.prediction if prediction_result else None
                ),
            }

        except Exception as e:
            self.logger.error(f"銘柄{symbol}学習エラー: {str(e)}")
            return {"symbol": symbol, "status": "error", "error": str(e)}

    async def _generate_auto_recommendations(
        self, symbols: List[str]
    ) -> List[AutoRecommendation]:
        """自動推奨生成"""
        self.logger.info("🎯 投資推奨自動生成中...")

        recommendations = []

        try:
            for symbol in symbols:
                recommendation = await self._analyze_single_symbol(symbol)
                if recommendation:
                    recommendations.append(recommendation)

            # リスク・リターンでソート
            recommendations.sort(key=lambda x: x.expected_return, reverse=True)

            self.logger.info(f"✅ 推奨生成完了: {len(recommendations)}銘柄")
            return recommendations

        except Exception as e:
            self.logger.error(f"推奨生成エラー: {str(e)}")
            return recommendations

    async def _analyze_single_symbol(self, symbol: str) -> Optional[AutoRecommendation]:
        """単一銘柄分析"""
        try:
            # 価格データ取得
            price_data = self.data_provider.get_stock_data(symbol)

            if price_data.empty:
                return None

            current_price = price_data["Close"].iloc[-1]

            # 1. 予測実行 - 非同期で実行
            loop = asyncio.get_event_loop()
            prediction_result = await loop.run_in_executor(
                None, self.hybrid_predictor.predict, symbol, PredictionMode.AUTO
            )

            if not prediction_result:
                return None

            # 2. センチメント分析
            sentiment_result = self.sentiment_analyzer.analyze_comprehensive_sentiment(
                symbol=symbol, price_data=price_data
            )

            # 3. 戦略生成
            strategies = self.strategy_generator.generate_comprehensive_strategy(
                symbol, price_data
            )
            signals = self.strategy_generator.generate_trading_signals(
                symbol,
                price_data,
                sentiment_data={
                    "current_sentiment": {"score": sentiment_result.sentiment_score}
                },
            )

            # 4. リスク分析
            portfolio_data = {"positions": {symbol: 100000}, "total_value": 100000}
            risk_analysis = self.risk_manager.analyze_portfolio_risk(
                portfolio_data, {symbol: price_data}
            )

            # 5. 売買タイミング計算
            buy_timing, sell_timing = self._calculate_optimal_timing(
                prediction_result, sentiment_result, signals
            )

            # 6. 総合判定
            if self._should_recommend(
                prediction_result, sentiment_result, risk_analysis
            ):
                return self._create_recommendation(
                    symbol,
                    current_price,
                    prediction_result,
                    sentiment_result,
                    buy_timing,
                    sell_timing,
                    risk_analysis,
                )

            return None

        except Exception as e:
            self.logger.error(f"銘柄{symbol}分析エラー: {str(e)}")
            return None

    def _calculate_optimal_timing(
        self, prediction, sentiment, signals
    ) -> Tuple[datetime, datetime]:
        """最適売買タイミング計算"""
        current_time = datetime.now()

        # 買いタイミング
        if sentiment.sentiment_score > 0.3 and prediction.confidence > 0.7:
            # ポジティブな場合は早めのエントリー
            buy_date = current_time + timedelta(days=1)
        elif sentiment.sentiment_score > 0:
            # 軽微ポジティブなら2-3日様子見
            buy_date = current_time + timedelta(days=2)
        else:
            # ネガティブなら1週間待機
            buy_date = current_time + timedelta(days=7)

        # 売りタイミング（投資期間ベース）
        base_hold_period = self.auto_settings["investment_period_days"]

        # 信頼度による調整
        if prediction.confidence > 0.8:
            hold_period = base_hold_period + 10  # 高信頼度なら長期保有
        elif prediction.confidence < 0.6:
            hold_period = base_hold_period - 10  # 低信頼度なら早期売却
        else:
            hold_period = base_hold_period

        sell_date = buy_date + timedelta(days=hold_period)

        return buy_date, sell_date

    def _should_recommend(self, prediction, sentiment, risk_analysis) -> bool:
        """推奨判定"""
        # 最小信頼度チェック
        if prediction.confidence < self.auto_settings["min_confidence"]:
            return False

        # リスクスコアチェック
        if risk_analysis.total_risk_score > self.auto_settings["max_risk_score"]:
            return False

        # 予測価格上昇チェック
        if prediction.prediction <= 0:
            return False

        return True

    def _create_recommendation(
        self,
        symbol: str,
        current_price: float,
        prediction,
        sentiment,
        buy_timing: datetime,
        sell_timing: datetime,
        risk_analysis,
    ) -> AutoRecommendation:
        """推奨情報作成"""

        # 目標価格（予測価格ベース）
        target_price = prediction.prediction

        # ストップロス（5%下）
        stop_loss = current_price * 0.95

        # 期待リターン計算
        expected_return = (target_price - current_price) / current_price

        # 企業名取得（簡略化）
        company_name = self._get_company_name(symbol)

        # 推奨理由生成
        reasoning = self._generate_reasoning(prediction, sentiment, risk_analysis)

        return AutoRecommendation(
            symbol=symbol,
            company_name=company_name,
            action=ActionType.BUY,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            buy_date=buy_timing,
            sell_date=sell_timing,
            expected_return=expected_return,
            confidence=prediction.confidence,
            reasoning=reasoning,
            risk_level=risk_analysis.risk_level.value,
        )

    def _get_company_name(self, symbol: str) -> str:
        """企業名取得"""
        company_map = {
            "6758.T": "ソニーグループ",
            "7203.T": "トヨタ自動車",
            "8306.T": "三菱UFJ銀行",
            "4502.T": "武田薬品工業",
            "9984.T": "ソフトバンクグループ",
            "6861.T": "キーエンス",
            "7974.T": "任天堂",
            "4689.T": "ヤフー",
            "8035.T": "東京エレクトロン",
            "6098.T": "リクルートホールディングス",
        }
        return company_map.get(symbol, symbol)

    def _generate_reasoning(self, prediction, sentiment, risk_analysis) -> str:
        """推奨理由生成"""
        reasons = []

        # 予測ベースの理由
        if prediction.confidence > 0.8:
            reasons.append(f"高信頼度予測({prediction.confidence:.1%})")

        # センチメントベースの理由
        if sentiment.sentiment_score > 0.5:
            reasons.append("強いポジティブセンチメント")
        elif sentiment.sentiment_score > 0.2:
            reasons.append("ポジティブセンチメント")

        # リスクベースの理由
        if risk_analysis.risk_level.value == "low":
            reasons.append("低リスク")
        elif risk_analysis.risk_level.value == "medium":
            reasons.append("中程度リスク")

        if not reasons:
            reasons.append("総合的判断により推奨")

        return " + ".join(reasons)

    def _display_recommendations(self, recommendations: List[AutoRecommendation]):
        """推奨結果表示"""
        if not recommendations:
            print("\n[結果] 現在推奨できる銘柄がありません")
            return

        print(f"\n[結果] フルオート投資推奨 ({len(recommendations)}銘柄)")
        print("=" * 80)

        for i, rec in enumerate(recommendations, 1):
            print(f"\n【推奨 #{i}】{rec.company_name} ({rec.symbol})")
            print(f"  買い価格: ¥{rec.entry_price:,.0f}")
            print(f"  目標価格: ¥{rec.target_price:,.0f}")
            print(f"  ストップロス: ¥{rec.stop_loss:,.0f}")
            print(f"  買い時期: {rec.buy_date.strftime('%Y年%m月%d日 %H時頃')}")
            print(f"  売り時期: {rec.sell_date.strftime('%Y年%m月%d日 %H時頃')}")
            print(f"  期待リターン: {rec.expected_return:.1%}")
            print(f"  信頼度: {rec.confidence:.1%}")
            print(f"  リスクレベル: {rec.risk_level}")
            print(f"  理由: {rec.reasoning}")

        print("\n" + "=" * 80)
        print("[注意] これらは予測に基づく推奨であり、投資は自己責任で行ってください")

    def get_system_status(self) -> Dict[str, Any]:
        """システム状況取得"""
        return {
            "tse_optimization_ready": True,
            "hybrid_predictor_ready": True,
            "sentiment_analyzer_ready": True,
            "strategy_generator_ready": True,
            "risk_manager_ready": True,
            "auto_settings": self.auto_settings,
            "last_run": datetime.now(),
        }


# メイン実行関数
async def run_full_auto():
    """フルオート実行"""
    system = FullAutoInvestmentSystem()
    recommendations = await system.run_full_auto_analysis()
    return recommendations


if __name__ == "__main__":
    # テスト実行
    asyncio.run(run_full_auto())
