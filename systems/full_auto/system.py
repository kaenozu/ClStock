"""完全自動投資システムの中核実装。"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from archive.old_systems.medium_term_prediction import MediumTermPredictionSystem
from config.settings import get_settings
from data.stock_data import StockDataProvider
from data_retrieval_script_generator import generate_colab_data_retrieval_script
from trading.tse import PortfolioBacktester
from trading.tse.analysis import StockProfile
from trading.tse.optimizer import PortfolioOptimizer

from .adapters import (
    AutoRecommendation,
    HybridPredictorAdapter,
    RiskAssessment,
    RiskManagerAdapter,
    SentimentAnalyzerAdapter,
    StrategyGeneratorAdapter,
)

__all__ = ["FullAutoInvestmentSystem", "run_full_auto"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FullAutoInvestmentSystem:
    """完全自動投資推奨システム"""

    def __init__(
        self,
        max_symbols: Optional[int] = None,
        *,
        settings: Optional[Any] = None,
        data_provider: Optional[StockDataProvider] = None,
        predictor: Optional[HybridPredictorAdapter] = None,
        optimizer: Optional[PortfolioOptimizer] = None,
        sentiment_analyzer: Optional[SentimentAnalyzerAdapter] = None,
        strategy_generator: Optional[StrategyGeneratorAdapter] = None,
        risk_manager: Optional[RiskManagerAdapter] = None,
        medium_system: Optional[MediumTermPredictionSystem] = None,
        backtester: Optional[PortfolioBacktester] = None,
        script_generator: Optional[Callable[..., str]] = None,
    ) -> None:
        self.data_provider = data_provider or StockDataProvider()
        self.predictor = predictor or HybridPredictorAdapter()
        self.optimizer = optimizer or PortfolioOptimizer()
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzerAdapter()
        self.strategy_generator = strategy_generator or StrategyGeneratorAdapter()
        self.risk_manager = risk_manager or RiskManagerAdapter()
        self.medium_system = medium_system or MediumTermPredictionSystem()
        self.backtester = backtester or PortfolioBacktester(self.data_provider)
        self.script_generator = script_generator or generate_colab_data_retrieval_script
        self.failed_symbols: set[str] = set()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_symbols = self._resolve_max_symbols(max_symbols)
        self.settings = settings or get_settings()

    def _resolve_max_symbols(self, max_symbols: Optional[int]) -> Optional[int]:
        if max_symbols is not None:
            if max_symbols <= 0:
                raise ValueError("max_symbols must be a positive integer.")
            return max_symbols

        env_value = os.getenv("CLSTOCK_MAX_AUTO_TICKERS")
        if not env_value:
            return None

        try:
            parsed = int(env_value)
            if parsed <= 0:
                self.logger.warning(
                    "CLSTOCK_MAX_AUTO_TICKERS must be positive; ignoring value '%s'.",
                    env_value,
                )
                return None
            return parsed
        except ValueError:
            self.logger.warning(
                "CLSTOCK_MAX_AUTO_TICKERS is not an integer ('%s'); ignoring.",
                env_value,
            )
            return None

    def _build_stock_profiles(
        self,
        processed_data: Dict[str, pd.DataFrame],
    ) -> List[StockProfile]:
        profiles: List[StockProfile] = []

        for symbol, data in processed_data.items():
            if data is None or data.empty:
                continue

            try:
                close = data["Close"] if "Close" in data else data.iloc[:, 0]
                close = close.dropna()
                if close.empty:
                    continue

                volume_series = (
                    data.get("Volume") if isinstance(data, pd.DataFrame) else None
                )
                if volume_series is not None:
                    volume_series = volume_series.dropna()

                returns = close.pct_change().dropna()
                volatility = (
                    float(returns.std() * (252**0.5)) if not returns.empty else 0.0
                )

                start_price = float(close.iloc[0])
                end_price = float(close.iloc[-1])
                profit_potential = (
                    ((end_price - start_price) / start_price)
                    if start_price not in (0, 0.0)
                    else 0.0
                )

                if volume_series is not None and not volume_series.empty:
                    market_cap = end_price * float(volume_series.iloc[-1])
                else:
                    market_cap = end_price

                diversity_score = 1.0 / (1.0 + max(volatility, 0.0))
                combined_score = profit_potential + diversity_score

                sector = data.attrs.get("info", {}).get("sector", "unknown")

                profiles.append(
                    StockProfile(
                        symbol=symbol,
                        sector=sector,
                        market_cap=float(market_cap),
                        volatility=float(volatility),
                        profit_potential=float(profit_potential),
                        diversity_score=float(diversity_score),
                        combined_score=float(combined_score),
                    ),
                )
            except Exception:
                self.logger.debug(
                    "Failed to build stock profile for %s",
                    symbol,
                    exc_info=True,
                )
                continue

        return profiles

    def _optimize_portfolio(
        self,
        processed_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, List[StockProfile]]:
        profiles = self._build_stock_profiles(processed_data)

        if not profiles:
            return {"selected_stocks": [], "selected_profiles": []}

        try:
            candidate_sizes = list(getattr(self, "portfolio_sizes", []))
        except Exception:
            candidate_sizes = []

        if not candidate_sizes:
            try:
                from trading.tse.optimizer import PORTFOLIO_SIZES as DEFAULT_SIZES
            except Exception:
                DEFAULT_SIZES = []

            candidate_sizes = [
                size for size in DEFAULT_SIZES if isinstance(size, int) and size > 0
            ]

        max_available = len(profiles)
        if max_available > 0 and max_available not in candidate_sizes:
            candidate_sizes.append(max_available)

        unique_sizes = sorted(
            {size for size in candidate_sizes if isinstance(size, int) and size > 0},
        )

        if not unique_sizes:
            unique_sizes = [max_available]

        backtester = self.backtester
        optimization_results: Dict[int, Dict[str, Any]] = {}

        for size in unique_sizes:
            try:
                if hasattr(self.optimizer, "optimize_portfolio"):
                    selected_profiles = self.optimizer.optimize_portfolio(
                        profiles,
                        target_size=size,
                    )
                elif hasattr(self.optimizer, "optimize"):
                    selected_profiles = self.optimizer.optimize(
                        profiles,
                        target_size=size,
                    )  # type: ignore[attr-defined]
                else:
                    raise AttributeError(
                        "Optimizer does not support portfolio optimisation methods",
                    )

                if not isinstance(selected_profiles, list):
                    selected_profiles = []

                backtest_result: Optional[Dict[str, Any]] = None
                selected_symbols = [
                    profile.symbol
                    for profile in selected_profiles
                    if hasattr(profile, "symbol")
                ]

                if selected_symbols:
                    try:
                        backtest_result = backtester.backtest_portfolio(
                            selected_symbols,
                        )
                    except Exception:
                        logger.exception("バックテストに失敗しました (size=%s)", size)

                optimization_results[size] = {
                    "selected_profiles": selected_profiles,
                    "backtest": backtest_result,
                }
            except Exception:
                logger.exception(
                    "ポートフォリオ最適化処理中にエラーが発生しました (size=%s)",
                    size,
                )

        # 最適な結果を選択（最も高いリターンを持つ結果）
        best_candidate = None
        best_return = -float("inf")
        fallback_candidate = None

        for size, result in optimization_results.items():
            if not isinstance(result, dict):
                continue

            selected_profiles = result.get("selected_profiles") or []
            if not selected_profiles:
                continue

            if fallback_candidate is None:
                fallback_candidate = (size, selected_profiles, result.get("backtest"))

            backtest_data = result.get("backtest") or {}
            return_rate = None
            if isinstance(backtest_data, dict):
                try:
                    raw_rate = backtest_data.get("return_rate")
                    if raw_rate is not None:
                        return_rate = float(raw_rate)
                except (TypeError, ValueError):
                    return_rate = None

            if return_rate is not None and return_rate > best_return:
                best_return = return_rate
                best_candidate = (size, selected_profiles, backtest_data)

        final_candidate = best_candidate or fallback_candidate

        if final_candidate is None:
            # 何らかの理由で最適な結果がない場合は最初のプロファイルを使用
            first_result = next(
                iter(optimization_results.values()),
                {"selected_profiles": [], "backtest": {}},
            )
            selected_profiles = first_result.get("selected_profiles", [])
        else:
            best_size, selected_profiles, backtest_data = final_candidate

        selected_stocks = [
            profile.symbol
            for profile in selected_profiles
            if hasattr(profile, "symbol")
        ]
        return {
            "selected_stocks": selected_stocks,
            "selected_profiles": selected_profiles,
        }

    async def run_full_auto_analysis(self) -> List[AutoRecommendation]:
        """完全自動分析実行"""
        try:
            print("[開始] 完全自動投資推奨分析を開始します...")

            # 1. 東証4000銘柄リストを取得
            print("[ステップ 1/4] (25%) - TSE4000銘柄リストを取得中...")
            print("=" * 60)
            target_stocks = self.settings.target_stocks
            all_symbols = list(target_stocks.keys())
            if self.max_symbols is not None and self.max_symbols < len(all_symbols):
                all_symbols = all_symbols[: self.max_symbols]
                print(
                    f"[情報] 解析対象を {len(all_symbols)} 銘柄に制限 (max={self.max_symbols}).",
                )
            print(f"[情報] 取得銘柄数: {len(all_symbols)}")

            if not all_symbols:
                print("[警告] 東証4000銘柄リストが空です。処理を終了します。")
                return []

            # 2. 株価データを取得・前処理
            print("[ステップ 2/4] (50%) - 株価データを取得・前処理中...")
            print("=" * 60)
            processed_data = {}
            failed_count = 0
            total_symbols = len(all_symbols)

            for i, symbol in enumerate(all_symbols):
                company_name = target_stocks.get(symbol, symbol)
                try:
                    data = self.data_provider.get_stock_data(
                        symbol,
                        period="2y",
                    )
                    if data is not None and not data.empty:
                        data.attrs.setdefault("info", {})["longName"] = company_name
                        processed_data[symbol] = data
                    else:
                        self.failed_symbols.add(symbol)  # データ取得失敗を記録
                        failed_count += 1
                        logger.warning(
                            f"データ取得失敗: {symbol} (取得データが空またはNone)",
                        )

                    # 進捗表示 (10銘柄ごと)
                    if (i + 1) % 10 == 0 or (i + 1) == total_symbols:
                        progress = ((i + 1) / total_symbols) * 100
                        print(
                            f"  進捗: {progress:.0f}% ({i + 1}/{total_symbols}) - 失敗: {failed_count}銘柄",
                        )

                except Exception as e:
                    self.failed_symbols.add(symbol)  # データ取得失敗を記録
                    failed_count += 1
                    logger.error(f"データ取得中にエラーが発生しました: {symbol} - {e}")

            print(
                f"[完了] データ取得処理完了 - 成功: {len(processed_data)}銘柄, 失敗: {failed_count}銘柄",
            )

            # 3. ポートフォリオ最適化
            print("[ステップ 3/4] (75%) - ポートフォリオ最適化を実行中...")
            print("=" * 60)
            try:
                optimized_portfolio = self._optimize_portfolio(processed_data)

                if not optimized_portfolio:
                    print(
                        "[警告] ポートフォリオ最適化に失敗しました。空の結果が返されました。",
                    )
                    recommendations = []
                else:
                    # selected_profiles を optimized_portfolio から取得
                    selected_profiles = optimized_portfolio.get("selected_profiles", [])
                    # selected_stocks を optimized_portfolio から取得
                    selected_stocks = optimized_portfolio.get("selected_stocks", [])

                    if not selected_profiles:  # selected_profiles が空の場合もチェック
                        print(
                            "[警告] 最適化結果から選定されたプロファイルがありません。",
                        )
                        recommendations = []
                    else:
                        # 実際にデータがある銘柄のみを対象とする
                        available_selected_stocks = [
                            s for s in selected_stocks if s in processed_data
                        ]

                        print(
                            f"[情報] 最適化完了 - 選定銘柄数: {len(selected_profiles)} (内 {len(available_selected_stocks)} がデータ取得済み)",
                        )

                        if not available_selected_stocks:
                            print(
                                "[警告] 最適化結果にデータが存在する銘柄がありません。",
                            )
                            recommendations = []
                        else:
                            print("[ステップ 4/4] (100%) - 個別銘柄分析と推奨生成中...")
                            print("=" * 60)
                            recommendations = []
                            analysis_failed_count = 0

                            for symbol in available_selected_stocks:
                                try:
                                    recommendation = await self._analyze_single_stock(
                                        symbol,
                                        processed_data.get(symbol),
                                    )
                                    if recommendation:
                                        recommendations.append(recommendation)
                                    else:
                                        analysis_failed_count += 1

                                except Exception as e:
                                    analysis_failed_count += 1
                                    logger.error(
                                        f"個別銘柄分析中にエラーが発生しました: {symbol} - {e}",
                                    )

                            print(
                                f"[完了] 個別銘柄分析完了 - 成功: {len(recommendations)}銘柄, 失敗: {analysis_failed_count}銘柄",
                            )

            except Exception as e:
                print(
                    f"[エラー] ポートフォリオ最適化中に予期せぬエラーが発生しました: {e}",
                )
                logger.exception("ポートフォリオ最適化エラーの詳細:")
                recommendations = []

            # 結果表示
            # processed_data が空でも、self.failed_symbols に記録された銘柄のためのスクリプト生成を試みるために、
            # _display_recommendations を呼び出す。
            # recommendations が空でも _display_recommendations は処理を実行し、
            # self._generate_data_retrieval_script() を呼び出す。
            self._display_recommendations(recommendations)
            return recommendations

        except Exception as e:
            print(
                f"[致命的エラー] 完全自動分析プロセスで予期せぬエラーが発生しました: {e}",
            )
            logger.exception("完全自動分析プロセスのエラー詳細:")
            return []

    async def _analyze_single_stock(
        self,
        symbol: str,
        data: Optional[pd.DataFrame],
    ) -> Optional[AutoRecommendation]:
        """個別銘柄分析"""
        try:
            if data is None or data.empty:
                logger.warning(f"分析対象データが無効です: {symbol}")
                return None

            # 1. 予測モデル適用
            predictions = self.predictor.predict(symbol, data)
            if not predictions:
                logger.warning(f"{symbol}: 予測モデル適用失敗")
                return None

            predicted_price = predictions["predicted_price"]
            current_price = data["Close"].iloc[-1]

            # 2. リスク分析
            risk_analysis = self.risk_manager.analyze_risk(symbol, data, predictions)
            if not risk_analysis:
                logger.warning(f"{symbol}: リスク分析失敗")
                # リスク分析が失敗した場合も処理を継続するか、None を返すかを検討
                # ここでは、リスク分析がなくても戦略生成を試みる
                risk_analysis_for_payload = None
            else:
                risk_analysis_for_payload = risk_analysis

            # 3. 感情分析 (ニュース等はダミー)
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(symbol)

            # 4. 戦略生成
            strategy = self.strategy_generator.generate_strategy(
                symbol,
                data,
                predictions,
                risk_analysis,
                sentiment_result,
            )

            # 5. 推奨情報構築
            if strategy:  # trading_strategy -> strategy
                entry_price = strategy.get(
                    "entry_price",
                    float(current_price),
                )  # 'entry_price' は strategy から取得する
                # stop_loss_pct などは、strategy の中にあるか、デフォルト値を使う
                stop_loss_pct = strategy.get("stop_loss_pct", 0.05)
                take_profit_pct = strategy.get(
                    "take_profit_pct",
                    strategy.get("expected_return", 0.0),
                )

                # stop_loss を計算
                stop_loss = entry_price * (1 - stop_loss_pct)
                # expected_return_pct を計算
                expected_return_pct = strategy.get("expected_return", take_profit_pct)
                target_price = entry_price * (1 + expected_return_pct)

                strategy_payload = {
                    "entry_price": entry_price,
                    "target_price": target_price,
                    "stop_loss": stop_loss,
                    "expected_return": expected_return_pct,
                    "confidence_score": strategy.get(
                        "confidence_score",
                        0.0,
                    ),  # strategy から confidence_score を取得
                    "sentiment_score": sentiment_result.get(
                        "sentiment_score",
                        0.0,
                    ),  # sentiment_result から取得
                    "predicted_price": predicted_price,
                    "take_profit_pct": take_profit_pct,
                    "stop_loss_pct": stop_loss_pct,
                }

                # 期待リターン計算
                expected_return = expected_return_pct

                # 信頼度 (戦略スコア、リスクスコア、予測信頼度から算出)
                strategy_confidence = max(
                    min(strategy.get("confidence_score", 0.0), 1.0),
                    0.0,  # strategy から confidence_score を取得
                )
                if risk_analysis_for_payload:
                    risk_score = getattr(risk_analysis_for_payload, "risk_score", None)
                    if risk_score is None:
                        raw_risk = getattr(risk_analysis_for_payload, "raw", None)
                        if raw_risk is not None:
                            risk_score = getattr(raw_risk, "total_risk_score", None)
                    if risk_score is None:
                        risk_score = getattr(
                            risk_analysis_for_payload,
                            "total_risk_score",
                            0.5,
                        )
                else:
                    risk_score = 0.5
                try:
                    risk_score = float(risk_score)
                except (TypeError, ValueError):
                    risk_score = 0.5
                risk_adjusted_confidence = max(min(1.0 - risk_score, 1.0), 0.0)
                # predictions から confidence を取得
                prediction_confidence = predictions.get("confidence", 0.0)
                confidence = (
                    strategy_confidence
                    + risk_adjusted_confidence
                    + prediction_confidence
                ) / 3

                # 理由付け (リスク分析と戦略から簡易生成)
                reasoning = self._generate_reasoning(risk_analysis, strategy_payload)

                # 買い日時・売り日時 (例: 即日買い、1ヶ月後売り)
                buy_date = datetime.now()
                sell_date = buy_date + timedelta(days=30)

                # risk_analysis_for_payload が None の場合や risk_level が存在しない場合に備えて安全にアクセス
                risk_level_value = "unknown"
                if risk_analysis_for_payload is not None:
                    risk_level_attr = getattr(
                        risk_analysis_for_payload,
                        "risk_level",
                        None,
                    )
                    if risk_level_attr is not None:
                        risk_level_value = getattr(risk_level_attr, "value", "unknown")

                return AutoRecommendation(
                    symbol=symbol,
                    company_name=data.attrs.get("info", {}).get("longName", symbol),
                    entry_price=entry_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    expected_return=expected_return,
                    confidence=confidence,
                    risk_level=risk_level_value,
                    buy_date=buy_date,
                    sell_date=sell_date,
                    reasoning=reasoning,
                )
            logger.warning(f"{symbol}: 戦略生成失敗")
            return None

        except Exception as e:
            logger.error(f"{symbol} 分析中にエラー発生: {e}")
            return None

    def _perform_portfolio_risk_analysis(
        self,
        symbol: str,
        current_price: float,
        price_data: pd.DataFrame,
        predicted_price: float,
    ):
        try:
            portfolio_data = {
                "portfolio_value": current_price,
                "cash": 0.0,
                "positions": {symbol: current_price},
                "target_allocation": {symbol: 1.0},
                "expected_prices": {symbol: predicted_price},
                "metadata": {
                    "analysis_type": "single_stock",
                    "generated_at": datetime.now(),
                },
            }
            price_map = {symbol: price_data}
            return self.risk_manager.analyze_portfolio_risk(portfolio_data, price_map)
        except Exception as exc:
            logger.error(f"{symbol}: ポートフォリオリスク分析の準備に失敗: {exc}")
            return None

    def _generate_reasoning(self, risk_analysis, strategy) -> str:
        """理由付け簡易生成"""
        reasons = []

        # 予測からの期待リターン
        if "expected_return" in strategy and strategy["expected_return"] > 0.05:
            reasons.append("高期待リターン")
        elif "expected_return" in strategy and strategy["expected_return"] > 0.02:
            reasons.append("中期待リターン")

        # リスクレベルの低さ
        if (
            risk_analysis
            and hasattr(risk_analysis, "risk_level")
            and risk_analysis.risk_level
            and hasattr(risk_analysis.risk_level, "value")
            and risk_analysis.risk_level.value == "low"
        ):
            reasons.append("低リスク")
        elif (
            risk_analysis
            and hasattr(risk_analysis, "risk_level")
            and risk_analysis.risk_level
            and hasattr(risk_analysis.risk_level, "value")
            and risk_analysis.risk_level.value == "medium"
        ):
            reasons.append("中程度リスク")

        if not reasons:
            reasons.append("独自分析に基づく推奨")

        return " + ".join(reasons)

    def _display_recommendations(self, recommendations: List[AutoRecommendation]):
        """推奨結果表示"""
        if not recommendations:
            print("\n[情報] 現在の推奨銘柄がありません")
            # recommendations が空でも、self.failed_symbols に記録された銘柄のためのスクリプト生成を試みる
            # self._generate_data_retrieval_script() を呼び出す。
        else:
            # print(f"stdout encoding: {sys.stdout.encoding}, errors: {sys.stdout.errors}")
            print(f"\n[結果] 完全自動投資推奨 ({len(recommendations)}銘柄)")
            print("=" * 80)

            for i, rec in enumerate(recommendations, 1):
                print(f"\n--- 推奨 #{i} --- {rec.company_name} ({rec.symbol})")
                print(f"  買い価格: JPY {rec.entry_price:,.0f}")
                print(f"  目標価格: JPY {rec.target_price:,.0f}")
                print(f"  ストップロス: JPY {rec.stop_loss:,.0f}")
                print(f"  買い日時: {rec.buy_date.strftime('%Y年%m月%d日 %H時%M分')}")
                print(f"  売り日時: {rec.sell_date.strftime('%Y年%m月%d日 %H時%M分')}")
                print(f"  期待リターン: {rec.expected_return:.1%}")
                print(f"  信頼度: {rec.confidence:.1%}")
                print(f"  リスクレベル: {rec.risk_level}")
                print(f"  理由: {rec.reasoning}")

            print("\n" + "=" * 80)
            print("[注意] 上記は参考情報です。投資判断はご自身で行ってください。")

        # データ取得に失敗した銘柄がある場合、Google Colab用スクリプトを生成
        self._generate_data_retrieval_script()

    def _generate_data_retrieval_script(self):
        """Generate a Google Colab helper script for symbols that failed to download."""
        failed_symbols = list(self.failed_symbols or [])
        self.logger.info(
            "Starting _generate_data_retrieval_script. failed_symbols: %s",
            failed_symbols,
        )
        print(
            f"[INFO] _generate_data_retrieval_script called. failed_symbols: {failed_symbols}",
        )

        if not failed_symbols:
            self.logger.info("No failed symbols detected; skipping script generation.")
            print(
                "[INFO] No failed symbols detected. Skipping Google Colab script generation.",
            )
            return

        script_output_dir = Path("data") / "retrieval_scripts"
        script_output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Script output directory prepared: %s", script_output_dir)
        print(f"[INFO] Script output directory: {script_output_dir}")

        for index, symbol in enumerate(failed_symbols):
            self.logger.debug("Failed symbol #%d: %s", index, symbol)

        self.logger.info("Calling data retrieval script generator.")
        try:
            generated_script = self.script_generator(
                missing_symbols=failed_symbols,
                period="1y",
                output_dir=".",
            )
        except Exception as exc:
            self.logger.error(
                "Data retrieval script generation failed",
                exc_info=True,
            )
            print(
                f"[ERROR] Failed to generate Google Colab data retrieval script: {exc}",
            )
            return

        if not generated_script or not generated_script.strip():
            self.logger.warning("Generated script is empty.")
            print(
                "[WARNING] Generated data retrieval script is empty. Nothing will be written.",
            )
            return

        script_length = len(generated_script)
        self.logger.info("Generated script length: %d characters", script_length)
        if script_length <= 200:
            self.logger.debug("Generated script contents: %s", generated_script)
        else:
            self.logger.debug("Generated script head: %s", generated_script[:200])
            self.logger.debug("Generated script tail: %s", generated_script[-200:])

        script_file_path = script_output_dir / "colab_data_fetcher.py"
        try:
            with open(
                script_file_path,
                "w",
                encoding="utf-8-sig",
                errors="strict",
            ) as handle:
                handle.write(generated_script)
        except UnicodeEncodeError as exc:
            self.logger.error(
                "UnicodeEncodeError while writing %s",
                script_file_path,
                exc_info=True,
            )
            print(
                f"[ERROR] Unicode encoding error while writing {script_file_path}: {exc}",
            )
            return
        except Exception as exc:
            self.logger.error(
                "Unexpected error while writing %s",
                script_file_path,
                exc_info=True,
            )
            print(f"[ERROR] Unexpected error while writing {script_file_path}: {exc}")
            return

        self.logger.info(
            "Saved Google Colab data retrieval script to %s",
            script_file_path,
        )
        print(f"[INFO] Saved Google Colab data retrieval script to {script_file_path}")


async def run_full_auto(max_symbols: Optional[int] = None) -> List[AutoRecommendation]:
    """Convenience coroutine to execute the full auto investment analysis."""
    system = FullAutoInvestmentSystem(max_symbols=max_symbols)
    return await system.run_full_auto_analysis()
