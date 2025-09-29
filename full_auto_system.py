import argparse
import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from data.stock_data import StockDataProvider, TickerInfo
from data_retrieval_script_generator import generate_colab_data_retrieval_script


# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutoRecommendation:
    """自動推奨結果クラス"""
    def __init__(self, symbol: str, company_name: str, entry_price: float, target_price: float,
                 stop_loss: float, expected_return: float, confidence: float, risk_level: str,
                 buy_date: datetime, sell_date: datetime, reasoning: str):
        self.symbol = symbol
        self.company_name = company_name
        self.entry_price = entry_price
        self.target_price = target_price
        self.stop_loss = stop_loss
        self.expected_return = expected_return
        self.confidence = confidence
        self.risk_level = risk_level
        self.buy_date = buy_date
        self.sell_date = sell_date
        self.reasoning = reasoning


class SimplePricePredictor:
    """Deterministic price projection based on moving averages and momentum."""

    def predict(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        closes = data["Close"].astype(float)
        if closes.empty:
            return {}

        recent_close = closes.iloc[-1]
        window_short = closes.rolling(window=5, min_periods=1).mean().iloc[-1]
        window_long = closes.rolling(window=20, min_periods=1).mean().iloc[-1]

        momentum = 0.0
        if len(closes) > 10:
            momentum = (recent_close / closes.iloc[-10] - 1.0)

        trend_factor = 0.0
        if window_long:
            trend_factor = (window_short - window_long) / max(window_long, 1e-9)

        projected_return = 0.5 * momentum + 0.5 * trend_factor
        projected_return = max(min(projected_return, 0.15), -0.15)
        predicted_price = recent_close * (1.0 + projected_return)

        confidence = min(max(abs(projected_return) * 3, 0.1), 0.9)

        return {
            "predicted_price": float(predicted_price),
            "projected_return": float(projected_return),
            "confidence": float(confidence),
        }


class SimplePortfolioOptimizer:
    """Select top symbols by trailing momentum."""

    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    def optimize(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        if not price_data:
            return {"selected_stocks": []}

        scores: List[tuple[str, float]] = []
        for symbol, df in price_data.items():
            closes = df.get("Close")
            if closes is None or len(closes) < 30:
                continue

            recent = closes.iloc[-1]
            base = closes.iloc[-21]
            if base <= 0:
                continue
            momentum = (recent / base) - 1.0
            scores.append((symbol, float(momentum)))

        scores.sort(key=lambda item: item[1], reverse=True)
        selected = [symbol for symbol, _ in scores[: self.top_n]]
        return {"selected_stocks": selected, "momentum_scores": scores}


class SimpleSentimentAnalyzer:
    """Placeholder sentiment analyser returning neutral scores."""

    def analyze_sentiment(self, context: Dict[str, Any]) -> float:
        _ = context
        return 0.0


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class RiskAnalysis:
    risk_level: RiskLevel
    risk_score: float
    volatility: float


class SimpleRiskManager:
    """Approximate risk scoring based on annualised volatility."""

    def analyze_risk(
        self, data: pd.DataFrame, predictions: Dict[str, Any]
    ) -> Optional[RiskAnalysis]:
        if data is None or data.empty:
            return None

        closes = data["Close"].astype(float)
        returns = closes.pct_change().dropna()
        if returns.empty:
            return RiskAnalysis(RiskLevel.MEDIUM, 0.5, 0.0)

        volatility = float(returns.std() * (252 ** 0.5))
        risk_score = min(max(volatility / 0.4, 0.0), 1.0)

        if risk_score < 0.33:
            risk_level = RiskLevel.LOW
        elif risk_score < 0.66:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.HIGH

        return RiskAnalysis(risk_level=risk_level, risk_score=risk_score, volatility=volatility)


class SimpleStrategyGenerator:
    """Generate entry/exit levels using projected returns and risk."""

    def generate_strategy(
        self,
        predictions: Dict[str, Any],
        risk_analysis: RiskAnalysis,
        sentiment_score: float,
        current_price: float,
    ) -> Dict[str, Any]:
        predicted_price = predictions.get("predicted_price", current_price)
        projected_return = predictions.get("projected_return", 0.0)

        entry_price = float(current_price)
        target_multiplier = 1.0 + max(projected_return, 0.02) + sentiment_score * 0.1
        target_price = float(entry_price * max(target_multiplier, 0.9))

        base_stop = 0.04 + risk_analysis.risk_score * 0.08
        stop_loss = float(entry_price * (1.0 - min(base_stop, 0.2)))

        confidence = predictions.get("confidence", 0.4)
        confidence = float(max(min(confidence - risk_analysis.risk_score * 0.2, 0.9), 0.1))

        return {
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "confidence_score": confidence,
            "expected_return": target_price / entry_price - 1.0,
        }


class FullAutoInvestmentSystem:
    """完全自動投資推奨システム"""

    def __init__(self, max_symbols: Optional[int] = None):
        self.data_provider = StockDataProvider()
        self.predictor = SimplePricePredictor()
        self.optimizer = SimplePortfolioOptimizer()
        self.sentiment_analyzer = SimpleSentimentAnalyzer()
        self.strategy_generator = SimpleStrategyGenerator()
        self.risk_manager = SimpleRiskManager()
        self.failed_symbols = set()  # データ取得に失敗した銘柄を記録
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_symbols = self._resolve_max_symbols(max_symbols)

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
                "CLSTOCK_MAX_AUTO_TICKERS is not an integer ('%s'); ignoring.", env_value
            )
            return None

    async def run_full_auto_analysis(self) -> List[AutoRecommendation]:
        """完全自動分析実行"""
        try:
            print("[開始] 完全自動投資推奨分析を開始します...")
            
            # 1. 東証4000銘柄リストを取得
            print("[ステップ 1/4] (25%) - TSE4000銘柄リストを取得中...")
            print("=" * 60)
            all_tickers = self.data_provider.get_all_tickers()
            if self.max_symbols is not None and self.max_symbols < len(all_tickers):
                all_tickers = all_tickers[: self.max_symbols]
                print(
                    f"[情報] 解析対象を {len(all_tickers)} 銘柄に制限 (max={self.max_symbols})."
                )
            print(f"[情報] 取得銘柄数: {len(all_tickers)}")

            if not all_tickers:
                print("[警告] 東証4000銘柄リストが空です。処理を終了します。")
                return []

            # 2. 株価データを取得・前処理
            print("[ステップ 2/4] (50%) - 株価データを取得・前処理中...")
            print("=" * 60)
            processed_data = {}
            failed_count = 0

            for i, ticker_info in enumerate(all_tickers):
                if isinstance(ticker_info, TickerInfo):
                    symbol = ticker_info.symbol
                else:
                    symbol = str(ticker_info)
                try:
                    data = self.data_provider.get_stock_data(symbol, period="2y")
                    if data is not None and not data.empty:
                        processed_data[symbol] = data
                    else:
                        self.failed_symbols.add(symbol)  # データ取得失敗を記録
                        failed_count += 1
                        logger.warning(f"データ取得失敗: {symbol} (取得データが空またはNone)")
                    
                    # 進捗表示 (10銘柄ごと)
                    if (i + 1) % 10 == 0 or (i + 1) == len(all_tickers):
                        progress = ((i + 1) / len(all_tickers)) * 100
                        print(f"  進捗: {progress:.0f}% ({i+1}/{len(all_tickers)}) - 失敗: {failed_count}銘柄")
                        
                except Exception as e:
                    self.failed_symbols.add(symbol)  # データ取得失敗を記録
                    failed_count += 1
                    logger.error(f"データ取得中にエラーが発生しました: {symbol} - {e}")
            
            print(f"[完了] データ取得処理完了 - 成功: {len(processed_data)}銘柄, 失敗: {failed_count}銘柄")
            
            # 3. ポートフォリオ最適化
            print("[ステップ 3/4] (75%) - ポートフォリオ最適化を実行中...")
            print("=" * 60)
            try:
                optimized_portfolio = self.optimizer.optimize(processed_data)

                if not optimized_portfolio or 'selected_stocks' not in optimized_portfolio:
                    print("[警告] ポートフォリオ最適化に失敗しました。空の結果が返されました。")
                    # processed_data が空でも、self.failed_symbols に記録された銘柄のためのスクリプト生成を試みるために、
                    # _display_recommendations を呼び出す。
                    recommendations = []
                else:
                    selected_stocks = optimized_portfolio['selected_stocks']
                    print(f"[情報] 最適化完了 - 選定銘柄数: {len(selected_stocks)}")
                    
                    if not selected_stocks:
                        print("[警告] 最適化結果に選定銘柄がありません。")
                        recommendations = []
                    else:
                        # 4. 個別銘柄分析と推奨生成
                        print("[ステップ 4/4] (100%) - 個別銘柄分析と推奨生成中...")
                        print("=" * 60)
                        recommendations = []
                        analysis_failed_count = 0
                        
                        for symbol in selected_stocks:
                            try:
                                recommendation = await self._analyze_single_stock(
                                    symbol, processed_data.get(symbol)
                                )
                                if recommendation:
                                    recommendations.append(recommendation)
                                else:
                                    analysis_failed_count += 1
                                    
                            except Exception as e:
                                analysis_failed_count += 1
                                logger.error(f"個別銘柄分析中にエラーが発生しました: {symbol} - {e}")
                        
                        print(f"[完了] 個別銘柄分析完了 - 成功: {len(recommendations)}銘柄, 失敗: {analysis_failed_count}銘柄")
            
            except Exception as e:
                print(f"[エラー] ポートフォリオ最適化中に予期せぬエラーが発生しました: {e}")
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
            print(f"[致命的エラー] 完全自動分析プロセスで予期せぬエラーが発生しました: {e}")
            logger.exception("完全自動分析プロセスのエラー詳細:")
            return []

    async def _analyze_single_stock(
        self, symbol: str, data: Optional[pd.DataFrame]
    ) -> Optional[AutoRecommendation]:
        """個別銘柄分析"""
        try:
            if data is None or data.empty:
                logger.warning(f"分析対象データが無効です: {symbol}")
                return None

            # 1. 予測モデル適用
            predictions = self.predictor.predict(symbol, data)
            if not predictions or 'predicted_price' not in predictions:
                logger.warning(f"{symbol}: 予測モデル適用失敗")
                return None
            
            predicted_price = predictions['predicted_price']
            current_price = data['Close'].iloc[-1]
            
            # 2. リスク分析
            risk_analysis = self.risk_manager.analyze_risk(data, predictions)
            if not risk_analysis:
                logger.warning(f"{symbol}: リスク分析失敗")
                return None

            # 3. 感情分析 (ニュース等はダミー)
            sentiment_score = self.sentiment_analyzer.analyze_sentiment(
                {"symbol": symbol}
            )
            
            # 4. 戦略生成
            strategy = self.strategy_generator.generate_strategy(
                predictions, risk_analysis, sentiment_score, current_price
            )
            
            # 5. 推奨情報構築
            if strategy and 'entry_price' in strategy:
                entry_price = strategy['entry_price']
                target_price = strategy['target_price']
                stop_loss = strategy['stop_loss']
                
                # 期待リターン計算
                expected_return = (target_price - entry_price) / entry_price
                
                # 信頼度 (戦略スコアとリスクスコアから算出)
                strategy_confidence = strategy.get('confidence_score', 0.5)
                risk_adjusted_confidence = 1.0 - risk_analysis.risk_score
                confidence = (strategy_confidence + risk_adjusted_confidence) / 2
                
                # 理由付け (リスク分析と戦略から簡易生成)
                reasoning = self._generate_reasoning(risk_analysis, strategy)
                
                # 買い日時・売り日時 (例: 即日買い、1ヶ月後売り)
                buy_date = datetime.now()
                sell_date = buy_date + timedelta(days=30)
                
                return AutoRecommendation(
                    symbol=symbol,
                    company_name=data.attrs.get('info', {}).get('longName', symbol),
                    entry_price=entry_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    expected_return=expected_return,
                    confidence=confidence,
                    risk_level=risk_analysis.risk_level.value,
                    buy_date=buy_date,
                    sell_date=sell_date,
                    reasoning=reasoning
                )
            else:
                logger.warning(f"{symbol}: 戦略生成失敗")
                return None
                
        except Exception as e:
            logger.error(f"{symbol} 分析中にエラー発生: {e}")
            return None

    def _generate_reasoning(self, risk_analysis, strategy) -> str:
        """理由付け簡易生成"""
        reasons = []
        
        # 予測からの期待リターン
        if 'expected_return' in strategy and strategy['expected_return'] > 0.05:
            reasons.append("高期待リターン")
        elif 'expected_return' in strategy and strategy['expected_return'] > 0.02:
            reasons.append("中期待リターン")
            
        # リスクレベルの低さ
        if risk_analysis and risk_analysis.risk_level.value == "low":
            reasons.append("低リスク")
        elif risk_analysis and risk_analysis.risk_level.value == "medium":
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
        self.logger.info("Starting _generate_data_retrieval_script. failed_symbols: %s", failed_symbols)
        print(f"[INFO] _generate_data_retrieval_script called. failed_symbols: {failed_symbols}")

        if not failed_symbols:
            self.logger.info("No failed symbols detected; skipping script generation.")
            print("[INFO] No failed symbols detected. Skipping Google Colab script generation.")
            return

        script_output_dir = Path("data") / "retrieval_scripts"
        script_output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Script output directory prepared: %s", script_output_dir)
        print(f"[INFO] Script output directory: {script_output_dir}")

        for index, symbol in enumerate(failed_symbols):
            self.logger.debug("Failed symbol #%d: %s", index, symbol)

        self.logger.info("Calling generate_colab_data_retrieval_script.")
        try:
            generated_script = generate_colab_data_retrieval_script(
                missing_symbols=failed_symbols,
                period="1y",
                output_dir="."
            )
        except Exception as exc:
            self.logger.error("generate_colab_data_retrieval_script failed", exc_info=True)
            print(f"[ERROR] Failed to generate Google Colab data retrieval script: {exc}")
            return

        if not generated_script or not generated_script.strip():
            self.logger.warning("Generated script is empty.")
            print("[WARNING] Generated data retrieval script is empty. Nothing will be written.")
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
            with open(script_file_path, "w", encoding="utf-8-sig", errors="strict") as handle:
                handle.write(generated_script)
        except UnicodeEncodeError as exc:
            self.logger.error("UnicodeEncodeError while writing %s", script_file_path, exc_info=True)
            print(f"[ERROR] Unicode encoding error while writing {script_file_path}: {exc}")
            return
        except Exception as exc:
            self.logger.error("Unexpected error while writing %s", script_file_path, exc_info=True)
            print(f"[ERROR] Unexpected error while writing {script_file_path}: {exc}")
            return

        self.logger.info("Saved Google Colab data retrieval script to %s", script_file_path)
        print(f"[INFO] Saved Google Colab data retrieval script to {script_file_path}")




def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full auto investment pipeline",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Limit the number of tickers processed (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--prefer-local-data",
        action="store_true",
        help="Prioritize local CSV data before calling yfinance.",
    )
    parser.add_argument(
        "--skip-local-data",
        action="store_true",
        help="Force yfinance downloads even if local CSV data exists.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    if args.prefer_local_data and args.skip_local_data:
        parser.error("--prefer-local-data and --skip-local-data cannot be used together.")

    if args.max_tickers is not None and args.max_tickers <= 0:
        parser.error("--max-tickers must be a positive integer.")

    if args.prefer_local_data:
        os.environ["CLSTOCK_PREFER_LOCAL_DATA"] = "1"
    elif args.skip_local_data:
        os.environ["CLSTOCK_PREFER_LOCAL_DATA"] = "0"

    try:
        asyncio.run(run_full_auto(max_symbols=args.max_tickers))
    except KeyboardInterrupt:
        print("[INFO] Full auto run interrupted by user.")
        return 130

    return 0

async def run_full_auto(max_symbols: Optional[int] = None) -> List[AutoRecommendation]:
    """Convenience coroutine to execute the full auto investment analysis."""
    system = FullAutoInvestmentSystem(max_symbols=max_symbols)
    return await system.run_full_auto_analysis()


if __name__ == "__main__":
    raise SystemExit(main())
