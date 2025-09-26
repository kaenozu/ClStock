import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import yfinance as yf

from data.stock_data import StockDataProvider
from ml_models.hybrid_predictor import HybridPredictor
from optimization.tse_optimizer import TSEPortfolioOptimizer
from sentiment.sentiment_analyzer import SentimentAnalyzer
from strategies.strategy_generator import StrategyGenerator
from risk.risk_manager import RiskManager
from archive.old_systems.medium_term_prediction import MediumTermPredictionSystem
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


class FullAutoInvestmentSystem:
    """完全自動投資推奨システム"""

    def __init__(self):
        self.data_provider = StockDataProvider()
        self.predictor = HybridPredictor()
        self.optimizer = TSEPortfolioOptimizer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.strategy_generator = StrategyGenerator()
        self.risk_manager = RiskManager()
        self.medium_system = MediumTermPredictionSystem()
        self.failed_symbols = set()  # データ取得に失敗した銘柄を記録
        self.logger = logging.getLogger(self.__class__.__name__)

    async def run_full_auto_analysis(self) -> List[AutoRecommendation]:
        """完全自動分析実行"""
        try:
            print("[開始] 完全自動投資推奨分析を開始します...")
            
            # 1. 東証4000銘柄リストを取得
            print("[ステップ 1/4] (25%) - TSE4000銘柄リストを取得中...")
            print("=" * 60)
            all_tickers = self.data_provider.get_all_tickers()
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
                symbol = ticker_info.symbol
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
                    print(f"[情報] 最適化完了 - 選択銘柄数: {len(selected_stocks)}")
                    
                    if not selected_stocks:
                        print("[警告] 最適化結果に選択銘柄がありません。")
                        recommendations = []
                    else:
                        # 4. 個別銘柄分析と推奨生成
                        print("[ステップ 4/4] (100%) - 個別銘柄分析と推奨生成中...")
                        print("=" * 60)
                        recommendations = []
                        analysis_failed_count = 0
                        
                        for symbol in selected_stocks:
                            try:
                                recommendation = await self._analyze_single_stock(symbol, processed_data.get(symbol))
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

    async def _analyze_single_stock(self, symbol: str, data) -> Optional[AutoRecommendation]:
        """個別銘柄分析"""
        try:
            if data is None or data.empty:
                logger.warning(f"分析対象データが無効です: {symbol}")
                return None
            
            # 1. 予測モデル適用
            predictions = self.predictor.predict(data)
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
            sentiment_score = self.sentiment_analyzer.analyze_sentiment({"symbol": symbol})
            
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
                risk_adjusted_confidence = 1.0 - risk_analysis.risk_score  # リスクが低いほど信頼度高
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
        if risk_analysis.risk_level.value == "low":
            reasons.append("低リスク")
        elif risk_analysis.risk_level.value == "medium":
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
                print(f"  買い価格: ¥{rec.entry_price:,.0f}")
                print(f"  目標価格: ¥{rec.target_price:,.0f}")
                print(f"  ストップロス: ¥{rec.stop_loss:,.0f}")
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
        """データ取得に失敗した銘柄用のGoogle Colabスクリプトを生成"""
        # self.failed_symbols は、yfinanceでもCSVでもデータを取得できなかった銘柄のみを記録している
        self.logger.info(f"_generate_data_retrieval_scriptが呼び出されました。self.failed_symbolsの内容: {self.failed_symbols}")
        print(f"[デバッグ] _generate_data_retrieval_scriptが呼び出されました。self.failed_symbolsの内容: {self.failed_symbols}")
        # self.failed_symbols が空の場合もログに出力
        if not self.failed_symbols:
            self.logger.info("_generate_data_retrieval_script: self.failed_symbols が空です。")
            print("[情報] _generate_data_retrieval_script: self.failed_symbols が空です。")
            return
        if self.failed_symbols:
            self.logger.info(f"yfinance+CSVで最終的にデータ取得に失敗した銘柄数: {len(self.failed_symbols)}")
            self.logger.info(f"yfinance+CSVで最終的に失敗: {list(self.failed_symbols)}")
            print(f"[情報] yfinance+CSVで最終的にデータ取得に失敗した銘柄数: {len(self.failed_symbols)}")
            print(f"[情報] yfinance+CSVで最終的に失敗: {list(self.failed_symbols)}")

            try:
                # from data_retrieval_script_generator import generate_colab_data_retrieval_script

                # スクリプト出力用のディレクトリを定義
                script_output_dir = Path(\"data\") / \"retrieval_scripts\"
                script_output_dir.mkdir(parents=True, exist_ok=True)  # ディレクトリがなければ作成
                self.logger.info(f\"スクリプト出力ディレクトリを作成しました: {script_output_dir}\")

                # データ取得に失敗した銘柄のみを対象にする
                # self.failed_symbols の内容を詳細にログに出力
                self.logger.info(f"self.failed_symbols の内容 (repr): {repr(list(self.failed_symbols))}")
                self.logger.info(f"self.failed_symbols の内容 (str): {list(self.failed_symbols)}")
                for i, symbol in enumerate(self.failed_symbols):
                    self.logger.info(f"  - 銘柄 #{i}: {repr(symbol)} (str: {symbol})")
                self.logger.info("generate_colab_data_retrieval_scriptを呼び出します。")
                try:
                    generated_script = generate_colab_data_retrieval_script(
                        missing_symbols=list(self.failed_symbols), # リストのまま渡す
                        period="1y", # 必要に応じて変更
                        output_dir="." # Colab内での出力先（デフォルト） 
                    )
                    self.logger.info("generate_colab_data_retrieval_scriptの呼び出しが完了しました。")
                    self.logger.info(f"生成されたスクリプトの長さ: {len(generated_script)}")
                    # 生成されたスクリプトの先頭と末尾の一部をログに出力（デバッグ用）
                    if len(generated_script) > 200:
                        self.logger.info(f"生成されたスクリプトの先頭200文字: {generated_script[:200]}")
                        self.logger.info(f"生成されたスクリプトの末尾200文字: {generated_script[-200:]}")
                    else:
                        self.logger.info(f"生成されたスクリプトの内容: {generated_script}")
                    # 生成されたスクリプトが空でないことを確認
                    if not generated_script.strip():
                        self.logger.warning("生成されたスクリプトが空です。")
                        print("[警告] 生成されたスクリプトが空です。")
                        return
                except Exception as e:
                    self.logger.error(f"generate_colab_data_retrieval_scriptの呼び出しでエラーが発生しました: {e}")
                    self.logger.error(f"missing_symbolsの内容: {self.failed_symbols}")
                    print(f"[エラー] generate_colab_data_retrieval_scriptの呼び出しでエラーが発生しました: {e}")
                    print(f"[エラー] missing_symbolsの内容: {self.failed_symbols}")
                    import traceback
                    self.logger.error(f"generate_colab_data_retrieval_scriptの呼び出しでエラーが発生しました:\n{traceback.format_exc()}")
                    return

                # 生成されたスクリプトをファイルに保存 (エンコーディングエラー対策)
                script_file_path = script_output_dir / "colab_data_fetcher.py"
                self.logger.info(f"スクリプトファイルパス: {script_file_path}")
                try:
                    # Windows環境でのエンコーディング対応を強化
                    with open(script_file_path, "w", encoding="utf-8-sig", errors="strict") as f:
                        f.write(generated_script)
                    self.logger.info(f"Google Colab用データ取得スクリプトを生成しました: {script_file_path}")
                    print(f"[情報] Google Colab用データ取得スクリプトを生成しました: {script_file_path}")
                except UnicodeEncodeError as e:
                    self.logger.error(f"ファイル書き込み時にUnicodeエンコーディングエラーが発生しました: {e}")
                    self.logger.error(f"エラーが発生したファイルパス: {script_file_path}")
                    self.logger.error(f"生成されたスクリプトの内容 (先頭100文字): {generated_script[:100]}")
                    print(f"[エラー] ファイル書き込み時にエンコーディングエラーが発生しました: {e}")
                    print(f"[エラー] エラーが発生したファイルパス: {script_file_path}")
                    print(f"[エラー] 生成されたスクリプトの内容 (先頭100文字): {generated_script[:100]}")
                except Exception as e:
                    self.logger.error(f"ファイル書き込み時に予期せぬエラーが発生しました: {e}")
                    print(f"[エラー] ファイル書き込み時に予期せぬエラーが発生しました: {e}")

            except Exception as e:
                self.logger.error(f"Google Colab用データ取得スクリプトの生成中にエラーが発生しました: {e}")
                print(f"[エラー] Google Colab用データ取得スクリプトの生成中にエラーが発生しました: {e}")

                # スクリプト出力用のディレクトリを定義
                script_output_dir = Path(\"data\") / \"retrieval_scripts\"
                script_output_dir.mkdir(parents=True, exist_ok=True)  # ディレクトリがなければ作成
                self.logger.info(f\"スクリプト出力ディレクトリを作成しました: {script_output_dir}\")

                # データ取得に失敗した銘柄のみを対象にする
                # self.failed_symbols の内容を詳細にログに出力
                self.logger.info(f\"self.failed_symbols の内容 (repr): {repr(list(self.failed_symbols))}\")
                self.logger.info(f\"self.failed_symbols の内容 (str): {list(self.failed_symbols)}\")
                for i, symbol in enumerate(self.failed_symbols):
                    self.logger.info(f\"  - 銘柄 #{i}: {repr(symbol)} (str: {symbol})\")
                self.logger.info(\"generate_colab_data_retrieval_scriptを呼び出します。\")
                try:
                    generated_script = generate_colab_data_retrieval_script(
                        missing_symbols=list(self.failed_symbols), # リストのまま渡す
                        period="1y", # 必要に応じて変更
                        output_dir="." # Colab内での出力先（デフォルト） 
                    )
                    self.logger.info(\"generate_colab_data_retrieval_scriptの呼び出しが完了しました。\")
                    self.logger.info(f\"生成されたスクリプトの長さ: {len(generated_script)}\")
                    # 生成されたスクリプトの先頭と末尾の一部をログに出力（デバッグ用）
                    if len(generated_script) > 200:
                        self.logger.info(f\"生成されたスクリプトの先頭200文字: {generated_script[:200]}\")
                        self.logger.info(f\"生成されたスクリプトの末尾200文字: {generated_script[-200:]}\")
                    else:
                        self.logger.info(f\"生成されたスクリプトの内容: {generated_script}\")
                    # 生成されたスクリプトが空でないことを確認
                    if not generated_script.strip():
                        self.logger.warning(\"生成されたスクリプトが空です。\")
                        print(\"[警告] 生成されたスクリプトが空です。\")
                        return
                except Exception as e:
                    self.logger.error(f\"generate_colab_data_retrieval_scriptの呼び出しでエラーが発生しました: {e}\")
                    self.logger.error(f\"missing_symbolsの内容: {self.failed_symbols}\")
                    print(f\"[エラー] generate_colab_data_retrieval_scriptの呼び出しでエラーが発生しました: {e}\")
                    print(f\"[エラー] missing_symbolsの内容: {self.failed_symbols}\")
                    import traceback
                    self.logger.error(f\"generate_colab_data_retrieval_scriptの呼び出しでエラーが発生しました:\\n{traceback.format_exc()}\")
                    return

                # 生成されたスクリプトをファイルに保存 (エンコーディングエラー対策)
                script_file_path = script_output_dir / \"colab_data_fetcher.py\"
                self.logger.info(f\"スクリプトファイルパス: {script_file_path}\")
                try:
                    # Windows環境でのエンコーディング対応を強化
                    with open(script_file_path, \"w\", encoding=\"utf-8-sig\", errors=\"strict\") as f:
                        f.write(generated_script)
                    self.logger.info(f\"Google Colab用データ取得スクリプトを生成しました: {script_file_path}\")
                    print(f\"[情報] Google Colab用データ取得スクリプトを生成しました: {script_file_path}\")
                except UnicodeEncodeError as e:
                    self.logger.error(f\"ファイル書き込み時にUnicodeエンコーディングエラーが発生しました: {e}\")
                    self.logger.error(f\"エラーが発生したファイルパス: {script_file_path}\")
                    self.logger.error(f\"生成されたスクリプトの内容 (先頭100文字): {generated_script[:100]}\")
                    print(f\"[エラー] ファイル書き込み時にエンコーディングエラーが発生しました: {e}\")
                    print(f\"[エラー] エラーが発生したファイルパス: {script_file_path}\")
                    print(f\"[エラー] 生成されたスクリプトの内容 (先頭100文字): {generated_script[:100]}\")
                except Exception as e:
                    self.logger.error(f\"ファイル書き込み時に予期せぬエラーが発生しました: {e}\")
                    print(f\"[エラー] ファイル書き込み時に予期せぬエラーが発生しました: {e}\")

            except Exception as e:
                self.logger.error(f\"Google Colab用データ取得スクリプトの生成中にエラーが発生しました: {e}\")
                print(f\"[エラー] Google Colab用データ取得スクリプトの生成中にエラーが発生しました: {e}\")

    def get_system_status(self) -> Dict[str, Any]:
        """システムステータス取得"""
        return {
            "tse_optimization_ready": True,
            "hybrid_predictor_ready": True,
            "sentiment_analyzer_ready": True,
            "strategy_generator_ready": True,
            "risk_manager_ready": True,
            "auto_settings": {
                "data_source_priority": ["yfinance", "csv_cache"],
                "prediction_horizon_days": 30,
                "risk_tolerance": "medium"
            },
            "last_run": datetime.now(),
        }


# メイン処理定義
async def run_full_auto():
    """完全自動実行"""
    system = FullAutoInvestmentSystem()
    recommendations = await system.run_full_auto_analysis()
    return recommendations


if __name__ == "__main__":
    # 実行処理
    asyncio.run(run_full_auto())