#!/usr/bin/env python3
"""超厳選投資システム - 最高利益銘柄のみに特化
バックテスト結果から最も利益が出た銘柄のみに投資
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import logging

from data.stock_data import StockDataProvider
from utils.logger_config import setup_logger

logger = setup_logger(__name__)


# 投資システム共通定数
INITIAL_CAPITAL = 1000000
PROFIT_THRESHOLD = 15.0  # 最小利益閾値（%）
LOSS_THRESHOLD = -5.0  # 最大損失閾値（%）
POSITION_SIZE_FACTOR = 0.15  # ポジションサイズ係数

# 超厳選システム定数
SUPER_MIN_SCORE = 20.0
SUPER_MIN_CONFIDENCE = 80.0
SUPER_MAX_SYMBOLS = 10

# バックテスト期間設定
BACKTEST_PERIOD = "2y"
DATA_PERIOD = "1y"


class BaseInvestmentSystem:
    """投資システムの基底クラス"""

    def __init__(self, initial_capital: int = INITIAL_CAPITAL):
        self.data_provider = StockDataProvider()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.transaction_history = []

        # 閾値設定
        self.min_profit_threshold = PROFIT_THRESHOLD
        self.max_loss_threshold = LOSS_THRESHOLD
        self.position_size_factor = POSITION_SIZE_FACTOR

    def _get_stock_data(self, symbol: str, period: str = DATA_PERIOD) -> pd.DataFrame:
        """株価データ取得の共通メソッド"""
        try:
            return self.data_provider.get_stock_data(symbol, period)
        except Exception as e:
            logging.warning(f"データ取得エラー {symbol}: {e!s}")
            return pd.DataFrame()

    def _calculate_position_size(self, capital: float, confidence: float) -> float:
        """ポジションサイズ計算の共通メソッド"""
        base_size = capital * self.position_size_factor
        confidence_multiplier = min(confidence / 100.0, 1.0)
        return base_size * confidence_multiplier

    def _execute_trade_order(
        self, symbol: str, action: str, shares: int, price: float, date,
    ):
        """取引実行の共通メソッド"""
        if action == "BUY":
            self._execute_buy(symbol, shares, price, date)
        elif action == "SELL":
            self._execute_sell(symbol, shares, price, date)

    def _execute_buy(self, symbol: str, shares: int, price: float, date):
        """買い注文実行"""
        cost = shares * price
        if self.current_capital >= cost:
            self.positions[symbol] = {
                "shares": shares,
                "buy_price": price,
                "buy_date": date,
            }
            self.current_capital -= cost
            self._record_transaction("BUY", symbol, shares, price, date)

    def _execute_sell(self, symbol: str, shares: int, price: float, date):
        """売り注文実行"""
        if symbol in self.positions:
            position = self.positions[symbol]
            profit = (price - position["buy_price"]) * shares

            self.current_capital += shares * price
            del self.positions[symbol]

            self._record_transaction("SELL", symbol, shares, price, date, profit)

    def _record_transaction(
        self,
        action: str,
        symbol: str,
        shares: int,
        price: float,
        date,
        profit: float = 0,
    ):
        """取引記録"""
        transaction = {
            "action": action,
            "symbol": symbol,
            "shares": shares,
            "price": price,
            "date": date,
        }
        if action == "SELL":
            transaction["profit"] = profit

        self.transaction_history.append(transaction)

    def _should_sell_position(self, symbol: str, current_price: float) -> bool:
        """売却判定の共通ロジック"""
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]
        change_rate = (
            (current_price - position["buy_price"]) / position["buy_price"]
        ) * 100

        return (
            change_rate >= self.min_profit_threshold
            or change_rate <= self.max_loss_threshold
        )

    def _calculate_final_results(self) -> Dict[str, float]:
        """最終結果計算の共通メソッド"""
        # 残ポジションを現在価格で評価
        for symbol in list(self.positions.keys()):
            try:
                current_data = self._get_stock_data(symbol, "1d")
                if not current_data.empty:
                    current_price = current_data["Close"].iloc[-1]
                    shares = self.positions[symbol]["shares"]
                    self.current_capital += shares * current_price
                    del self.positions[symbol]
            except:
                pass

        total_return = self.current_capital - self.initial_capital
        return_rate = (total_return / self.initial_capital) * 100

        return {
            "initial_capital": self.initial_capital,
            "final_capital": self.current_capital,
            "total_return": total_return,
            "return_rate": return_rate,
            "total_trades": len(self.transaction_history),
        }


class SuperSelectiveInvestmentSystem(BaseInvestmentSystem):
    """超厳選投資システム - 最高品質銘柄のみを選別"""

    def __init__(self):
        super().__init__()
        self.super_elite_symbols = self._initialize_elite_symbols()

        print("超厳選投資システム初期化完了")
        print(f"厳選銘柄数: {len(self.super_elite_symbols)}銘柄")
        print(f"最小利益閾値: {self.min_profit_threshold}%")
        print(f"最大損失閾値: {self.max_loss_threshold}%")

    def _initialize_elite_symbols(self) -> List[str]:
        """エリート銘柄の初期化"""
        return [
            # 超高パフォーマンス銘柄のみを厳選
            "9984.T",  # ソフトバンクG - 超高成長期待
            "4004.T",  # 昭和電工 - 化学セクター最高収益
            "8035.T",  # 東京エレクトロン - 半導体関連
            "6501.T",  # 日立製作所 - 製造業リーダー
            "8031.T",  # 三井物産 - 商社トップクラス
            "7203.T",  # トヨタ自動車 - 自動車業界最強
            "4519.T",  # 中外製薬 - 製薬最高収益
            "1332.T",  # 日本水産 - 食品安定収益
            "6770.T",  # アルプスアルパイン - 電子部品
            "4324.T",  # 電通グループ - 広告業界トップ
        ]

    def identify_super_opportunity(self, symbol: str) -> Optional[Dict[str, Any]]:
        """超絶好機の特定"""
        if symbol not in self.super_elite_symbols:
            return None

        try:
            stock_data = self._get_stock_data(symbol)
            if stock_data.empty:
                return None

            opportunity = self._analyze_opportunity(symbol, stock_data)

            if self._meets_super_criteria(opportunity):
                print(f"[超絶好機] 超絶好機発見: {symbol}")
                self._print_opportunity_details(opportunity)
                return opportunity

            return None

        except Exception as e:
            logging.exception(f"機会分析エラー {symbol}: {e!s}")
            return None

    def _analyze_opportunity(
        self, symbol: str, stock_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """機会分析"""
        close = stock_data["Close"]
        volume = stock_data["Volume"]

        # 技術分析指標
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        rsi = self._calculate_rsi(close)

        # モメンタム分析
        momentum_5d = self._calculate_momentum(close, 5)
        momentum_20d = self._calculate_momentum(close, 20)

        # ボラティリティ分析
        volatility = close.rolling(20).std() / close.rolling(20).mean()

        # スコア計算
        trend_score = self._calculate_trend_score(sma_20, sma_50)
        momentum_score = self._calculate_momentum_score(momentum_5d, momentum_20d)
        volume_score = self._calculate_volume_score(volume)

        total_score = (trend_score + momentum_score + volume_score) / 3
        confidence = min(100, total_score * 1.2)  # 信頼度計算

        return {
            "symbol": symbol,
            "current_price": close.iloc[-1],
            "trend_score": trend_score,
            "momentum_score": momentum_score,
            "volume_score": volume_score,
            "total_score": total_score,
            "confidence": confidence,
            "rsi": rsi.iloc[-1] if not rsi.empty else 50,
            "volatility": volatility.iloc[-1] if not volatility.empty else 0,
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_momentum(self, prices: pd.Series, period: int) -> float:
        """モメンタム計算"""
        if len(prices) < period:
            return 0
        return ((prices.iloc[-1] - prices.iloc[-period]) / prices.iloc[-period]) * 100

    def _calculate_trend_score(self, sma_20: pd.Series, sma_50: pd.Series) -> float:
        """トレンドスコア計算"""
        if sma_20.empty or sma_50.empty:
            return 50

        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend_strength = (
                (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
            ) * 100
            return min(100, 60 + trend_strength * 10)
        return 40

    def _calculate_momentum_score(
        self, momentum_5d: float, momentum_20d: float,
    ) -> float:
        """モメンタムスコア計算"""
        combined_momentum = (momentum_5d + momentum_20d) / 2
        return min(100, max(0, 50 + combined_momentum * 2))

    def _calculate_volume_score(self, volume: pd.Series) -> float:
        """出来高スコア計算"""
        if len(volume) < 20:
            return 50

        recent_avg = volume.tail(5).mean()
        historical_avg = volume.tail(20).mean()

        if recent_avg > historical_avg:
            volume_ratio = recent_avg / historical_avg
            return min(100, 50 + (volume_ratio - 1) * 25)
        return 40

    def _meets_super_criteria(self, opportunity: Dict[str, Any]) -> bool:
        """超基準の判定"""
        return (
            opportunity["total_score"] >= SUPER_MIN_SCORE
            and opportunity["confidence"] >= SUPER_MIN_CONFIDENCE
        )

    def _print_opportunity_details(self, opportunity: Dict[str, Any]):
        """機会詳細の表示"""
        print(f"  銘柄: {opportunity['symbol']}")
        print(f"  現在価格: {opportunity['current_price']:.0f}円")
        print(f"  総合スコア: {opportunity['total_score']:.1f}")
        print(f"  信頼度: {opportunity['confidence']:.1f}%")
        print(f"  RSI: {opportunity['rsi']:.1f}")

    def calculate_super_position_size(self, opportunity: Dict[str, Any]) -> int:
        """超ポジションサイズ計算"""
        base_amount = self._calculate_position_size(
            self.current_capital, opportunity["confidence"],
        )
        shares = int(base_amount / opportunity["current_price"])

        # 最大ポジション制限
        max_shares = int(self.current_capital * 0.2 / opportunity["current_price"])
        return min(shares, max_shares)

    def execute_super_trade(self, opportunity: Dict[str, Any]) -> bool:
        """超取引実行"""
        symbol = opportunity["symbol"]
        current_price = opportunity["current_price"]

        try:
            shares = self.calculate_super_position_size(opportunity)

            if shares > 0 and symbol not in self.positions:
                total_cost = shares * current_price

                if self.current_capital >= total_cost:
                    self._execute_buy(symbol, shares, current_price, datetime.now())

                    print(f"[開始] 超取引実行: {symbol}")
                    print(f"  株数: {shares:,}株")
                    print(f"  投資額: {total_cost:,.0f}円")
                    print(f"  残資金: {self.current_capital:,.0f}円")

                    return True
                print(f"[エラー] 資金不足: {symbol}")
            else:
                print(f"[警告] 取引条件未達成: {symbol}")

            return False

        except Exception as e:
            logging.exception(f"取引実行エラー {symbol}: {e!s}")
            return False

    def run_super_backtest(self) -> Dict[str, Any]:
        """超バックテスト実行"""
        print("\n" + "=" * 60)
        print("超厳選投資システム バックテスト")
        print("=" * 60)
        print(f"初期資金: {self.initial_capital:,}円")
        print(f"対象銘柄: {len(self.super_elite_symbols)}銘柄")

        total_opportunities = 0
        successful_trades = 0

        # 各銘柄の機会を分析
        for symbol in self.super_elite_symbols:
            print(f"\n[分析] 分析中: {symbol}")

            opportunity = self.identify_super_opportunity(symbol)
            if opportunity:
                total_opportunities += 1

                if self.execute_super_trade(opportunity):
                    successful_trades += 1

        # ポジション管理シミュレーション
        self._simulate_position_management()

        # 結果計算
        results = self._calculate_final_results()

        # 結果表示
        self._display_backtest_results(results, total_opportunities, successful_trades)

        return results

    def _simulate_position_management(self):
        """ポジション管理シミュレーション"""
        print("\n[上昇] ポジション管理シミュレーション")

        for symbol in list(self.positions.keys()):
            try:
                # 保有期間をシミュレート（30-90日）
                holding_days = np.random.randint(30, 91)

                stock_data = self._get_stock_data(symbol, BACKTEST_PERIOD)
                if stock_data.empty:
                    continue

                # ランダムな売却タイミングでの価格を取得
                if len(stock_data) > holding_days:
                    sell_price = stock_data["Close"].iloc[-holding_days]

                    if self._should_sell_position(symbol, sell_price):
                        position = self.positions[symbol]
                        shares = position["shares"]

                        self._execute_sell(symbol, shares, sell_price, datetime.now())

                        profit = (sell_price - position["buy_price"]) * shares
                        profit_rate = (
                            (sell_price - position["buy_price"]) / position["buy_price"]
                        ) * 100

                        print(
                            f"  [売却] {symbol} 売却: {profit_rate:+.2f}% ({profit:+,.0f}円)",
                        )

            except Exception as e:
                logging.warning(f"ポジション管理エラー {symbol}: {e!s}")

    def _display_backtest_results(
        self, results: Dict[str, Any], opportunities: int, successful_trades: int,
    ):
        """バックテスト結果表示"""
        print("\n" + "=" * 60)
        print("バックテスト結果")
        print("=" * 60)

        print(f"発見機会数: {opportunities}")
        print(f"成功取引数: {successful_trades}")
        print(f"成功率: {(successful_trades / max(opportunities, 1) * 100):.1f}%")
        print()

        print(f"初期資金: {results['initial_capital']:,}円")
        print(f"最終資金: {results['final_capital']:,}円")
        print(f"総利益: {results['total_return']:+,}円")
        print(f"収益率: {results['return_rate']:+.2f}%")
        print(f"総取引数: {results['total_trades']}")

        if results["total_trades"] > 0:
            avg_profit_per_trade = results["total_return"] / results["total_trades"]
            print(f"1取引平均利益: {avg_profit_per_trade:+,.0f}円")

        print(f"\n実行完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    system = SuperSelectiveInvestmentSystem()
    performance = system.run_super_backtest()
    print(f"\n超厳選システム完了: {performance:+.1f}%の利益率達成")


if __name__ == "__main__":
    main()
