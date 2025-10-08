#!/usr/bin/env python3
"""ウルトラ最適化投資システム - 超高利益追求モデル
成長株 + 高ボラティリティ銘柄に特化した積極運用
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import logging

from super_selective_system import BACKTEST_PERIOD, BaseInvestmentSystem

from utils.logger_config import setup_logger

logger = setup_logger(__name__)


# ウルトラ最適化システム定数
ULTRA_MIN_SCORE = 25.0
ULTRA_MIN_CONFIDENCE = 85.0
ULTRA_MAX_SYMBOLS = 15

# 高度分析パラメータ
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# 最適化重み
TECHNICAL_WEIGHT = 0.4
MOMENTUM_WEIGHT = 0.3
VOLUME_WEIGHT = 0.2
VOLATILITY_WEIGHT = 0.1


class UltraOptimizedInvestmentSystem(BaseInvestmentSystem):
    """ウルトラ最適化投資システム - 最高峰の分析と実行"""

    def __init__(self):
        super().__init__()
        self.ultra_performance_symbols = self._initialize_ultra_symbols()

        print("ウルトラ最適化投資システム初期化完了")
        print(f"ウルトラ銘柄数: {len(self.ultra_performance_symbols)}銘柄")
        print(f"最小利益閾値: {self.min_profit_threshold}%")
        print(f"最大損失閾値: {self.max_loss_threshold}%")

    def _initialize_ultra_symbols(self) -> List[str]:
        """ウルトラパフォーマンス銘柄の初期化"""
        return [
            # AIとTSE4000最適化で発見された最高収益組み合わせ
            "9984.T",  # ソフトバンクG - テック最強
            "4004.T",  # 昭和電工 - 化学セクター王者
            "4005.T",  # 住友化学 - 化学セクター準王者
            "8035.T",  # 東京エレクトロン - 半導体リーダー
            "6501.T",  # 日立製作所 - 総合電機トップ
            "8031.T",  # 三井物産 - 商社エリート
            "8058.T",  # 三菱商事 - 商社最強
            "7203.T",  # トヨタ自動車 - 自動車帝王
            "7269.T",  # スズキ - 自動車成長株
            "4519.T",  # 中外製薬 - 製薬革新者
            "1332.T",  # 日本水産 - 食品安定
            "6770.T",  # アルプスアルパイン - 電子部品
            "4324.T",  # 電通グループ - 広告界
            "1803.T",  # 清水建設 - 建設リーダー
            "5101.T",  # 横浜ゴム - 化学工業
        ]

    def identify_ultra_opportunity(self, symbol: str) -> Optional[Dict[str, Any]]:
        """ウルトラ級投資機会の特定"""
        if symbol not in self.ultra_performance_symbols:
            return None

        try:
            stock_data = self._get_stock_data(symbol, BACKTEST_PERIOD)
            if stock_data.empty:
                return None

            opportunity = self._perform_ultra_analysis(symbol, stock_data)

            if self._meets_ultra_criteria(opportunity):
                print(f"[高速] ウルトラ級機会発見: {symbol}")
                self._print_ultra_opportunity_details(opportunity)
                return opportunity

            return None

        except Exception as e:
            logging.exception(f"ウルトラ分析エラー {symbol}: {e!s}")
            return None

    def _perform_ultra_analysis(
        self,
        symbol: str,
        stock_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """ウルトラ級分析実行"""
        close = stock_data["Close"]
        volume = stock_data["Volume"]
        high = stock_data["High"]
        low = stock_data["Low"]

        # 高度技術分析
        technical_indicators = self._calculate_technical_indicators(
            close,
            high,
            low,
            volume,
        )

        # マルチタイムフレーム分析
        multi_timeframe = self._analyze_multiple_timeframes(close)

        # 高度モメンタム分析
        momentum_analysis = self._perform_momentum_analysis(close)

        # 出来高プロファイル分析
        volume_profile = self._analyze_volume_profile(close, volume)

        # ボラティリティ分析
        volatility_analysis = self._analyze_volatility(close)

        # 統合スコア計算
        ultra_score = self._calculate_ultra_score(
            technical_indicators,
            momentum_analysis,
            volume_profile,
            volatility_analysis,
        )

        return {
            "symbol": symbol,
            "current_price": close.iloc[-1],
            "technical_score": technical_indicators["composite_score"],
            "momentum_score": momentum_analysis["composite_momentum"],
            "volume_score": volume_profile["volume_strength"],
            "volatility_score": volatility_analysis["optimal_volatility"],
            "ultra_score": ultra_score,
            "confidence": min(100, ultra_score * 1.1),
            "risk_level": self._assess_risk_level(volatility_analysis),
            "entry_signal": technical_indicators["entry_signal"],
            "indicators": technical_indicators,
        }

    def _calculate_technical_indicators(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
    ) -> Dict[str, Any]:
        """技術指標の計算"""
        # 移動平均
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        ema_12 = close.ewm(span=12).mean()

        # ボリンジャーバンド
        bb_upper, bb_lower = self._calculate_bollinger_bands(close)

        # MACD
        macd_line, macd_signal, macd_histogram = self._calculate_macd(close)

        # RSI
        rsi = self._calculate_rsi(close)

        # ストキャスティクス
        stoch_k, stoch_d = self._calculate_stochastic(high, low, close)

        # ADX（トレンド強度）
        adx = self._calculate_adx(high, low, close)

        # エントリーシグナル判定
        entry_signal = self._determine_entry_signal(
            close,
            sma_20,
            sma_50,
            bb_upper,
            bb_lower,
            macd_line,
            macd_signal,
            rsi,
        )

        # 複合スコア
        composite_score = self._calculate_technical_composite_score(
            sma_20,
            sma_50,
            rsi,
            macd_line,
            macd_signal,
            stoch_k,
            adx,
        )

        return {
            "sma_20": sma_20.iloc[-1] if not sma_20.empty else 0,
            "sma_50": sma_50.iloc[-1] if not sma_50.empty else 0,
            "bb_position": self._calculate_bb_position(
                close.iloc[-1],
                bb_upper.iloc[-1],
                bb_lower.iloc[-1],
            ),
            "macd_signal_strength": self._calculate_macd_strength(
                macd_line,
                macd_signal,
            ),
            "rsi": rsi.iloc[-1] if not rsi.empty else 50,
            "stoch_momentum": (
                (stoch_k.iloc[-1] + stoch_d.iloc[-1]) / 2 if not stoch_k.empty else 50
            ),
            "trend_strength": adx.iloc[-1] if not adx.empty else 25,
            "composite_score": composite_score,
            "entry_signal": entry_signal,
        }

    def _calculate_bollinger_bands(
        self,
        close: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """ボリンジャーバンド計算"""
        sma = close.rolling(BOLLINGER_PERIOD).mean()
        std = close.rolling(BOLLINGER_PERIOD).std()
        upper = sma + (std * BOLLINGER_STD)
        lower = sma - (std * BOLLINGER_STD)
        return upper, lower

    def _calculate_macd(
        self,
        close: pd.Series,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD計算"""
        ema_fast = close.ewm(span=MACD_FAST).mean()
        ema_slow = close.ewm(span=MACD_SLOW).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=MACD_SIGNAL).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram

    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """ストキャスティクス計算"""
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        # ゼロ除算を防ぐ安全チェック
        price_range = highest_high - lowest_low
        k_percent = 100 * (
            (close - lowest_low) / price_range.where(price_range != 0, 1)
        )
        d_percent = k_percent.rolling(d_period).mean()
        return k_percent, d_percent

    def _calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """ADX（平均方向性指数）計算"""
        # 簡略化したADX計算
        tr = pd.concat(
            [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_dm = (high - high.shift(1)).where(
            (high - high.shift(1)) > (low.shift(1) - low),
            0,
        )
        minus_dm = (low.shift(1) - low).where(
            (low.shift(1) - low) > (high - high.shift(1)),
            0,
        )

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx

    def _analyze_multiple_timeframes(self, close: pd.Series) -> Dict[str, float]:
        """マルチタイムフレーム分析"""
        timeframes = {"short": 5, "medium": 20, "long": 50}

        trends = {}
        for name, period in timeframes.items():
            if len(close) >= period:
                ma = close.rolling(period).mean()
                current_trend = (close.iloc[-1] - ma.iloc[-1]) / ma.iloc[-1] * 100
                trends[f"{name}_trend"] = current_trend
            else:
                trends[f"{name}_trend"] = 0

        return trends

    def _perform_momentum_analysis(self, close: pd.Series) -> Dict[str, float]:
        """高度モメンタム分析"""
        momentum_periods = [5, 10, 20, 50]
        momentum_scores = []

        for period in momentum_periods:
            if len(close) >= period:
                momentum = (
                    (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period]
                ) * 100
                momentum_scores.append(momentum)

        if momentum_scores:
            composite_momentum = sum(momentum_scores) / len(momentum_scores)
            momentum_consistency = 1 - (
                np.std(momentum_scores) / (abs(np.mean(momentum_scores)) + 1)
            )
        else:
            composite_momentum = 0
            momentum_consistency = 0

        return {
            "composite_momentum": composite_momentum,
            "momentum_consistency": momentum_consistency * 100,
            "acceleration": self._calculate_momentum_acceleration(close),
        }

    def _calculate_momentum_acceleration(self, close: pd.Series) -> float:
        """モメンタム加速度計算"""
        if len(close) < 10:
            return 0

        recent_momentum = ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]) * 100
        past_momentum = ((close.iloc[-5] - close.iloc[-10]) / close.iloc[-10]) * 100

        return recent_momentum - past_momentum

    def _analyze_volume_profile(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> Dict[str, float]:
        """出来高プロファイル分析"""
        if len(volume) < 20:
            return {
                "volume_strength": 50,
                "volume_trend": 0,
                "price_volume_correlation": 0,
            }

        # 出来高トレンド
        volume_ma_short = volume.rolling(5).mean()
        volume_ma_long = volume.rolling(20).mean()
        volume_trend = (
            (volume_ma_short.iloc[-1] - volume_ma_long.iloc[-1])
            / volume_ma_long.iloc[-1]
        ) * 100

        # 価格と出来高の相関
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        correlation = price_change.tail(20).corr(volume_change.tail(20))

        # 出来高強度
        current_volume = volume.iloc[-1]
        avg_volume = volume.tail(20).mean()
        volume_strength = min(100, (current_volume / avg_volume) * 50)

        return {
            "volume_strength": volume_strength,
            "volume_trend": volume_trend,
            "price_volume_correlation": (
                correlation * 100 if not np.isnan(correlation) else 0
            ),
        }

    def _analyze_volatility(self, close: pd.Series) -> Dict[str, float]:
        """ボラティリティ分析"""
        if len(close) < 20:
            return {
                "current_volatility": 0,
                "volatility_trend": 0,
                "optimal_volatility": 50,
            }

        # 現在のボラティリティ
        returns = close.pct_change().dropna()
        current_vol = returns.tail(20).std() * np.sqrt(252) * 100

        # ボラティリティトレンド
        vol_short = returns.tail(10).std() * np.sqrt(252) * 100
        vol_long = (
            returns.tail(30).std() * np.sqrt(252) * 100
            if len(returns) >= 30
            else current_vol
        )
        vol_trend = vol_short - vol_long

        # 最適ボラティリティスコア（15-25%が理想）
        if 15 <= current_vol <= 25:
            optimal_score = 100
        elif 10 <= current_vol < 15 or 25 < current_vol <= 35:
            optimal_score = 75
        elif 5 <= current_vol < 10 or 35 < current_vol <= 50:
            optimal_score = 50
        else:
            optimal_score = 25

        return {
            "current_volatility": current_vol,
            "volatility_trend": vol_trend,
            "optimal_volatility": optimal_score,
        }

    def _calculate_ultra_score(
        self,
        technical: Dict,
        momentum: Dict,
        volume: Dict,
        volatility: Dict,
    ) -> float:
        """ウルトラスコア計算"""
        tech_score = technical["composite_score"]
        mom_score = momentum["composite_momentum"] + 50  # 正規化
        vol_score = volume["volume_strength"]
        volatility_score = volatility["optimal_volatility"]

        weighted_score = (
            tech_score * TECHNICAL_WEIGHT
            + mom_score * MOMENTUM_WEIGHT
            + vol_score * VOLUME_WEIGHT
            + volatility_score * VOLATILITY_WEIGHT
        )

        return min(100, max(0, weighted_score))

    def _meets_ultra_criteria(self, opportunity: Dict[str, Any]) -> bool:
        """ウルトラ基準の判定"""
        return (
            opportunity["ultra_score"] >= ULTRA_MIN_SCORE
            and opportunity["confidence"] >= ULTRA_MIN_CONFIDENCE
            and opportunity["entry_signal"]
        )

    def _assess_risk_level(self, volatility_analysis: Dict[str, float]) -> str:
        """リスクレベル評価"""
        vol = volatility_analysis["current_volatility"]
        if vol < 15:
            return "低リスク"
        if vol < 25:
            return "中リスク"
        if vol < 40:
            return "高リスク"
        return "超高リスク"

    def _print_ultra_opportunity_details(self, opportunity: Dict[str, Any]):
        """ウルトラ機会詳細の表示"""
        print(f"  銘柄: {opportunity['symbol']}")
        print(f"  現在価格: {opportunity['current_price']:.0f}円")
        print(f"  ウルトラスコア: {opportunity['ultra_score']:.1f}")
        print(f"  信頼度: {opportunity['confidence']:.1f}%")
        print(f"  リスクレベル: {opportunity['risk_level']}")
        print(f"  テクニカル: {opportunity['technical_score']:.1f}")
        print(f"  モメンタム: {opportunity['momentum_score']:.1f}")
        print(f"  出来高: {opportunity['volume_score']:.1f}")

    def calculate_ultra_position_size(self, opportunity: Dict[str, Any]) -> int:
        """ウルトラポジションサイズ計算"""
        base_amount = self._calculate_position_size(
            self.current_capital,
            opportunity["confidence"],
        )

        # リスク調整
        risk_multiplier = self._get_risk_multiplier(opportunity["risk_level"])
        adjusted_amount = base_amount * risk_multiplier

        shares = int(adjusted_amount / opportunity["current_price"])

        # 最大ポジション制限（資金の25%まで）
        max_shares = int(self.current_capital * 0.25 / opportunity["current_price"])
        return min(shares, max_shares)

    def _get_risk_multiplier(self, risk_level: str) -> float:
        """リスク乗数取得"""
        multipliers = {
            "低リスク": 1.2,
            "中リスク": 1.0,
            "高リスク": 0.7,
            "超高リスク": 0.4,
        }
        return multipliers.get(risk_level, 1.0)

    def execute_ultra_trade(self, opportunity: Dict[str, Any]) -> bool:
        """ウルトラ取引実行"""
        symbol = opportunity["symbol"]
        current_price = opportunity["current_price"]

        try:
            shares = self.calculate_ultra_position_size(opportunity)

            if shares > 0 and symbol not in self.positions:
                total_cost = shares * current_price

                if self.current_capital >= total_cost:
                    self._execute_buy(symbol, shares, current_price, datetime.now())

                    print(f"[高速] ウルトラ取引実行: {symbol}")
                    print(f"  株数: {shares:,}株")
                    print(f"  投資額: {total_cost:,.0f}円")
                    print(f"  残資金: {self.current_capital:,.0f}円")
                    print(f"  リスクレベル: {opportunity['risk_level']}")

                    return True
                print(f"[エラー] 資金不足: {symbol}")
            else:
                print(f"[警告] 取引条件未達成: {symbol}")

            return False

        except Exception as e:
            logging.exception(f"ウルトラ取引実行エラー {symbol}: {e!s}")
            return False

    def run_ultra_backtest(self) -> Dict[str, Any]:
        """ウルトラバックテスト実行"""
        print("\n" + "=" * 70)
        print("ウルトラ最適化投資システム バックテスト")
        print("=" * 70)
        print(f"初期資金: {self.initial_capital:,}円")
        print(f"対象銘柄: {len(self.ultra_performance_symbols)}銘柄")

        ultra_opportunities = 0
        successful_ultra_trades = 0
        risk_distribution = {
            "低リスク": 0,
            "中リスク": 0,
            "高リスク": 0,
            "超高リスク": 0,
        }

        # 各銘柄のウルトラ機会を分析
        for symbol in self.ultra_performance_symbols:
            print(f"\n[分析] ウルトラ分析中: {symbol}")

            opportunity = self.identify_ultra_opportunity(symbol)
            if opportunity:
                ultra_opportunities += 1
                risk_distribution[opportunity["risk_level"]] += 1

                if self.execute_ultra_trade(opportunity):
                    successful_ultra_trades += 1

        # 高度ポジション管理シミュレーション
        self._simulate_ultra_position_management()

        # 結果計算
        results = self._calculate_final_results()

        # ウルトラ結果表示
        self._display_ultra_backtest_results(
            results,
            ultra_opportunities,
            successful_ultra_trades,
            risk_distribution,
        )

        return results

    def _simulate_ultra_position_management(self):
        """ウルトラポジション管理シミュレーション"""
        print("\n[分析] ウルトラポジション管理シミュレーション")

        for symbol in list(self.positions.keys()):
            try:
                # 動的保有期間（20-60日、リスクに応じて調整）
                base_holding_days = np.random.randint(20, 61)

                stock_data = self._get_stock_data(symbol, BACKTEST_PERIOD)
                if stock_data.empty:
                    continue

                # 段階的利確戦略
                self._execute_staged_profit_taking(
                    symbol,
                    stock_data,
                    base_holding_days,
                )

            except Exception as e:
                logging.warning(f"ウルトラポジション管理エラー {symbol}: {e!s}")

    def _execute_staged_profit_taking(
        self,
        symbol: str,
        stock_data: pd.DataFrame,
        holding_days: int,
    ):
        """段階的利確実行"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        total_shares = position["shares"]
        buy_price = position["buy_price"]

        # 3段階利確戦略
        profit_targets = [0.05, 0.10, 0.15]  # 5%, 10%, 15%
        sell_ratios = [0.3, 0.4, 0.3]  # 30%, 40%, 30%

        remaining_shares = total_shares
        total_profit = 0

        for i, (target, ratio) in enumerate(zip(profit_targets, sell_ratios)):
            if len(stock_data) > holding_days + i * 10:
                current_price = stock_data["Close"].iloc[-(holding_days - i * 10)]
                profit_rate = (current_price - buy_price) / buy_price

                if profit_rate >= target and remaining_shares > 0:
                    sell_shares = int(total_shares * ratio)
                    sell_shares = min(sell_shares, remaining_shares)

                    if sell_shares > 0:
                        profit = (current_price - buy_price) * sell_shares
                        total_profit += profit
                        remaining_shares -= sell_shares

                        self.current_capital += sell_shares * current_price

                        print(
                            f"  [上昇] {symbol} 段階利確{i + 1}: {profit_rate * 100:+.1f}% "
                            f"({sell_shares:,}株, {profit:+,.0f}円)",
                        )

        # 残り株式の処理
        if remaining_shares > 0:
            self.positions[symbol]["shares"] = remaining_shares
        else:
            del self.positions[symbol]

    def _display_ultra_backtest_results(
        self,
        results: Dict[str, Any],
        opportunities: int,
        successful_trades: int,
        risk_distribution: Dict[str, int],
    ):
        """ウルトラバックテスト結果表示"""
        print("\n" + "=" * 70)
        print("ウルトラバックテスト結果")
        print("=" * 70)

        print(f"ウルトラ機会発見数: {opportunities}")
        print(f"成功取引数: {successful_trades}")
        print(f"成功率: {(successful_trades / max(opportunities, 1) * 100):.1f}%")
        print()

        print("リスク分布:")
        for risk_level, count in risk_distribution.items():
            if count > 0:
                percentage = (count / opportunities) * 100 if opportunities > 0 else 0
                print(f"  {risk_level}: {count}件 ({percentage:.1f}%)")
        print()

        print(f"初期資金: {results['initial_capital']:,}円")
        print(f"最終資金: {results['final_capital']:,}円")
        print(f"総利益: {results['total_return']:+,}円")
        print(f"収益率: {results['return_rate']:+.2f}%")
        print(f"総取引数: {results['total_trades']}")

        if results["total_trades"] > 0:
            avg_profit_per_trade = results["total_return"] / results["total_trades"]
            print(f"1取引平均利益: {avg_profit_per_trade:+,.0f}円")

        # パフォーマンス評価
        if results["return_rate"] > 20:
            performance_grade = "S級 (優秀)"
        elif results["return_rate"] > 15:
            performance_grade = "A級 (良好)"
        elif results["return_rate"] > 10:
            performance_grade = "B級 (標準)"
        elif results["return_rate"] > 5:
            performance_grade = "C級 (改善余地)"
        else:
            performance_grade = "D級 (要見直し)"

        print(f"パフォーマンス評価: {performance_grade}")
        print(f"\n実行完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 技術指標のヘルパーメソッド
    def _calculate_bb_position(self, price: float, upper: float, lower: float) -> float:
        """ボリンジャーバンド内のポジション計算"""
        if upper == lower:
            return 50
        return ((price - lower) / (upper - lower)) * 100

    def _calculate_macd_strength(
        self,
        macd_line: pd.Series,
        macd_signal: pd.Series,
    ) -> float:
        """MACDシグナル強度計算"""
        if macd_line.empty or macd_signal.empty:
            return 0

        current_diff = macd_line.iloc[-1] - macd_signal.iloc[-1]
        return min(100, max(-100, current_diff * 10))

    def _determine_entry_signal(
        self,
        close: pd.Series,
        sma_20: pd.Series,
        sma_50: pd.Series,
        bb_upper: pd.Series,
        bb_lower: pd.Series,
        macd_line: pd.Series,
        macd_signal: pd.Series,
        rsi: pd.Series,
    ) -> bool:
        """エントリーシグナル判定"""
        if any(
            s.empty
            for s in [sma_20, sma_50, bb_upper, bb_lower, macd_line, macd_signal, rsi]
        ):
            return False

        # 複数条件の総合判定
        conditions = [
            close.iloc[-1] > sma_20.iloc[-1],  # 短期トレンド上昇
            sma_20.iloc[-1] > sma_50.iloc[-1],  # 長期トレンド上昇
            macd_line.iloc[-1] > macd_signal.iloc[-1],  # MACD買いシグナル
            30 < rsi.iloc[-1] < 70,  # RSI適正範囲
            bb_lower.iloc[-1] < close.iloc[-1] < bb_upper.iloc[-1],  # BB内部
        ]

        return sum(conditions) >= 4  # 5条件中4条件以上で買いシグナル

    def _calculate_technical_composite_score(
        self,
        sma_20: pd.Series,
        sma_50: pd.Series,
        rsi: pd.Series,
        macd_line: pd.Series,
        macd_signal: pd.Series,
        stoch_k: pd.Series,
        adx: pd.Series,
    ) -> float:
        """技術指標複合スコア計算"""
        scores = []

        # トレンドスコア
        if not sma_20.empty and not sma_50.empty:
            trend_score = 70 if sma_20.iloc[-1] > sma_50.iloc[-1] else 30
            scores.append(trend_score)

        # RSIスコア
        if not rsi.empty:
            rsi_val = rsi.iloc[-1]
            if 40 <= rsi_val <= 60:
                rsi_score = 80
            elif 30 <= rsi_val <= 70:
                rsi_score = 60
            else:
                rsi_score = 40
            scores.append(rsi_score)

        # MACDスコア
        if not macd_line.empty and not macd_signal.empty:
            macd_score = 70 if macd_line.iloc[-1] > macd_signal.iloc[-1] else 30
            scores.append(macd_score)

        # ストキャスティクススコア
        if not stoch_k.empty:
            stoch_val = stoch_k.iloc[-1]
            if 20 <= stoch_val <= 80:
                stoch_score = 60
            else:
                stoch_score = 40
            scores.append(stoch_score)

        # ADXスコア
        if not adx.empty:
            adx_val = adx.iloc[-1]
            if adx_val > 25:
                adx_score = min(100, 50 + adx_val)
            else:
                adx_score = 40
            scores.append(adx_score)

        return sum(scores) / len(scores) if scores else 50


def main():
    system = UltraOptimizedInvestmentSystem()
    performance = system.run_ultra_backtest()
    print(f"\n[完了] ウルトラ最適化完了: {performance:+.1f}%の利益率達成！")


if __name__ == "__main__":
    main()
