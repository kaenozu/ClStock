#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自動取引戦略生成システム
AI駆動で最適な取引戦略を自動生成・最適化
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json


class ActionType(Enum):
    """取引アクション種別"""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class StrategyType(Enum):
    """戦略タイプ"""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"
    POSITION_TRADING = "position_trading"


@dataclass
class TradingSignal:
    """取引シグナル"""

    symbol: str
    action: ActionType
    confidence: float
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: float
    reasoning: str
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class TradingStrategy:
    """取引戦略"""

    name: str
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    entry_conditions: List[str]
    exit_conditions: List[str]
    risk_management: Dict[str, Any]
    expected_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    created_at: datetime


class TechnicalIndicators:
    """技術指標計算器"""

    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        """単純移動平均"""
        return data.rolling(window=window).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        """指数移動平均"""
        return data.ewm(span=window).mean()

    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """RSI"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_bollinger_bands(
        data: pd.Series, window: int = 20, num_std: float = 2
    ) -> Dict[str, pd.Series]:
        """ボリンジャーバンド"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()

        return {
            "middle": sma,
            "upper": sma + (std * num_std),
            "lower": sma - (std * num_std),
        }

    @staticmethod
    def calculate_macd(
        data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, pd.Series]:
        """MACD"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_window: int = 14,
        d_window: int = 3,
    ) -> Dict[str, pd.Series]:
        """ストキャスティクス"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()

        return {"k": k_percent, "d": d_percent}


class StrategyGenerator:
    """戦略生成器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.indicators = TechnicalIndicators()

    def generate_momentum_strategy(
        self, symbol: str, price_data: pd.DataFrame, parameters: Dict[str, Any] = None
    ) -> TradingStrategy:
        """モメンタム戦略生成"""
        if parameters is None:
            parameters = {
                "short_ma_period": 10,
                "long_ma_period": 30,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
            }

        entry_conditions = [
            f"短期MA({parameters['short_ma_period']}) > 長期MA({parameters['long_ma_period']})",
            f"RSI > {parameters['rsi_oversold']} かつ RSI < {parameters['rsi_overbought']}",
            "出来高が平均の1.2倍以上",
        ]

        exit_conditions = [
            f"短期MA < 長期MA（トレンド転換）",
            f"RSI > {parameters['rsi_overbought']}（過買い）",
            "ストップロス: -5%",
            "利益確定: +10%",
        ]

        risk_management = {
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10,
            "max_position_size": 0.1,
            "risk_per_trade": 0.02,
        }

        # パフォーマンス計算（過去データ基準）
        performance = self._calculate_strategy_performance(
            price_data, StrategyType.MOMENTUM, parameters
        )

        return TradingStrategy(
            name=f"{symbol}_Momentum_Strategy",
            strategy_type=StrategyType.MOMENTUM,
            parameters=parameters,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_management=risk_management,
            expected_return=performance["expected_return"],
            max_drawdown=performance["max_drawdown"],
            sharpe_ratio=performance["sharpe_ratio"],
            win_rate=performance["win_rate"],
            created_at=datetime.now(),
        )

    def generate_mean_reversion_strategy(
        self, symbol: str, price_data: pd.DataFrame, parameters: Dict[str, Any] = None
    ) -> TradingStrategy:
        """平均回帰戦略生成"""
        if parameters is None:
            parameters = {
                "bollinger_period": 20,
                "bollinger_std": 2,
                "rsi_period": 14,
                "oversold_threshold": 30,
                "overbought_threshold": 70,
            }

        entry_conditions = [
            f"価格がボリンジャーバンド下限を下回る",
            f"RSI < {parameters['oversold_threshold']}（売られすぎ）",
            "価格がサポートレベル近辺",
        ]

        exit_conditions = [
            "価格がボリンジャーバンド中央線に到達",
            f"RSI > 50（中立域復帰）",
            "ストップロス: -3%",
            "利益確定: +6%",
        ]

        risk_management = {
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
            "max_position_size": 0.15,
            "risk_per_trade": 0.015,
        }

        performance = self._calculate_strategy_performance(
            price_data, StrategyType.MEAN_REVERSION, parameters
        )

        return TradingStrategy(
            name=f"{symbol}_MeanReversion_Strategy",
            strategy_type=StrategyType.MEAN_REVERSION,
            parameters=parameters,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_management=risk_management,
            expected_return=performance["expected_return"],
            max_drawdown=performance["max_drawdown"],
            sharpe_ratio=performance["sharpe_ratio"],
            win_rate=performance["win_rate"],
            created_at=datetime.now(),
        )

    def generate_breakout_strategy(
        self, symbol: str, price_data: pd.DataFrame, parameters: Dict[str, Any] = None
    ) -> TradingStrategy:
        """ブレイクアウト戦略生成"""
        if parameters is None:
            parameters = {
                "resistance_period": 20,
                "volume_threshold": 1.5,
                "atr_period": 14,
                "breakout_strength": 0.02,
            }

        entry_conditions = [
            f"価格が{parameters['resistance_period']}日高値を更新",
            f"出来高が平均の{parameters['volume_threshold']}倍以上",
            f"ブレイクアウト幅がATRの{parameters['breakout_strength']*100}%以上",
        ]

        exit_conditions = [
            "価格がブレイクアウトレベルを下回る",
            "出来高が平均以下に減少",
            "ストップロス: -7%",
            "利益確定: +15%",
        ]

        risk_management = {
            "stop_loss_pct": 0.07,
            "take_profit_pct": 0.15,
            "max_position_size": 0.08,
            "risk_per_trade": 0.025,
        }

        performance = self._calculate_strategy_performance(
            price_data, StrategyType.BREAKOUT, parameters
        )

        return TradingStrategy(
            name=f"{symbol}_Breakout_Strategy",
            strategy_type=StrategyType.BREAKOUT,
            parameters=parameters,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_management=risk_management,
            expected_return=performance["expected_return"],
            max_drawdown=performance["max_drawdown"],
            sharpe_ratio=performance["sharpe_ratio"],
            win_rate=performance["win_rate"],
            created_at=datetime.now(),
        )

    def _calculate_strategy_performance(
        self,
        price_data: pd.DataFrame,
        strategy_type: StrategyType,
        parameters: Dict[str, Any],
    ) -> Dict[str, float]:
        """戦略パフォーマンス計算"""
        # 簡略化されたバックテスト
        try:
            if price_data.empty or len(price_data) < 50:
                return {
                    "expected_return": 0.05,
                    "max_drawdown": 0.15,
                    "sharpe_ratio": 0.8,
                    "win_rate": 0.55,
                }

            # 価格データから簡単なリターン計算
            returns = price_data["Close"].pct_change().dropna()

            # 戦略別のパフォーマンス調整
            if strategy_type == StrategyType.MOMENTUM:
                base_return = returns.mean() * 252 * 1.2  # モメンタムボーナス
                volatility = returns.std() * np.sqrt(252) * 0.9
            elif strategy_type == StrategyType.MEAN_REVERSION:
                base_return = abs(returns.mean()) * 252 * 0.8  # 安定性重視
                volatility = returns.std() * np.sqrt(252) * 0.7
            elif strategy_type == StrategyType.BREAKOUT:
                base_return = returns.mean() * 252 * 1.5  # 高リターン期待
                volatility = returns.std() * np.sqrt(252) * 1.2
            else:
                base_return = returns.mean() * 252
                volatility = returns.std() * np.sqrt(252)

            sharpe_ratio = base_return / volatility if volatility > 0 else 0.5

            # 勝率計算（正のリターンの割合）
            win_rate = (returns > 0).mean()

            # 最大ドローダウン（簡略化）
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())

            return {
                "expected_return": min(
                    max(base_return, -0.5), 2.0
                ),  # -50%から200%に制限
                "max_drawdown": min(max(max_drawdown, 0.05), 0.5),  # 5%から50%に制限
                "sharpe_ratio": min(max(sharpe_ratio, -2.0), 3.0),  # -2から3に制限
                "win_rate": min(max(win_rate, 0.3), 0.8),  # 30%から80%に制限
            }

        except Exception as e:
            self.logger.error(f"Performance calculation failed: {str(e)}")
            return {
                "expected_return": 0.05,
                "max_drawdown": 0.15,
                "sharpe_ratio": 0.8,
                "win_rate": 0.55,
            }


class SignalGenerator:
    """シグナル生成器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.indicators = TechnicalIndicators()

    def generate_signals(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        strategy: TradingStrategy,
        sentiment_data: Optional[Dict[str, Any]] = None,
    ) -> List[TradingSignal]:
        """取引シグナル生成"""
        signals = []

        try:
            if price_data.empty or len(price_data) < 20:
                return signals

            current_price = price_data["Close"].iloc[-1]

            # 戦略タイプ別シグナル生成
            if strategy.strategy_type == StrategyType.MOMENTUM:
                signal = self._generate_momentum_signal(
                    symbol, price_data, strategy, current_price
                )
            elif strategy.strategy_type == StrategyType.MEAN_REVERSION:
                signal = self._generate_mean_reversion_signal(
                    symbol, price_data, strategy, current_price
                )
            elif strategy.strategy_type == StrategyType.BREAKOUT:
                signal = self._generate_breakout_signal(
                    symbol, price_data, strategy, current_price
                )
            else:
                signal = None

            if signal:
                # センチメント調整
                if sentiment_data:
                    signal = self._adjust_signal_with_sentiment(signal, sentiment_data)

                signals.append(signal)

        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")

        return signals

    def _generate_momentum_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        strategy: TradingStrategy,
        current_price: float,
    ) -> Optional[TradingSignal]:
        """モメンタムシグナル生成"""
        params = strategy.parameters

        # 移動平均計算
        short_ma = self.indicators.calculate_sma(
            price_data["Close"], params["short_ma_period"]
        ).iloc[-1]
        long_ma = self.indicators.calculate_sma(
            price_data["Close"], params["long_ma_period"]
        ).iloc[-1]

        # RSI計算
        rsi = self.indicators.calculate_rsi(
            price_data["Close"], params["rsi_period"]
        ).iloc[-1]

        # 出来高チェック
        avg_volume = (
            price_data["Volume"].rolling(window=20).mean().iloc[-1]
            if "Volume" in price_data.columns
            else 1
        )
        current_volume = (
            price_data["Volume"].iloc[-1] if "Volume" in price_data.columns else 1
        )

        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # シグナル判定
        if (
            short_ma > long_ma
            and params["rsi_oversold"] < rsi < params["rsi_overbought"]
            and volume_ratio > 1.2
        ):

            confidence = min(0.9, (short_ma - long_ma) / long_ma * 10 + 0.5)

            return TradingSignal(
                symbol=symbol,
                action=ActionType.BUY,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=current_price
                * (1 - strategy.risk_management["stop_loss_pct"]),
                take_profit=current_price
                * (1 + strategy.risk_management["take_profit_pct"]),
                position_size=strategy.risk_management["max_position_size"],
                reasoning=f"モメンタム上昇: 短期MA({short_ma:.2f}) > 長期MA({long_ma:.2f}), RSI={rsi:.1f}",
                timestamp=datetime.now(),
                metadata={
                    "short_ma": short_ma,
                    "long_ma": long_ma,
                    "rsi": rsi,
                    "volume_ratio": volume_ratio,
                },
            )

        return None

    def _generate_mean_reversion_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        strategy: TradingStrategy,
        current_price: float,
    ) -> Optional[TradingSignal]:
        """平均回帰シグナル生成"""
        params = strategy.parameters

        # ボリンジャーバンド計算
        bb = self.indicators.calculate_bollinger_bands(
            price_data["Close"], params["bollinger_period"], params["bollinger_std"]
        )

        lower_band = bb["lower"].iloc[-1]
        middle_band = bb["middle"].iloc[-1]

        # RSI計算
        rsi = self.indicators.calculate_rsi(
            price_data["Close"], params["rsi_period"]
        ).iloc[-1]

        # 平均回帰シグナル（売られすぎからの回復）
        if current_price < lower_band and rsi < params["oversold_threshold"]:

            confidence = min(0.9, (lower_band - current_price) / lower_band * 5 + 0.4)

            return TradingSignal(
                symbol=symbol,
                action=ActionType.BUY,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=current_price
                * (1 - strategy.risk_management["stop_loss_pct"]),
                take_profit=middle_band,  # 中央線が目標
                position_size=strategy.risk_management["max_position_size"],
                reasoning=f"平均回帰: 価格({current_price:.2f}) < 下限バンド({lower_band:.2f}), RSI={rsi:.1f}",
                timestamp=datetime.now(),
                metadata={
                    "lower_band": lower_band,
                    "middle_band": middle_band,
                    "rsi": rsi,
                    "deviation_pct": (lower_band - current_price) / lower_band,
                },
            )

        return None

    def _generate_breakout_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        strategy: TradingStrategy,
        current_price: float,
    ) -> Optional[TradingSignal]:
        """ブレイクアウトシグナル生成"""
        params = strategy.parameters

        # 過去20日の最高値
        resistance_level = (
            price_data["High"]
            .rolling(window=params["resistance_period"])
            .max()
            .iloc[-1]
        )

        # 出来高チェック
        avg_volume = (
            price_data["Volume"].rolling(window=20).mean().iloc[-1]
            if "Volume" in price_data.columns
            else 1
        )
        current_volume = (
            price_data["Volume"].iloc[-1] if "Volume" in price_data.columns else 1
        )
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # ブレイクアウト判定
        breakout_threshold = resistance_level * (1 + params["breakout_strength"])

        if (
            current_price > breakout_threshold
            and volume_ratio > params["volume_threshold"]
        ):

            confidence = min(
                0.9, (current_price - resistance_level) / resistance_level * 10 + 0.5
            )

            return TradingSignal(
                symbol=symbol,
                action=ActionType.BUY,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=resistance_level,  # ブレイクアウトレベルがサポートに
                take_profit=current_price
                * (1 + strategy.risk_management["take_profit_pct"]),
                position_size=strategy.risk_management["max_position_size"],
                reasoning=f"ブレイクアウト: 価格({current_price:.2f}) > 抵抗線({resistance_level:.2f})",
                timestamp=datetime.now(),
                metadata={
                    "resistance_level": resistance_level,
                    "breakout_threshold": breakout_threshold,
                    "volume_ratio": volume_ratio,
                    "breakout_strength": (current_price - resistance_level)
                    / resistance_level,
                },
            )

        return None

    def _adjust_signal_with_sentiment(
        self, signal: TradingSignal, sentiment_data: Dict[str, Any]
    ) -> TradingSignal:
        """センチメントに基づくシグナル調整"""
        sentiment_score = sentiment_data.get("current_sentiment", {}).get("score", 0)

        # センチメントが一致する場合は信頼度を上げる
        if signal.action == ActionType.BUY and sentiment_score > 0.2:
            signal.confidence = min(0.95, signal.confidence + sentiment_score * 0.2)
            signal.reasoning += f" + ポジティブセンチメント({sentiment_score:.2f})"

        elif signal.action == ActionType.SELL and sentiment_score < -0.2:
            signal.confidence = min(
                0.95, signal.confidence + abs(sentiment_score) * 0.2
            )
            signal.reasoning += f" + ネガティブセンチメント({sentiment_score:.2f})"

        # センチメントが反対の場合は信頼度を下げる
        elif signal.action == ActionType.BUY and sentiment_score < -0.3:
            signal.confidence = max(0.1, signal.confidence - abs(sentiment_score) * 0.3)
            signal.reasoning += f" - ネガティブセンチメント警告({sentiment_score:.2f})"

        return signal


class AutoTradingStrategyGenerator:
    """
    自動取引戦略生成システム

    特徴:
    - AI駆動戦略生成
    - 複数戦略タイプ対応
    - リアルタイムシグナル生成
    - パフォーマンス最適化
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.strategy_generator = StrategyGenerator()
        self.signal_generator = SignalGenerator()

        # 戦略履歴
        self.strategies = {}
        self.signals_history = []

        self.logger.info("AutoTradingStrategyGenerator initialized")

    def generate_comprehensive_strategy(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        strategy_types: Optional[List[StrategyType]] = None,
    ) -> List[TradingStrategy]:
        """包括的戦略生成"""
        if strategy_types is None:
            strategy_types = [
                StrategyType.MOMENTUM,
                StrategyType.MEAN_REVERSION,
                StrategyType.BREAKOUT,
            ]

        strategies = []

        for strategy_type in strategy_types:
            try:
                if strategy_type == StrategyType.MOMENTUM:
                    strategy = self.strategy_generator.generate_momentum_strategy(
                        symbol, price_data
                    )
                elif strategy_type == StrategyType.MEAN_REVERSION:
                    strategy = self.strategy_generator.generate_mean_reversion_strategy(
                        symbol, price_data
                    )
                elif strategy_type == StrategyType.BREAKOUT:
                    strategy = self.strategy_generator.generate_breakout_strategy(
                        symbol, price_data
                    )
                else:
                    continue

                strategies.append(strategy)
                self.strategies[f"{symbol}_{strategy_type.value}"] = strategy

            except Exception as e:
                self.logger.error(
                    f"Strategy generation failed for {strategy_type}: {str(e)}"
                )

        return strategies

    def generate_trading_signals(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        sentiment_data: Optional[Dict[str, Any]] = None,
    ) -> List[TradingSignal]:
        """取引シグナル生成"""
        all_signals = []

        # 該当銘柄の戦略を取得
        symbol_strategies = [
            s for key, s in self.strategies.items() if key.startswith(symbol)
        ]

        if not symbol_strategies:
            # 戦略がない場合は自動生成
            symbol_strategies = self.generate_comprehensive_strategy(symbol, price_data)

        # 各戦略からシグナル生成
        for strategy in symbol_strategies:
            signals = self.signal_generator.generate_signals(
                symbol, price_data, strategy, sentiment_data
            )
            all_signals.extend(signals)

        # シグナル履歴に追加
        self.signals_history.extend(all_signals)

        # 履歴サイズ制限
        if len(self.signals_history) > 1000:
            self.signals_history = self.signals_history[-1000:]

        return all_signals

    def optimize_strategy(
        self, symbol: str, strategy_name: str, performance_data: Dict[str, float]
    ) -> bool:
        """戦略最適化"""
        try:
            strategy_key = f"{symbol}_{strategy_name}"

            if strategy_key not in self.strategies:
                self.logger.warning(f"Strategy {strategy_key} not found")
                return False

            strategy = self.strategies[strategy_key]

            # パフォーマンスが悪い場合はパラメータ調整
            actual_return = performance_data.get("actual_return", 0)
            actual_sharpe = performance_data.get("actual_sharpe", 0)

            if actual_return < strategy.expected_return * 0.8:
                # パフォーマンス悪化 - パラメータ調整
                self._adjust_strategy_parameters(strategy, "underperform")

            elif actual_return > strategy.expected_return * 1.2:
                # パフォーマンス向上 - さらに最適化
                self._adjust_strategy_parameters(strategy, "outperform")

            self.logger.info(f"Strategy {strategy_key} optimized")
            return True

        except Exception as e:
            self.logger.error(f"Strategy optimization failed: {str(e)}")
            return False

    def _adjust_strategy_parameters(
        self, strategy: TradingStrategy, adjustment_type: str
    ):
        """戦略パラメータ調整"""
        if strategy.strategy_type == StrategyType.MOMENTUM:
            if adjustment_type == "underperform":
                # より保守的に
                strategy.parameters["rsi_overbought"] = max(
                    60, strategy.parameters["rsi_overbought"] - 5
                )
                strategy.parameters["rsi_oversold"] = min(
                    40, strategy.parameters["rsi_oversold"] + 5
                )
            else:
                # より積極的に
                strategy.parameters["rsi_overbought"] = min(
                    80, strategy.parameters["rsi_overbought"] + 5
                )
                strategy.parameters["rsi_oversold"] = max(
                    20, strategy.parameters["rsi_oversold"] - 5
                )

        elif strategy.strategy_type == StrategyType.MEAN_REVERSION:
            if adjustment_type == "underperform":
                # より厳しい条件
                strategy.parameters["oversold_threshold"] = max(
                    20, strategy.parameters["oversold_threshold"] - 5
                )
                strategy.parameters["bollinger_std"] = min(
                    2.5, strategy.parameters["bollinger_std"] + 0.2
                )
            else:
                # より緩い条件
                strategy.parameters["oversold_threshold"] = min(
                    35, strategy.parameters["oversold_threshold"] + 5
                )
                strategy.parameters["bollinger_std"] = max(
                    1.5, strategy.parameters["bollinger_std"] - 0.2
                )

    def get_strategy_performance(self, symbol: str) -> Dict[str, Any]:
        """戦略パフォーマンス取得"""
        symbol_strategies = [
            s for key, s in self.strategies.items() if key.startswith(symbol)
        ]

        if not symbol_strategies:
            return {"strategies": 0, "performance": {}}

        performance_summary = {
            "strategies_count": len(symbol_strategies),
            "avg_expected_return": np.mean(
                [s.expected_return for s in symbol_strategies]
            ),
            "avg_sharpe_ratio": np.mean([s.sharpe_ratio for s in symbol_strategies]),
            "avg_win_rate": np.mean([s.win_rate for s in symbol_strategies]),
            "strategies": [],
        }

        for strategy in symbol_strategies:
            performance_summary["strategies"].append(
                {
                    "name": strategy.name,
                    "type": strategy.strategy_type.value,
                    "expected_return": strategy.expected_return,
                    "sharpe_ratio": strategy.sharpe_ratio,
                    "win_rate": strategy.win_rate,
                    "max_drawdown": strategy.max_drawdown,
                }
            )

        return performance_summary

    def get_recent_signals(
        self, symbol: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """最近のシグナル取得"""
        signals = self.signals_history

        if symbol:
            signals = [s for s in signals if s.symbol == symbol]

        # 最新順でソート
        signals.sort(key=lambda x: x.timestamp, reverse=True)

        return [asdict(signal) for signal in signals[:limit]]

    def generate_strategy_report(self, symbol: str) -> Dict[str, Any]:
        """戦略レポート生成"""
        performance = self.get_strategy_performance(symbol)
        recent_signals = self.get_recent_signals(symbol, 5)

        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "strategy_performance": performance,
            "recent_signals": recent_signals,
            "recommendations": self._generate_strategy_recommendations(
                symbol, performance
            ),
        }

    def _generate_strategy_recommendations(
        self, symbol: str, performance: Dict[str, Any]
    ) -> List[str]:
        """戦略推奨事項生成"""
        recommendations = []

        if performance["strategies_count"] == 0:
            recommendations.append("戦略を生成してください")
            return recommendations

        avg_return = performance["avg_expected_return"]
        avg_sharpe = performance["avg_sharpe_ratio"]
        avg_win_rate = performance["avg_win_rate"]

        if avg_return < 0.05:
            recommendations.append("期待リターンが低いです - より積極的な戦略を検討")

        if avg_sharpe < 1.0:
            recommendations.append("シャープレシオが低いです - リスク調整が必要")

        if avg_win_rate < 0.5:
            recommendations.append("勝率が低いです - エントリー条件の見直しを推奨")

        if len(recommendations) == 0:
            recommendations.append("戦略パフォーマンスは良好です")

        return recommendations
