"""
87%精度システムと統合された取引戦略定義

Precision87BreakthroughSystemとの完全連携により、
高精度な取引シグナルを生成し、実際の利益・損失を計算する
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

# 既存システムインポート
from models_new.precision.precision_87_system import Precision87BreakthroughSystem
from data.stock_data import StockDataProvider


class SignalType(Enum):
    """取引シグナルタイプ"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


@dataclass
class TradingSignal:
    """取引シグナル"""

    symbol: str
    signal_type: SignalType
    confidence: float
    predicted_price: float
    current_price: float
    expected_return: float
    position_size: float
    timestamp: datetime
    reasoning: str
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    precision_87_achieved: bool = False


@dataclass
class MarketConditions:
    """市場環境状況"""

    volatility: float
    trend_strength: float
    market_sentiment: float
    volume_profile: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH"


class TradingStrategy:
    """
    87%精度システムと統合された取引戦略

    実際の株価予測に基づく高精度取引システム
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        max_position_size: float = 0.1,
        precision_threshold: float = 85.0,
        confidence_threshold: float = 0.7,
    ):
        """
        Args:
            initial_capital: 初期資本
            max_position_size: 最大ポジションサイズ（資本に対する割合）
            precision_threshold: 精度閾値（この値以上で取引実行）
            confidence_threshold: 信頼度閾値
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.precision_threshold = precision_threshold
        self.confidence_threshold = confidence_threshold

        # 87%精度システム初期化
        self.precision_system = Precision87BreakthroughSystem()
        self.data_provider = StockDataProvider()

        # 取引パラメータ
        self.stop_loss_pct = 0.05  # 5% ストップロス
        self.take_profit_pct = 0.15  # 15% 利確
        self.min_expected_return = 0.03  # 最小期待リターン 3%

        # 取引コスト設定
        self.commission_rate = 0.001  # 0.1% 手数料
        self.spread_rate = 0.0005  # 0.05% スプレッド
        self.slippage_rate = 0.0002  # 0.02% スリッページ

        self.logger = logging.getLogger(__name__)

    def generate_trading_signal(
        self, symbol: str, current_capital: float
    ) -> Optional[TradingSignal]:
        """
        87%精度システムによる取引シグナル生成

        Args:
            symbol: 銘柄コード
            current_capital: 現在の資本

        Returns:
            取引シグナル or None
        """
        try:
            # 87%精度システムで予測実行
            prediction_result = self.precision_system.predict_with_87_precision(symbol)

            if "error" in prediction_result:
                self.logger.warning(
                    f"予測エラー {symbol}: {prediction_result['error']}"
                )
                return None

            # 基本データ取得
            predicted_price = prediction_result["final_prediction"]
            current_price = prediction_result.get("current_price", 100.0)
            confidence = prediction_result["final_confidence"]
            accuracy = prediction_result["final_accuracy"]
            change_rate = prediction_result.get("predicted_change_rate", 0.0)

            # 精度・信頼度チェック
            if accuracy < self.precision_threshold:
                self.logger.info(
                    f"精度不足 {symbol}: {accuracy:.1f}% < {self.precision_threshold}%"
                )
                return None

            if confidence < self.confidence_threshold:
                self.logger.info(
                    f"信頼度不足 {symbol}: {confidence:.2f} < {self.confidence_threshold}"
                )
                return None

            # 期待リターン計算
            expected_return = change_rate
            if abs(expected_return) < self.min_expected_return:
                self.logger.info(f"期待リターン不足 {symbol}: {expected_return:.3f}")
                return None

            # 市場環境分析
            market_conditions = self._analyze_market_conditions(symbol)

            # シグナルタイプ決定
            signal_type = self._determine_signal_type(
                expected_return, confidence, accuracy, market_conditions
            )

            if signal_type == SignalType.HOLD:
                return None

            # ポジションサイズ計算
            position_size = self._calculate_position_size(
                current_capital,
                expected_return,
                confidence,
                accuracy,
                market_conditions,
            )

            if position_size <= 0:
                return None

            # ストップロス・利確価格計算
            stop_loss_price, take_profit_price = self._calculate_exit_prices(
                current_price, predicted_price, signal_type, confidence
            )

            # 推論理由生成
            reasoning = self._generate_reasoning(
                symbol, accuracy, confidence, expected_return, market_conditions
            )

            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                predicted_price=predicted_price,
                current_price=current_price,
                expected_return=expected_return,
                position_size=position_size,
                timestamp=datetime.now(),
                reasoning=reasoning,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                precision_87_achieved=prediction_result.get(
                    "precision_87_achieved", False
                ),
            )

            self.logger.info(
                f"取引シグナル生成: {symbol} {signal_type.value} "
                f"精度:{accuracy:.1f}% 信頼度:{confidence:.2f} "
                f"期待リターン:{expected_return:.3f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"シグナル生成エラー {symbol}: {e}")
            return None

    def _analyze_market_conditions(self, symbol: str) -> MarketConditions:
        """市場環境分析"""
        try:
            # 履歴データ取得
            historical_data = self.data_provider.get_stock_data(symbol, period="3mo")

            if len(historical_data) < 20:
                # デフォルト値を返す
                return MarketConditions(
                    volatility=0.5,
                    trend_strength=0.5,
                    market_sentiment=0.5,
                    volume_profile=0.5,
                    risk_level="MEDIUM",
                )

            # ボラティリティ計算（過去20日の標準偏差）
            returns = historical_data["Close"].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)

            # トレンド強度（過去20日の価格変化）
            price_change = (
                historical_data["Close"].iloc[-1] - historical_data["Close"].iloc[-20]
            ) / historical_data["Close"].iloc[-20]
            trend_strength = abs(price_change)

            # 市場センチメント（RSI近似）
            close_prices = historical_data["Close"]
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            market_sentiment = rsi.iloc[-1] / 100.0

            # ボリュームプロファイル
            volume_avg = historical_data["Volume"].rolling(20).mean().iloc[-1]
            current_volume = historical_data["Volume"].iloc[-1]
            volume_profile = min(current_volume / volume_avg, 2.0) / 2.0

            # リスクレベル決定
            if volatility > 0.4 or trend_strength > 0.1:
                risk_level = "HIGH"
            elif volatility < 0.2 and trend_strength < 0.05:
                risk_level = "LOW"
            else:
                risk_level = "MEDIUM"

            return MarketConditions(
                volatility=float(np.clip(volatility, 0, 1)),
                trend_strength=float(np.clip(trend_strength, 0, 1)),
                market_sentiment=float(np.clip(market_sentiment, 0, 1)),
                volume_profile=float(np.clip(volume_profile, 0, 1)),
                risk_level=risk_level,
            )

        except Exception as e:
            self.logger.warning(f"市場環境分析エラー {symbol}: {e}")
            return MarketConditions(
                volatility=0.5,
                trend_strength=0.5,
                market_sentiment=0.5,
                volume_profile=0.5,
                risk_level="MEDIUM",
            )

    def _determine_signal_type(
        self,
        expected_return: float,
        confidence: float,
        accuracy: float,
        market_conditions: MarketConditions,
    ) -> SignalType:
        """シグナルタイプ決定"""

        # 87%達成時は積極的
        if accuracy >= 87.0:
            if expected_return > 0.02:  # 2%以上の期待リターン
                return SignalType.BUY
            elif expected_return < -0.02:
                return SignalType.SELL

        # 高精度・高信頼度時
        elif accuracy >= 85.0 and confidence >= 0.8:
            if expected_return > 0.03:  # 3%以上の期待リターン
                return SignalType.BUY
            elif expected_return < -0.03:
                return SignalType.SELL

        # 中精度時は慎重に
        elif accuracy >= self.precision_threshold:
            if market_conditions.risk_level != "HIGH":
                if expected_return > 0.05:  # 5%以上の期待リターン
                    return SignalType.BUY
                elif expected_return < -0.05:
                    return SignalType.SELL

        return SignalType.HOLD

    def _calculate_position_size(
        self,
        current_capital: float,
        expected_return: float,
        confidence: float,
        accuracy: float,
        market_conditions: MarketConditions,
    ) -> float:
        """ポジションサイズ計算（Kelly基準＋リスク調整）"""

        # 基本ポジションサイズ（資本の割合）
        base_size = self.max_position_size

        # 精度による調整（87%達成時は積極的）
        if accuracy >= 87.0:
            accuracy_multiplier = 1.2
        elif accuracy >= 85.0:
            accuracy_multiplier = 1.0
        else:
            accuracy_multiplier = 0.8

        # 信頼度による調整
        confidence_multiplier = confidence

        # 期待リターンによる調整（Kelly基準近似）
        return_multiplier = min(abs(expected_return) / 0.1, 1.0)

        # 市場環境による調整
        if market_conditions.risk_level == "HIGH":
            risk_multiplier = 0.5
        elif market_conditions.risk_level == "LOW":
            risk_multiplier = 1.0
        else:
            risk_multiplier = 0.8

        # 最終ポジションサイズ
        position_ratio = (
            base_size
            * accuracy_multiplier
            * confidence_multiplier
            * return_multiplier
            * risk_multiplier
        )

        # 最大10%、最小1%に制限
        position_ratio = np.clip(position_ratio, 0.01, 0.1)

        return current_capital * position_ratio

    def _calculate_exit_prices(
        self,
        current_price: float,
        predicted_price: float,
        signal_type: SignalType,
        confidence: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """ストップロス・利確価格計算"""

        if signal_type == SignalType.BUY:
            # 買いポジションの場合
            stop_loss_price = current_price * (1 - self.stop_loss_pct)
            take_profit_price = current_price * (1 + self.take_profit_pct)

            # 予測価格が利確価格より高い場合は予測価格を利確目標に
            if predicted_price > take_profit_price:
                take_profit_price = predicted_price

        elif signal_type == SignalType.SELL:
            # 売りポジションの場合
            stop_loss_price = current_price * (1 + self.stop_loss_pct)
            take_profit_price = current_price * (1 - self.take_profit_pct)

            # 予測価格が利確価格より低い場合は予測価格を利確目標に
            if predicted_price < take_profit_price:
                take_profit_price = predicted_price
        else:
            return None, None

        return stop_loss_price, take_profit_price

    def _generate_reasoning(
        self,
        symbol: str,
        accuracy: float,
        confidence: float,
        expected_return: float,
        market_conditions: MarketConditions,
    ) -> str:
        """取引理由生成"""

        reasons = []

        # 精度による理由
        if accuracy >= 87.0:
            reasons.append(f"87%精度達成 ({accuracy:.1f}%)")
        else:
            reasons.append(f"高精度予測 ({accuracy:.1f}%)")

        # 信頼度による理由
        reasons.append(f"信頼度 {confidence:.2f}")

        # 期待リターンによる理由
        if expected_return > 0:
            reasons.append(f"上昇予測 +{expected_return:.1f}%")
        else:
            reasons.append(f"下落予測 {expected_return:.1f}%")

        # 市場環境による理由
        reasons.append(f"市場リスク: {market_conditions.risk_level}")
        if market_conditions.volatility > 0.6:
            reasons.append("高ボラティリティ")
        if market_conditions.trend_strength > 0.7:
            reasons.append("強いトレンド")

        return " | ".join(reasons)

    def calculate_trading_costs(
        self, position_value: float, signal_type: SignalType
    ) -> Dict[str, float]:
        """取引コスト計算"""

        commission = position_value * self.commission_rate
        spread = position_value * self.spread_rate
        slippage = position_value * self.slippage_rate

        total_cost = commission + spread + slippage

        return {
            "commission": commission,
            "spread": spread,
            "slippage": slippage,
            "total_cost": total_cost,
            "cost_ratio": total_cost / position_value,
        }

    def evaluate_signal_performance(
        self, signal: TradingSignal, actual_price: float, days_elapsed: int
    ) -> Dict[str, Any]:
        """シグナル性能評価"""

        if signal.signal_type == SignalType.BUY:
            actual_return = (actual_price - signal.current_price) / signal.current_price
        elif signal.signal_type == SignalType.SELL:
            actual_return = (signal.current_price - actual_price) / signal.current_price
        else:
            actual_return = 0.0

        predicted_return = signal.expected_return

        # 予測精度
        prediction_error = abs(actual_return - predicted_return)
        prediction_accuracy = (
            max(0, 1 - prediction_error / abs(predicted_return))
            if predicted_return != 0
            else 0
        )

        # 取引コスト考慮後リターン
        costs = self.calculate_trading_costs(signal.position_size, signal.signal_type)
        net_return = actual_return - costs["cost_ratio"]

        return {
            "signal_id": f"{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}",
            "symbol": signal.symbol,
            "predicted_return": predicted_return,
            "actual_return": actual_return,
            "net_return": net_return,
            "prediction_accuracy": prediction_accuracy,
            "days_elapsed": days_elapsed,
            "profit_loss": signal.position_size * net_return,
            "costs": costs,
            "precision_87_achieved": signal.precision_87_achieved,
        }

    def get_strategy_info(self) -> Dict[str, Any]:
        """戦略情報取得"""
        return {
            "name": "87%精度統合戦略",
            "version": "1.0.0",
            "precision_threshold": self.precision_threshold,
            "confidence_threshold": self.confidence_threshold,
            "max_position_size": self.max_position_size,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "min_expected_return": self.min_expected_return,
            "trading_costs": {
                "commission_rate": self.commission_rate,
                "spread_rate": self.spread_rate,
                "slippage_rate": self.slippage_rate,
            },
            "integrated_systems": [
                "Precision87BreakthroughSystem",
                "MetaLearningOptimizer",
                "DQNReinforcementLearner",
            ],
        }
