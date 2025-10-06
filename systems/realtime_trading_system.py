#!/usr/bin/env python3
"""リアルタイム取引システム - 84.6%予測精度を活用した自動売買
リアルタイムデータ取得、シグナル検出、自動注文執行を統合
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import logging
import threading
import time
from datetime import datetime

import yfinance as yf

from utils.logger_config import setup_logger

logger = setup_logger(__name__)
from typing import Dict, List, Optional, Union

from config.settings import get_settings
from data.stock_data import StockDataProvider

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    import numpy as np

    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None

        def fit(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            # Avoid division by zero
            self.scale_[self.scale_ == 0] = 1.0
            return self

        def transform(self, X):
            if self.mean_ is None or self.scale_ is None:
                raise ValueError("Scaler has not been fitted yet.")
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X):
            return self.fit(X).transform(X)


try:
    from sklearn.linear_model import LogisticRegression
except ImportError:

    class LogisticRegression:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            # 簡易的なロジスティック回帰の代替実装
            # 実際のアルゴリズムは複雑なため、単純な線形分類器として近似
            self.classes_ = list(set(y))
            self.n_features_in_ = X.shape[1]
            return self

        def predict(self, X):
            # 簡易実装：Xの平均値に基づいてラベルを決定
            if not hasattr(self, "classes_"):
                raise ValueError("Model has not been fitted yet.")
            return [self.classes_[0 if x.mean() < 0.5 else 1] for x in X]

        def predict_proba(self, X):
            # 簡易実装：クラス0が0.5の確率、クラス1が0.5の確率（均等）
            if not hasattr(self, "classes_"):
                raise ValueError("Model has not been fitted yet.")
            prob = [[0.5, 0.5] for _ in X]
            return prob


class RealTimeDataProvider:
    def __init__(self):
        self.settings = get_settings()
        self.symbols = list(self.settings.target_stocks.keys())
        self.cache = {}
        self.last_update = {}

    def get_realtime_data(
        self, symbol: str, period: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """リアルタイムデータ取得"""
        try:
            # 日本株式の場合は.Tを追加
            yahoo_symbol = f"{symbol}.T"

            # yfinanceでリアルタイムデータ取得
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period=period, interval="1m")

            if len(data) == 0:
                logging.warning(f"データ取得失敗: {symbol}")
                return None

            # キャッシュ更新
            self.cache[symbol] = data
            self.last_update[symbol] = datetime.now()

            return data

        except Exception as e:
            logging.exception(f"リアルタイムデータ取得エラー {symbol}: {e}")
            return None

    def get_historical_context(self, symbol: str) -> Optional[pd.DataFrame]:
        """過去データを含むコンテキスト取得"""
        try:
            data_provider = StockDataProvider()
            historical = data_provider.get_stock_data(symbol, "3mo")
            return historical
        except Exception as e:
            logging.exception(f"履歴データ取得エラー {symbol}: {e}")
            return None


class Pattern846Detector:
    """84.6%パターン検出器"""

    def __init__(self):
        self.settings = get_settings()

    def detect_846_pattern(self, data: pd.DataFrame) -> Dict:
        """84.6%パターン検出

        Args:
            data: 株価データ（OHLCV形式）

        Returns:
            検出結果辞書

        """
        try:
            if len(data) < 50:
                return self._no_signal_result("データ不足")

            # 技術指標計算
            close = data["Close"]
            sma_10 = close.rolling(window=10).mean()
            sma_20 = close.rolling(window=20).mean()
            sma_50 = close.rolling(window=50).mean()

            # トレンド継続性チェック
            trend_duration = pd.Series(0, index=data.index)
            for i in range(7, len(data)):
                recent_data = close.iloc[i - 6 : i + 1]
                recent_up = sum(recent_data.diff() > 0)
                recent_down = sum(recent_data.diff() < 0)

                if recent_up >= 7 or recent_down >= 7:
                    trend_duration.iloc[i] = 1

            # 強力なアップトレンド条件
            strong_uptrend = (
                (sma_10 > sma_20)
                & (sma_20 > sma_50)
                & (close > sma_10)
                & (sma_10.pct_change(5) > 0.01)
                & (trend_duration == 1)
            )

            # 最新の信号
            latest_signal = 0
            confidence = 0.0
            reason = "シグナルなし"

            if len(strong_uptrend) > 0 and strong_uptrend.iloc[-1]:
                latest_signal = 1
                confidence = 0.846
                reason = "強力なアップトレンド検出"

            return {
                "signal": latest_signal,
                "confidence": confidence,
                "reason": reason,
                "current_price": float(close.iloc[-1]),
                "trend_data": {
                    "sma_10": float(sma_10.iloc[-1]) if len(sma_10) > 0 else 0,
                    "sma_20": float(sma_20.iloc[-1]) if len(sma_20) > 0 else 0,
                    "sma_50": float(sma_50.iloc[-1]) if len(sma_50) > 0 else 0,
                },
            }

        except Exception as e:
            logging.exception(f"パターン検出エラー: {e}")
            return self._no_signal_result(f"エラー: {e}")

    def _no_signal_result(self, reason: str) -> Dict:
        """シグナルなし結果"""
        return {
            "signal": 0,
            "confidence": 0.0,
            "reason": reason,
            "current_price": 0.0,
            "trend_data": {},
        }


class RiskManager:
    def __init__(self, initial_capital: float = 1000000):
        self.settings = get_settings()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict[str, Union[int, float, datetime]]] = {}
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()

    def calculate_position_size(
        self, symbol: str, signal: int, confidence: float, current_price: float,
    ) -> Dict[str, Union[int, float, str]]:
        """ポジションサイズ計算（Kelly基準 + リスク管理）"""
        # 日次取引制限確認
        if self.daily_trades >= self.settings.realtime.max_daily_trades:
            return {"size": 0, "reason": "日次取引上限"}

        # 総エクスポージャー確認
        total_exposure = sum(float(pos["value"]) for pos in self.positions.values())
        total_account_value = self.current_capital + total_exposure
        exposure_base = max(self.initial_capital, total_account_value)
        max_exposure = exposure_base * self.settings.realtime.max_total_exposure_pct

        if total_exposure >= max_exposure:
            return {"size": 0, "reason": "総エクスポージャー上限"}

        # Kelly基準ベースのポジションサイズ
        base_allocation = self.settings.realtime.max_position_size_pct

        # 信頼度による調整
        confidence_multiplier = confidence / 0.846  # 84.6%を基準

        # 最終ポジションサイズ
        position_ratio = base_allocation * confidence_multiplier
        position_ratio = min(
            position_ratio, self.settings.realtime.max_position_size_pct,
        )

        position_value = self.current_capital * position_ratio
        max_additional_exposure = max_exposure - total_exposure
        position_value = min(position_value, max_additional_exposure)

        if position_value <= 0:
            return {"size": 0, "reason": "総エクスポージャー上限"}

        position_size = int(position_value / current_price / 100) * 100  # 100株単位

        if position_size == 0:
            return {"size": 0, "reason": "最小単位未満"}

        return {
            "size": position_size,
            "value": position_size * current_price,
            "ratio": position_ratio,
            "reason": f"Kelly基準: {confidence:.1%}",
        }

    def update_positions(
        self, symbol: str, action: str, size: int, price: float,
    ) -> None:
        """ポジション更新"""
        if action == "BUY":
            if symbol in self.positions:
                # 既存ポジション拡大
                old_size = int(self.positions[symbol]["size"])
                old_price = float(self.positions[symbol]["avg_price"])
                new_size = old_size + size
                new_avg_price = (old_size * old_price + size * price) / new_size

                self.positions[symbol].update(
                    {
                        "size": new_size,
                        "avg_price": new_avg_price,
                        "value": new_size * price,
                    },
                )
            else:
                # 新規ポジション
                self.positions[symbol] = {
                    "size": size,
                    "avg_price": price,
                    "value": size * price,
                    "entry_time": datetime.now(),
                    "stop_loss": price
                    * (1 - self.settings.realtime.default_stop_loss_pct),
                    "take_profit": price
                    * (1 + self.settings.realtime.default_take_profit_pct),
                }

            self.current_capital -= size * price
            self.daily_trades += 1

        elif action == "SELL":
            if symbol in self.positions:
                current_size = int(self.positions[symbol]["size"])
                if current_size >= size:
                    self.positions[symbol]["size"] = current_size - size
                    self.positions[symbol]["value"] = (
                        int(self.positions[symbol]["size"]) * price
                    )

                    if int(self.positions[symbol]["size"]) == 0:
                        del self.positions[symbol]

                    self.current_capital += size * price
                    self.daily_trades += 1

    def check_daily_reset(self) -> None:
        """日次リセット確認"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_trades = 0
            self.last_reset_date = today


class OrderExecutor:
    def __init__(self, mode: str = "simulation"):
        self.mode = mode
        self.order_history: List[Dict[str, Union[str, int, float, datetime]]] = []

    def execute_order(
        self, symbol: str, action: str, size: int, price: float, confidence: float,
    ) -> Dict[str, Union[str, int, float, datetime]]:
        """注文執行"""
        order: Dict[str, Union[str, int, float, datetime]] = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "action": action,
            "size": size,
            "price": price,
            "confidence": confidence,
            "status": "executed" if self.mode == "simulation" else "pending",
        }

        if self.mode == "simulation":
            # シミュレーションモード
            logging.info(
                f"シミュレーション注文: {action} {symbol} {size}株 @{price:.0f}円 (信頼度: {confidence:.1%})",
            )
            order["order_id"] = f"SIM_{len(self.order_history)}"

        else:
            # 実際の取引（楽天証券API等との連携）
            # 実装時にAPIキーとセキュリティ認証が必要
            logging.warning("実際の取引は未実装")
            order["status"] = "failed"
            order["reason"] = "実際の取引API未実装"

        self.order_history.append(order)
        return order


class RealTimeTradingSystem:
    def __init__(self, initial_capital: float = 1000000):
        self.data_provider = RealTimeDataProvider()
        self.pattern_detector = Pattern846Detector()
        self.risk_manager = RiskManager(initial_capital)
        self.order_executor = OrderExecutor()
        self.settings = get_settings()

        self.is_running = False
        self.monitoring_thread = None

    def start_monitoring(self):
        """リアルタイム監視開始"""
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.start()

        logging.info("リアルタイム監視開始")
        logging.info(f"監視銘柄数: {len(self.settings.target_stocks)}")
        logging.info(f"更新間隔: {self.settings.realtime.update_interval_seconds}秒")

    def stop_monitoring(self):
        """監視停止"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logging.info("リアルタイム監視停止")

    def _monitoring_loop(self):
        """メイン監視ループ"""
        while self.is_running:
            try:
                # 日次リセット確認
                self.risk_manager.check_daily_reset()

                # 市場時間確認
                if (
                    self.settings.realtime.market_hours_only
                    and not self._is_market_hours()
                ):
                    time.sleep(300)  # 市場外は5分間隔
                    continue

                # 全銘柄をスキャン
                for symbol in list(self.settings.target_stocks.keys())[
                    :5
                ]:  # 最初の5銘柄でテスト
                    try:
                        self._process_symbol(symbol)
                    except Exception as e:
                        logging.exception(f"{symbol} 処理エラー: {e}")
                        continue

                # 待機
                time.sleep(self.settings.realtime.update_interval_seconds)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.exception(f"監視ループエラー: {e}")
                time.sleep(60)  # エラー時は1分待機

    def _process_symbol(self, symbol: str):
        """個別銘柄処理"""
        # 履歴データ取得（パターン検出用）
        historical_data = self.data_provider.get_historical_context(symbol)
        if historical_data is None or len(historical_data) < 50:
            return

        # リアルタイムデータで最新価格更新
        realtime_data = self.data_provider.get_realtime_data(symbol)
        if realtime_data is not None and len(realtime_data) > 0:
            # 最新価格で履歴データを更新
            latest_price = realtime_data["Close"].iloc[-1]
            historical_data.loc[historical_data.index[-1], "Close"] = latest_price

        # 84.6%パターン検出
        pattern_result = self.pattern_detector.detect_846_pattern(historical_data)

        if (
            pattern_result["signal"] != 0
            and pattern_result["confidence"]
            >= self.settings.realtime.pattern_confidence_threshold
        ):
            # ポジションサイズ計算
            position_info = self.risk_manager.calculate_position_size(
                symbol,
                pattern_result["signal"],
                pattern_result["confidence"],
                pattern_result["current_price"],
            )

            if position_info["size"] > 0:
                # 注文執行
                action = "BUY" if pattern_result["signal"] == 1 else "SELL"

                order_result = self.order_executor.execute_order(
                    symbol,
                    action,
                    position_info["size"],
                    pattern_result["current_price"],
                    pattern_result["confidence"],
                )

                if order_result["status"] == "executed":
                    # ポジション更新
                    self.risk_manager.update_positions(
                        symbol,
                        action,
                        position_info["size"],
                        pattern_result["current_price"],
                    )

                    logging.info(
                        f"注文執行完了: {symbol} {action} {position_info['size']}株",
                    )
                    logging.info(f"理由: {pattern_result['reason']}")

    def _is_market_hours(self) -> bool:
        """市場時間判定"""
        now = datetime.now()

        # 土日除外
        if now.weekday() >= 5:
            return False

        # 9:00-15:00（簡略化）
        market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=0, second=0, microsecond=0)

        return market_start <= now <= market_end

    def get_status_report(self) -> Dict:
        """システム状況レポート"""
        total_value = self.risk_manager.current_capital

        # ポジション評価
        for symbol, position in self.risk_manager.positions.items():
            try:
                current_data = self.data_provider.get_realtime_data(symbol)
                if current_data is not None:
                    current_price = current_data["Close"].iloc[-1]
                    position_value = position["size"] * current_price
                    total_value += position_value
            except Exception:
                continue

        total_return = (
            (total_value - self.risk_manager.initial_capital)
            / self.risk_manager.initial_capital
            * 100
        )

        return {
            "status": "running" if self.is_running else "stopped",
            "initial_capital": self.risk_manager.initial_capital,
            "current_cash": self.risk_manager.current_capital,
            "total_value": total_value,
            "total_return_pct": total_return,
            "positions_count": len(self.risk_manager.positions),
            "daily_trades": self.risk_manager.daily_trades,
            "order_history_count": len(self.order_executor.order_history),
        }


def main():
    """リアルタイム取引システム実行"""
    print("リアルタイム取引システム - 84.6%予測精度活用")
    print("=" * 60)

    # システム初期化
    system = RealTimeTradingSystem(initial_capital=1000000)

    try:
        # 監視開始
        system.start_monitoring()

        # 状況表示ループ
        while True:
            time.sleep(300)  # 5分ごとに状況表示

            status = system.get_status_report()
            print(f"\n=== システム状況 ({datetime.now().strftime('%H:%M:%S')}) ===")
            print(f"状態: {status['status']}")
            print(f"現金: {status['current_cash']:,.0f}円")
            print(f"総資産: {status['total_value']:,.0f}円")
            print(f"収益率: {status['total_return_pct']:+.1f}%")
            print(f"ポジション数: {status['positions_count']}")
            print(f"本日取引数: {status['daily_trades']}")

    except KeyboardInterrupt:
        print("\n監視停止中...")
        system.stop_monitoring()

        # 最終レポート
        final_status = system.get_status_report()
        print("\n=== 最終結果 ===")
        print(f"初期資金: {final_status['initial_capital']:,.0f}円")
        print(f"最終資産: {final_status['total_value']:,.0f}円")
        print(
            f"総収益: {final_status['total_value'] - final_status['initial_capital']:,.0f}円",
        )
        print(f"収益率: {final_status['total_return_pct']:+.1f}%")


if __name__ == "__main__":
    main()
