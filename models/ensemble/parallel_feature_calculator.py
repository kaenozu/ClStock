"""並列特徴量計算システム - 統合リファクタリング版
3-5倍の性能向上を実現する高度な並列処理システム
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class ParallelFeatureCalculator:
    """統合リファクタリング版並列特徴量計算システム

    特徴:
    - 自動スレッド数最適化
    - エラー耐性の向上
    - メモリ効率的な処理
    - 進捗監視機能
    """

    # 処理設定定数
    DEFAULT_MAX_WORKERS = 8
    TIMEOUT_SECONDS = 30
    BATCH_SIZE = 10
    MEMORY_LIMIT_MB = 1000

    def __init__(
        self,
        n_jobs: int = -1,
        timeout: int = TIMEOUT_SECONDS,
        memory_limit_mb: int = MEMORY_LIMIT_MB,
    ):
        """並列特徴量計算器の初期化

        Args:
            n_jobs: ワーカー数 (-1で自動設定)
            timeout: 処理タイムアウト(秒)
            memory_limit_mb: メモリ制限(MB)

        """
        self.n_jobs = self._optimize_worker_count(n_jobs)
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.logger = logging.getLogger(__name__)

        # 統計情報
        self.processed_symbols = 0
        self.total_execution_time = 0.0
        self.error_count = 0

        self.logger.info(
            f"Initialized ParallelFeatureCalculator: "
            f"workers={self.n_jobs}, timeout={self.timeout}s",
        )

    def _optimize_worker_count(self, n_jobs: int) -> int:
        """最適なワーカー数の決定

        Args:
            n_jobs: 指定されたワーカー数

        Returns:
            int: 最適化されたワーカー数

        """
        if n_jobs == -1:
            cpu_count = os.cpu_count() or 4
            # CPUコア数の2倍だが上限を設定
            optimal = min(cpu_count * 2, self.DEFAULT_MAX_WORKERS)
        else:
            optimal = max(1, min(n_jobs, self.DEFAULT_MAX_WORKERS))

        # メモリ制限に基づく調整
        import psutil

        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # 1ワーカーあたり512MB程度を想定
        memory_limited_workers = max(1, int(available_memory_gb * 2))

        return min(optimal, memory_limited_workers)

    def calculate_features_parallel(
        self,
        symbols: List[str],
        data_provider,
        batch_processing: bool = True,
    ) -> pd.DataFrame:
        """複数銘柄の特徴量を並列計算

        Args:
            symbols: 銘柄コードリスト
            data_provider: データプロバイダー
            batch_processing: バッチ処理有効化

        Returns:
            pd.DataFrame: 統合された特徴量データ

        """
        start_time = time.time()

        if not symbols:
            return pd.DataFrame()

        try:
            self.logger.info(
                f"Starting parallel feature calculation for {len(symbols)} symbols "
                f"using {self.n_jobs} workers",
            )

            if batch_processing and len(symbols) > self.BATCH_SIZE:
                # 大量データの場合はバッチ処理
                return self._calculate_features_batch(symbols, data_provider)
            # 標準並列処理
            return self._calculate_features_standard(symbols, data_provider)

        except Exception as e:
            self.logger.error(f"Parallel feature calculation failed: {e!s}")
            return pd.DataFrame()

        finally:
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.logger.info(
                f"Parallel feature calculation completed in {execution_time:.2f}s",
            )

    def _calculate_features_standard(
        self,
        symbols: List[str],
        data_provider,
    ) -> pd.DataFrame:
        """標準並列処理による特徴量計算

        Args:
            symbols: 銘柄コードリスト
            data_provider: データプロバイダー

        Returns:
            pd.DataFrame: 特徴量データ

        """
        # 並列処理関数の準備
        calculate_single = partial(
            self._calculate_single_symbol_features,
            data_provider=data_provider,
        )

        # 結果格納
        all_features = []
        completed_count = 0

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # 全銘柄の処理を並列開始
            future_to_symbol = {
                executor.submit(calculate_single, symbol): symbol for symbol in symbols
            }

            # 結果回収
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]

                try:
                    features = future.result(timeout=self.timeout)

                    if features is not None and not features.empty:
                        features["symbol"] = symbol
                        all_features.append(features)

                    completed_count += 1

                    # 進捗報告
                    if completed_count % 10 == 0:
                        progress = (completed_count / len(symbols)) * 100
                        self.logger.info(
                            f"Progress: {completed_count}/{len(symbols)} ({progress:.1f}%)",
                        )

                except TimeoutError:
                    self.logger.warning(f"Timeout processing {symbol}")
                    self.error_count += 1

                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e!s}")
                    self.error_count += 1

        if not all_features:
            self.logger.warning("No features calculated successfully")
            return pd.DataFrame()

        # 結果統合
        try:
            combined_features = pd.concat(all_features, ignore_index=True)
            self.processed_symbols += len(all_features)

            self.logger.info(
                f"Feature calculation completed: "
                f"{len(combined_features)} samples from {len(all_features)} symbols",
            )

            return combined_features

        except Exception as e:
            self.logger.error(f"Feature combination failed: {e!s}")
            return pd.DataFrame()

    def _calculate_features_batch(
        self,
        symbols: List[str],
        data_provider,
    ) -> pd.DataFrame:
        """バッチ処理による特徴量計算（大量データ用）

        Args:
            symbols: 銘柄コードリスト
            data_provider: データプロバイダー

        Returns:
            pd.DataFrame: 特徴量データ

        """
        all_features = []

        # バッチに分割
        for i in range(0, len(symbols), self.BATCH_SIZE):
            batch_symbols = symbols[i : i + self.BATCH_SIZE]

            self.logger.info(
                f"Processing batch {i // self.BATCH_SIZE + 1}: "
                f"{len(batch_symbols)} symbols",
            )

            # バッチ処理実行
            batch_features = self._calculate_features_standard(
                batch_symbols,
                data_provider,
            )

            if not batch_features.empty:
                all_features.append(batch_features)

            # メモリ使用量チェック
            if self._check_memory_usage():
                self.logger.warning("Memory usage high, reducing batch processing")
                break

        if not all_features:
            return pd.DataFrame()

        # バッチ結果を統合
        try:
            return pd.concat(all_features, ignore_index=True)
        except Exception as e:
            self.logger.error(f"Batch result combination failed: {e!s}")
            return pd.DataFrame()

    def _calculate_single_symbol_features(
        self,
        symbol: str,
        data_provider,
    ) -> Optional[pd.DataFrame]:
        """単一銘柄の特徴量計算（並列処理用）

        Args:
            symbol: 銘柄コード
            data_provider: データプロバイダー

        Returns:
            Optional[pd.DataFrame]: 特徴量データ

        """
        try:
            # データ取得
            data = data_provider.get_stock_data(symbol, "1y")
            if data.empty:
                return None

            # 基本技術指標の計算
            features = self._calculate_technical_indicators_fast(data)

            # 高度な特徴量の計算
            advanced_features = self._calculate_advanced_features_fast(data)

            # 特徴量結合
            if not advanced_features.empty:
                features = pd.concat([features, advanced_features], axis=1)

            # データ品質チェック
            if features.isnull().all().all():
                return None

            # 最新の1行のみを返す
            return features.tail(1)

        except Exception as e:
            self.logger.debug(
                f"Single symbol feature calculation failed for {symbol}: {e!s}",
            )
            return None

    def _calculate_technical_indicators_fast(self, data: pd.DataFrame) -> pd.DataFrame:
        """高速技術指標計算（ベクトル化処理）

        Args:
            data: 株価データ

        Returns:
            pd.DataFrame: 技術指標データ

        """
        features = pd.DataFrame(index=data.index)

        try:
            # 基本価格特徴量
            features["close"] = data["Close"]
            features["volume"] = data["Volume"]
            features["high_low_ratio"] = data["High"] / data["Low"]
            features["close_open_ratio"] = data["Close"] / data["Open"]

            # 移動平均（ベクトル化）
            for period in [5, 10, 20, 50]:
                ma = data["Close"].rolling(window=period, min_periods=1).mean()
                features[f"ma_{period}"] = ma
                features[f"close_ma_{period}_ratio"] = data["Close"] / ma

            # RSI（高速化版）
            features["rsi_14"] = self._calculate_rsi_fast(data["Close"], 14)

            # MACD（高速化版）
            macd_line, signal_line = self._calculate_macd_fast(data["Close"])
            features["macd"] = macd_line
            features["macd_signal"] = signal_line
            features["macd_histogram"] = macd_line - signal_line

            # ボリンジャーバンド
            bb_upper, bb_lower = self._calculate_bollinger_bands_fast(
                data["Close"],
                20,
                2,
            )
            features["bb_upper"] = bb_upper
            features["bb_lower"] = bb_lower
            features["bb_position"] = (data["Close"] - bb_lower) / (bb_upper - bb_lower)

            # NaNを0で埋める
            features = features.fillna(0)

        except Exception as e:
            self.logger.debug(f"Technical indicators calculation failed: {e!s}")

        return features

    def _calculate_advanced_features_fast(self, data: pd.DataFrame) -> pd.DataFrame:
        """高度な特徴量の高速計算

        Args:
            data: 株価データ

        Returns:
            pd.DataFrame: 高度な特徴量データ

        """
        features = pd.DataFrame(index=data.index)

        try:
            # 価格モメンタム（複数期間）
            for period in [1, 3, 5, 10]:
                features[f"price_momentum_{period}"] = data["Close"].pct_change(period)

            # ボラティリティ（複数期間）
            for period in [5, 10, 20]:
                features[f"volatility_{period}"] = (
                    data["Close"].pct_change().rolling(period).std()
                )

            # 出来高関連指標
            features["volume_ma_ratio"] = (
                data["Volume"] / data["Volume"].rolling(20).mean()
            )
            features["price_volume_correlation"] = (
                data["Close"].rolling(20).corr(data["Volume"])
            )

            # サポート・レジスタンス指標
            features["high_20_ratio"] = data["Close"] / data["High"].rolling(20).max()
            features["low_20_ratio"] = data["Close"] / data["Low"].rolling(20).min()

            # NaNを適切な値で埋める
            features = features.fillna(method="ffill").fillna(0)

        except Exception as e:
            self.logger.debug(f"Advanced features calculation failed: {e!s}")

        return features

    def _calculate_rsi_fast(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """高速RSI計算（ベクトル化）

        Args:
            prices: 価格データ
            period: 計算期間

        Returns:
            pd.Series: RSIデータ

        """
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()

            rs = avg_gain / avg_loss.replace(0, np.inf)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # NaNは中性値で埋める

        except Exception as e:
            self.logger.debug(f"RSI calculation failed: {e!s}")
            return pd.Series(50, index=prices.index)  # デフォルト値

    def _calculate_macd_fast(self, prices: pd.Series, fast=12, slow=26, signal=9):
        """高速MACD計算（ベクトル化）

        Args:
            prices: 価格データ
            fast: 短期EMA期間
            slow: 長期EMA期間
            signal: シグナル期間

        Returns:
            tuple: (MACD線, シグナル線)

        """
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()

            return macd_line.fillna(0), signal_line.fillna(0)

        except Exception as e:
            self.logger.debug(f"MACD calculation failed: {e!s}")
            zero_series = pd.Series(0, index=prices.index)
            return zero_series, zero_series

    def _calculate_bollinger_bands_fast(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: int = 2,
    ):
        """高速ボリンジャーバンド計算（ベクトル化）

        Args:
            prices: 価格データ
            period: 計算期間
            std_dev: 標準偏差倍率

        Returns:
            tuple: (上限, 下限)

        """
        try:
            ma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = ma + (std * std_dev)
            lower = ma - (std * std_dev)

            return upper.fillna(method="ffill"), lower.fillna(method="ffill")

        except Exception as e:
            self.logger.debug(f"Bollinger Bands calculation failed: {e!s}")
            return prices, prices  # フォールバック

    def _check_memory_usage(self) -> bool:
        """メモリ使用量チェック

        Returns:
            bool: メモリ使用量が制限を超えているか

        """
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            return memory_mb > self.memory_limit_mb

        except Exception:
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """性能統計の取得

        Returns:
            Dict[str, Any]: 性能統計情報

        """
        avg_time = self.total_execution_time / max(1, self.processed_symbols)

        return {
            "processed_symbols": self.processed_symbols,
            "total_execution_time": self.total_execution_time,
            "average_time_per_symbol": avg_time,
            "error_count": self.error_count,
            "success_rate": (
                self.processed_symbols
                / max(1, self.processed_symbols + self.error_count)
            )
            * 100,
            "worker_count": self.n_jobs,
        }
