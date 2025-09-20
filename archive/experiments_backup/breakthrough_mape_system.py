#!/usr/bin/env python3
"""
ChatGPTが言及した10-20% MAPE達成のための根本的新アプローチ
問題の核心：従来の手法の根本的見直し
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

from data.stock_data import StockDataProvider

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BreakthroughMAPESystem:
    """ChatGPT理論に基づく10-20% MAPE達成システム"""

    def __init__(self):
        self.data_provider = StockDataProvider()
        self.scaler = RobustScaler()

    def create_intelligent_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """ChatGPT理論に基づく知的特徴量作成"""
        features = pd.DataFrame(index=data.index)

        close = data["Close"]
        volume = data["Volume"]
        high = data["High"]
        low = data["Low"]

        # 1. 方向性重視の特徴量（分類的アプローチ）
        returns_1d = close.pct_change()
        returns_5d = close.pct_change(5)

        # 強いトレンドの検出
        sma_5 = close.rolling(5).mean()
        sma_20 = close.rolling(20).mean()

        # トレンド強度（重要）
        trend_strength = (sma_5 - sma_20) / sma_20
        features["trend_strength"] = trend_strength
        features["trend_consistency"] = trend_strength.rolling(5).std()

        # 価格位置（レンジ内での位置）
        for window in [10, 20]:
            rolling_max = high.rolling(window).max()
            rolling_min = low.rolling(window).min()
            price_position = (close - rolling_min) / (rolling_max - rolling_min)
            features[f"price_position_{window}"] = price_position

        # 2. ボラティリティ正規化された特徴量
        vol_20 = returns_1d.rolling(20).std()
        features["volatility_20"] = vol_20

        # 正規化リターン（重要）
        features["normalized_return_1d"] = returns_1d / vol_20
        features["normalized_return_5d"] = returns_5d / vol_20

        # 3. 平均回帰シグナル
        mean_return_20 = returns_1d.rolling(20).mean()
        features["mean_reversion_signal"] = (returns_1d - mean_return_20) / vol_20

        # 4. 出来高シグナル
        volume_sma = volume.rolling(20).mean()
        volume_ratio = volume / volume_sma
        features["volume_signal"] = volume_ratio
        features["volume_price_divergence"] = (volume_ratio - 1) * returns_1d

        # 5. RSI改良版（正規化）
        rsi_14 = self._calculate_rsi(close, 14)
        features["rsi_normalized"] = (rsi_14 - 50) / 50  # -1 to 1 range
        features["rsi_extremes"] = np.where(
            rsi_14 > 70, 1, np.where(rsi_14 < 30, -1, 0)
        )

        # 6. MACD改良版
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        features["macd_normalized"] = macd / close
        features["macd_divergence"] = (macd - macd_signal) / close

        # 7. 市場状況特徴量
        features["market_stress"] = (
            vol_20 / vol_20.rolling(60).mean()
        )  # 相対ボラティリティ

        # 8. ラグ特徴量（短期記憶）
        for lag in [1, 2, 3]:
            features[f"return_lag_{lag}"] = returns_1d.shift(lag)
            features[f"volume_signal_lag_{lag}"] = features["volume_signal"].shift(lag)

        return features

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_target_variables(
        self, data: pd.DataFrame, prediction_days: int = 7
    ) -> pd.Series:
        """改良されたターゲット変数作成"""
        close = data["Close"]

        # 複数日の平均リターンを使用（ノイズ削減）
        future_returns = []
        for i in range(1, prediction_days + 1):
            daily_return = close.shift(-i).pct_change()
            future_returns.append(daily_return)

        # 平均リターン（より安定）
        avg_future_return = pd.concat(future_returns, axis=1).mean(axis=1)

        return avg_future_return

    def preprocess_data(
        self, features: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """高度な前処理"""
        # 1. 欠損値処理
        features = features.fillna(method="ffill").fillna(0)

        # 2. 異常値処理（Winsorizing）
        for col in features.select_dtypes(include=[np.number]).columns:
            q01 = features[col].quantile(0.01)
            q99 = features[col].quantile(0.99)
            features[col] = np.clip(features[col], q01, q99)

        # 3. 無限値処理
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 4. 有効なサンプルのみ選択
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        features_clean = features[valid_idx]
        target_clean = target[valid_idx]

        return features_clean, target_clean

    def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """アンサンブルモデルトレーニング"""
        models = {
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=200, max_depth=8, random_state=42
            ),
            "huber": HuberRegressor(epsilon=1.35, max_iter=200),
        }

        # 時系列クロスバリデーション
        tscv = TimeSeriesSplit(n_splits=5)
        model_scores = {}
        trained_models = {}

        for name, model in models.items():
            scores = []
            fold_predictions = []
            fold_actuals = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # スケーリング
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)

                # モデル訓練
                model.fit(X_train_scaled, y_train)

                # 予測
                y_pred = model.predict(X_val_scaled)

                # MAPE計算（改良版）
                mape = self.calculate_robust_mape(y_val, y_pred)
                scores.append(mape)

                fold_predictions.extend(y_pred)
                fold_actuals.extend(y_val)

            avg_mape = np.mean(scores)
            model_scores[name] = {
                "mape": avg_mape,
                "std": np.std(scores),
                "predictions": fold_predictions,
                "actuals": fold_actuals,
            }

            # 最終モデルを全データで訓練
            X_scaled = self.scaler.fit_transform(X)
            model.fit(X_scaled, y)
            trained_models[name] = model

            print(f"{name}: MAPE {avg_mape:.2f}% ± {np.std(scores):.2f}%")

        return model_scores, trained_models

    def calculate_robust_mape(
        self, actual: pd.Series, predicted: pd.Series, threshold: float = 0.005
    ) -> float:
        """ロバストMAPE計算（小さなリターンのフィルタリング）"""
        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)

        # 閾値以上の動きのみ評価
        mask = np.abs(actual_arr) >= threshold

        if mask.sum() < 5:  # 最低5サンプル必要
            return float("inf")

        filtered_actual = actual_arr[mask]
        filtered_predicted = predicted_arr[mask]

        mape = (
            np.mean(np.abs((filtered_actual - filtered_predicted) / filtered_actual))
            * 100
        )
        return mape

    def test_breakthrough_system(self, symbols: List[str]) -> Dict:
        """突破システムのテスト"""
        print("ChatGPT理論による10-20% MAPE達成チャレンジ")
        print("=" * 60)

        all_predictions = []
        all_actuals = []
        symbol_results = {}

        for symbol in symbols[:10]:  # より多くの銘柄でテスト
            try:
                print(f"\n処理中: {symbol}")

                # データ取得（より長期間）
                data = self.data_provider.get_stock_data(symbol, "2y")
                if len(data) < 100:
                    continue

                # 特徴量作成
                features = self.create_intelligent_features(data)
                target = self.create_target_variables(data, prediction_days=7)

                # 前処理
                features_clean, target_clean = self.preprocess_data(features, target)

                if len(features_clean) < 50:
                    continue

                print(f"  有効サンプル: {len(features_clean)}")

                # 学習・評価（最新50%をテスト用）
                split_point = int(len(features_clean) * 0.5)
                X_train = features_clean.iloc[:split_point]
                y_train = target_clean.iloc[:split_point]
                X_test = features_clean.iloc[split_point:]
                y_test = target_clean.iloc[split_point:]

                if len(X_test) < 20:
                    continue

                # モデル訓練
                model_scores, trained_models = self.train_ensemble_models(
                    X_train, y_train
                )

                # ベストモデル選択
                best_model_name = min(
                    model_scores.keys(), key=lambda x: model_scores[x]["mape"]
                )
                best_model = trained_models[best_model_name]

                # テストデータで最終評価
                X_test_scaled = self.scaler.transform(X_test)
                test_predictions = best_model.predict(X_test_scaled)

                test_mape = self.calculate_robust_mape(y_test, test_predictions)

                symbol_results[symbol] = {
                    "mape": test_mape,
                    "best_model": best_model_name,
                    "test_samples": len(X_test),
                }

                all_predictions.extend(test_predictions)
                all_actuals.extend(y_test)

                print(f"  ベストモデル: {best_model_name}")
                print(f"  テストMAPE: {test_mape:.2f}%")

                if test_mape < 20:
                    print("  ✓ 目標範囲達成！")

            except Exception as e:
                logger.warning(f"Error processing {symbol}: {str(e)}")
                continue

        # 全体結果
        if all_predictions:
            overall_mape = self.calculate_robust_mape(
                pd.Series(all_actuals), pd.Series(all_predictions)
            )

            print(f"\n" + "=" * 60)
            print("最終結果")
            print("=" * 60)
            print(f"全体MAPE: {overall_mape:.2f}%")
            print(f"総テストサンプル: {len(all_predictions)}")

            # 銘柄別結果
            successful_symbols = [
                s for s, r in symbol_results.items() if r["mape"] < 20
            ]
            if successful_symbols:
                print(f"\n成功銘柄 (MAPE < 20%): {len(successful_symbols)}銘柄")
                for symbol in successful_symbols:
                    result = symbol_results[symbol]
                    print(f"  {symbol}: {result['mape']:.2f}% ({result['best_model']})")

            if overall_mape < 20:
                print(f"\n🎉 ChatGPT理論実証成功！ MAPE {overall_mape:.2f}%")
            elif overall_mape < 30:
                print(f"\n△ 大幅改善！目標まで残り{overall_mape - 20:.1f}%")
            else:
                print(f"\n継続改善が必要。現在{overall_mape:.2f}%")

            return {
                "overall_mape": overall_mape,
                "symbol_results": symbol_results,
                "total_samples": len(all_predictions),
                "successful_symbols": len(successful_symbols),
            }

        return {"error": "No valid results"}


def main():
    """メイン実行"""
    print("ChatGPT理論による10-20% MAPE達成システム")
    print("=" * 60)

    data_provider = StockDataProvider()
    symbols = list(data_provider.jp_stock_codes.keys())

    system = BreakthroughMAPESystem()
    results = system.test_breakthrough_system(symbols)

    if "error" not in results:
        print(f"\n最終評価:")
        print(f"目標MAPE 10-20%に対して実績{results['overall_mape']:.2f}%")
        if results["overall_mape"] <= 20:
            print("✓ ChatGPT理論の正当性確認！")
        else:
            print("△ さらなる改善が必要")


if __name__ == "__main__":
    main()
