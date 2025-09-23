"""
個別銘柄特化予測モデル
84.6%汎用パターンを各銘柄に最適化
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import logging
from datetime import datetime, timedelta
import joblib
import os

from data.stock_data import StockDataProvider
from config.settings import get_settings
from utils.exceptions import ModelTrainingError, PredictionError, InsufficientDataError

logger = logging.getLogger(__name__)


class StockSpecificPredictor:
    """個別銘柄特化予測システム"""

    def __init__(self):
        self.settings = get_settings()
        self.data_provider = StockDataProvider()
        self.models: Dict[str, Dict[str, Any]] = {}
        self.symbol_features: Dict[str, List[str]] = {}
        self.symbol_performance: Dict[str, Dict[str, float]] = {}

    def analyze_symbol_characteristics(
        self, symbol: str, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """銘柄固有の特性を分析"""
        try:
            if len(data) < 100:
                raise InsufficientDataError(symbol, len(data), 100)

            # 基本統計
            close = data["Close"]
            volume = data["Volume"]

            characteristics = {
                "symbol": symbol,
                "volatility": close.pct_change().std(),
                "avg_volume": volume.mean(),
                "volume_volatility": volume.std() / volume.mean(),
                "trend_persistence": self._calculate_trend_persistence(close),
                "price_range": (close.max() - close.min()) / close.mean(),
                "liquidity_score": self._calculate_liquidity_score(data),
                "sector_pattern": self._identify_sector_pattern(symbol),
            }

            logger.info(
                f"分析完了: {symbol} - ボラティリティ: {characteristics['volatility']:.3f}"
            )
            return characteristics

        except InsufficientDataError:
            raise
        except Exception as e:
            logger.error(f"銘柄特性分析エラー {symbol}: {e}")
            raise ModelTrainingError(f"SymbolAnalysis_{symbol}", str(e))

    def _calculate_trend_persistence(self, close: pd.Series) -> float:
        """トレンド持続性を計算"""
        returns = close.pct_change().dropna()

        # 連続した同方向の動きを検出
        up_streaks = []
        down_streaks = []
        current_streak = 1

        for i in range(1, len(returns)):
            if (returns.iloc[i] > 0) == (returns.iloc[i - 1] > 0):
                current_streak += 1
            else:
                if returns.iloc[i - 1] > 0:
                    up_streaks.append(current_streak)
                else:
                    down_streaks.append(current_streak)
                current_streak = 1

        avg_streak = (
            np.mean(up_streaks + down_streaks) if (up_streaks + down_streaks) else 1
        )
        return min(avg_streak / 5.0, 1.0)  # 正規化

    def _calculate_liquidity_score(self, data: pd.DataFrame) -> float:
        """流動性スコアを計算"""
        volume = data["Volume"]
        close = data["Close"]

        # 出来高の安定性
        volume_cv = volume.std() / volume.mean()

        # 価格変動と出来高の関係
        price_change = close.pct_change().abs()
        volume_change = volume.pct_change().abs()

        correlation = price_change.corr(volume_change)
        correlation = correlation if not pd.isna(correlation) else 0

        # 流動性スコア (低いほど良い)
        liquidity_score = 1.0 / (1.0 + volume_cv) * (1.0 + abs(correlation))
        return min(liquidity_score, 1.0)

    def _identify_sector_pattern(self, symbol: str) -> str:
        """セクターパターンを識別"""
        # 簡略化されたセクター分類
        tech_symbols = ["6758", "6861", "8035", "6701", "6503"]
        auto_symbols = ["7203", "7267", "7201", "7261", "7269"]
        finance_symbols = ["8316", "8411", "8306"]
        trading_symbols = ["8058", "8001", "8002", "8031"]

        if symbol in tech_symbols:
            return "technology"
        elif symbol in auto_symbols:
            return "automotive"
        elif symbol in finance_symbols:
            return "finance"
        elif symbol in trading_symbols:
            return "trading"
        else:
            return "general"

    def create_symbol_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """銘柄特化特徴量を作成"""
        df = data.copy()

        # 基本的な84.6%パターン特徴量
        df = self._add_846_features(df)

        # 銘柄固有特徴量
        characteristics = self.analyze_symbol_characteristics(symbol, data)

        # ボラティリティベース特徴量
        if characteristics["volatility"] > 0.03:  # 高ボラティリティ
            df = self._add_volatility_features(df)

        # 流動性ベース特徴量
        if characteristics["liquidity_score"] > 0.7:  # 高流動性
            df = self._add_liquidity_features(df)

        # セクター特化特徴量
        sector = characteristics["sector_pattern"]
        if sector == "technology":
            df = self._add_tech_features(df)
        elif sector == "automotive":
            df = self._add_auto_features(df)
        elif sector == "finance":
            df = self._add_finance_features(df)

        # 特徴量リストを保存
        feature_columns = [
            col
            for col in df.columns
            if col
            not in ["Open", "High", "Low", "Close", "Volume", "Symbol", "CompanyName"]
        ]
        self.symbol_features[symbol] = feature_columns

        return df

    def _add_846_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """84.6%パターンの基本特徴量"""
        close = df["Close"]

        # 移動平均
        df["sma_10"] = close.rolling(10).mean()
        df["sma_20"] = close.rolling(20).mean()
        df["sma_50"] = close.rolling(50).mean()

        # 84.6%の核心パターン
        df["ma_bullish"] = (
            (df["sma_10"] > df["sma_20"])
            & (df["sma_20"] > df["sma_50"])
            & (close > df["sma_10"])
        ).astype(int)

        df["ma_bearish"] = (
            (df["sma_10"] < df["sma_20"])
            & (df["sma_20"] < df["sma_50"])
            & (close < df["sma_10"])
        ).astype(int)

        # トレンド強度
        df["sma10_slope"] = df["sma_10"].pct_change(5)
        df["trend_strength"] = df["sma10_slope"].abs()

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高ボラティリティ銘柄向け特徴量"""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # ボラティリティ指標
        df["atr_14"] = self._calculate_atr(high, low, close, 14)
        df["volatility_ratio"] = df["atr_14"] / close

        # ボラティリティブレイクアウト
        df["vol_breakout"] = (
            df["volatility_ratio"] > df["volatility_ratio"].rolling(20).quantile(0.8)
        ).astype(int)

        return df

    def _add_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高流動性銘柄向け特徴量"""
        volume = df["Volume"]
        close = df["Close"]

        # 出来高指標
        df["volume_sma_20"] = volume.rolling(20).mean()
        df["volume_ratio"] = volume / df["volume_sma_20"]

        # 価格・出来高関係
        df["price_volume_trend"] = (
            (close.pct_change() * df["volume_ratio"]).rolling(5).mean()
        )

        return df

    def _add_tech_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクノロジー株向け特徴量"""
        close = df["Close"]

        # テック株は急激な変動が多い
        df["momentum_5"] = close.pct_change(5)
        df["momentum_acceleration"] = df["momentum_5"].diff()

        # RSI修正版（テック株向け）
        df["rsi_tech"] = self._calculate_rsi(close, 10)  # 短期RSI

        return df

    def _add_auto_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """自動車株向け特徴量"""
        close = df["Close"]

        # 自動車株は季節性がある
        df["quarterly_trend"] = close.rolling(60).mean() / close.rolling(120).mean()

        # 長期トレンド重視
        df["sma_200"] = close.rolling(200).mean()
        df["long_trend"] = (close > df["sma_200"]).astype(int)

        return df

    def _add_finance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """金融株向け特徴量"""
        close = df["Close"]

        # 金融株は金利感応度が高い
        df["rate_sensitivity"] = close.pct_change(20)  # 月次変化率

        # 安定性重視
        df["stability_score"] = 1.0 / (1.0 + close.pct_change().rolling(20).std())

        return df

    def _calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, window: int
    ) -> pd.Series:
        """ATR計算"""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    def _calculate_rsi(self, close: pd.Series, window: int) -> pd.Series:
        """RSI計算"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def train_symbol_model(
        self, symbol: str, lookback_period: str = "2y"
    ) -> Dict[str, Any]:
        """個別銘柄モデルを訓練"""
        try:
            logger.info(f"銘柄特化モデル訓練開始: {symbol}")

            # データ取得
            data = self.data_provider.get_stock_data(symbol, lookback_period)
            if len(data) < 100:
                raise InsufficientDataError(symbol, len(data), 100)

            # 特徴量作成
            df_features = self.create_symbol_features(symbol, data)

            # ターゲット作成（84.6%パターン：3日後0.5%以上上昇）
            target_return = 0.005  # 0.5%
            prediction_days = 3

            df_features["future_return"] = (
                df_features["Close"].pct_change(prediction_days).shift(-prediction_days)
            )
            df_features["target"] = (
                df_features["future_return"] > target_return
            ).astype(int)

            # データ準備
            feature_cols = self.symbol_features[symbol]
            X = df_features[feature_cols].fillna(0)
            y = df_features["target"]

            # 有効なデータのみ
            valid_indices = ~y.isna()
            X = X[valid_indices]
            y = y[valid_indices]

            if len(X) < 50:
                raise InsufficientDataError(symbol, len(X), 50)

            # 複数モデルでの訓練と選択
            models_to_try = {
                "logistic": LogisticRegression(random_state=42, max_iter=1000),
                "random_forest": RandomForestClassifier(
                    n_estimators=100, random_state=42, max_depth=10
                ),
            }

            best_model = None
            best_score = 0
            best_model_name = ""

            # 時系列分割でCV
            tscv = TimeSeriesSplit(n_splits=3)

            for model_name, model in models_to_try.items():
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    score = accuracy_score(y_val, pred)
                    scores.append(score)

                avg_score = np.mean(scores)
                logger.info(f"{symbol} {model_name}: {avg_score:.3f}")

                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    best_model_name = model_name

            # 最終訓練
            best_model.fit(X, y)

            # モデル保存
            model_info = {
                "model": best_model,
                "model_type": best_model_name,
                "features": feature_cols,
                "accuracy": best_score,
                "trained_date": datetime.now(),
                "symbol_characteristics": self.analyze_symbol_characteristics(
                    symbol, data
                ),
            }

            self.models[symbol] = model_info
            self.symbol_performance[symbol] = {
                "accuracy": best_score,
                "model_type": best_model_name,
                "training_samples": len(X),
            }

            logger.info(
                f"✅ {symbol} モデル訓練完了: {best_model_name} {best_score:.3f}"
            )
            return model_info

        except Exception as e:
            logger.error(f"❌ {symbol} モデル訓練エラー: {e}")
            raise ModelTrainingError(f"StockSpecific_{symbol}", str(e))

    def predict_symbol(
        self, symbol: str, current_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """個別銘柄予測実行"""
        try:
            if symbol not in self.models:
                raise ModelTrainingError(f"StockSpecific_{symbol}", "Model not trained")

            model_info = self.models[symbol]

            # 現在データ取得
            if current_data is None:
                current_data = self.data_provider.get_stock_data(symbol, "3mo")

            # 特徴量作成
            df_features = self.create_symbol_features(symbol, current_data)

            # 最新データで予測
            feature_cols = model_info["features"]
            X_current = df_features[feature_cols].iloc[-1:].fillna(0)

            # 予測実行
            model = model_info["model"]
            prediction = model.predict(X_current)[0]

            try:
                probability = model.predict_proba(X_current)[0]
                confidence = max(probability)
            except:
                confidence = 0.5 + abs(prediction - 0.5)

            result = {
                "symbol": symbol,
                "prediction": int(prediction),
                "confidence": float(confidence),
                "model_type": model_info["model_type"],
                "model_accuracy": model_info["accuracy"],
                "current_price": float(current_data["Close"].iloc[-1]),
                "prediction_time": datetime.now(),
                "signal": 1 if prediction == 1 and confidence > 0.7 else 0,
            }

            logger.info(f"予測完了 {symbol}: {prediction} (信頼度: {confidence:.3f})")
            return result

        except Exception as e:
            logger.error(f"予測エラー {symbol}: {e}")
            raise PredictionError(symbol, "StockSpecific", str(e))

    def batch_train_all_symbols(
        self, symbols: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """全銘柄の一括訓練"""
        if symbols is None:
            symbols = list(self.settings.target_stocks.keys())

        results = {}
        successful_count = 0

        logger.info(f"一括モデル訓練開始: {len(symbols)}銘柄")

        for symbol in symbols:
            try:
                result = self.train_symbol_model(symbol)
                results[symbol] = result
                successful_count += 1
            except Exception as e:
                logger.error(f"❌ {symbol} 訓練失敗: {e}")
                results[symbol] = {"error": str(e)}

        logger.info(f"✅ 一括訓練完了: {successful_count}/{len(symbols)} 成功")
        return results

    def save_models(self, model_dir: str = "models/stock_specific") -> None:
        """モデルをファイルに保存"""
        os.makedirs(model_dir, exist_ok=True)

        for symbol, model_info in self.models.items():
            model_path = os.path.join(model_dir, f"{symbol}_model.pkl")
            joblib.dump(model_info, model_path)
            logger.info(f"モデル保存: {model_path}")

    def load_models(self, model_dir: str = "models/stock_specific") -> None:
        """ファイルからモデルを読み込み"""
        if not os.path.exists(model_dir):
            logger.warning(f"モデルディレクトリが存在しません: {model_dir}")
            return

        for filename in os.listdir(model_dir):
            if filename.endswith("_model.pkl"):
                symbol = filename.replace("_model.pkl", "")
                model_path = os.path.join(model_dir, filename)

                try:
                    model_info = joblib.load(model_path)
                    self.models[symbol] = model_info
                    logger.info(f"モデル読み込み: {symbol}")
                except Exception as e:
                    logger.error(f"モデル読み込みエラー {symbol}: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """性能サマリーを取得"""
        if not self.symbol_performance:
            return {"error": "No trained models"}

        accuracies = [perf["accuracy"] for perf in self.symbol_performance.values()]

        return {
            "total_models": len(self.symbol_performance),
            "average_accuracy": np.mean(accuracies),
            "best_accuracy": max(accuracies),
            "worst_accuracy": min(accuracies),
            "models_above_80pct": sum(1 for acc in accuracies if acc > 0.8),
            "models_above_846pct": sum(1 for acc in accuracies if acc > 0.846),
            "individual_performance": self.symbol_performance,
        }
