#!/usr/bin/env python3
"""
超高性能予測システムテストスクリプト
"""

import time
import logging
from datetime import datetime
from models.predictor import StockPredictor
from data.stock_data import StockDataProvider

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ultra_performance_system():
    """超高性能システムのテスト"""

    print("=" * 60)
    print("超高性能予測システム テスト開始")
    print("=" * 60)

    # データプロバイダー初期化
    data_provider = StockDataProvider()
    test_symbols = list(data_provider.jp_stock_codes.keys())[:5]  # 最初の5銘柄でテスト

    print(f"テスト対象銘柄: {test_symbols}")
    print()

    # 1. 従来モデル（ルールベース）のテスト
    print("1. 従来モデル（ルールベース）テスト")
    print("-" * 40)

    predictor_basic = StockPredictor(use_ml_model=False, use_ultra_mode=False)

    start_time = time.time()
    basic_scores = {}

    for symbol in test_symbols:
        score = predictor_basic.calculate_score(symbol)
        basic_scores[symbol] = score
        print(f"{symbol}: {score:.1f}")

    basic_time = time.time() - start_time
    print(f"実行時間: {basic_time:.2f}秒")
    print()

    # 2. 機械学習モデルのテスト
    print("2. 機械学習モデルテスト")
    print("-" * 40)

    predictor_ml = StockPredictor(use_ml_model=True, ml_model_type="xgboost", use_ultra_mode=False)

    start_time = time.time()
    ml_scores = {}

    for symbol in test_symbols:
        score = predictor_ml.calculate_score(symbol)
        ml_scores[symbol] = score
        print(f"{symbol}: {score:.1f}")

    ml_time = time.time() - start_time
    print(f"実行時間: {ml_time:.2f}秒")
    print()

    # 3. 超高性能モードのテスト（統合システム）
    print("3. 超高性能モードテスト")
    print("-" * 40)

    try:
        predictor_ultra = StockPredictor(use_ultra_mode=True)

        start_time = time.time()
        ultra_scores = {}

        for symbol in test_symbols:
            score = predictor_ultra.calculate_score(symbol)
            ultra_scores[symbol] = score
            print(f"{symbol}: {score:.1f}")

        ultra_time = time.time() - start_time
        print(f"実行時間: {ultra_time:.2f}秒")
        print()

        # 性能比較
        print("4. 性能比較")
        print("-" * 40)
        print(f"{'銘柄':<10} {'ルール':<8} {'ML':<8} {'Ultra':<8} {'差分':<8}")
        print("-" * 45)

        for symbol in test_symbols:
            basic = basic_scores[symbol]
            ml = ml_scores[symbol]
            ultra = ultra_scores[symbol]
            diff = ultra - basic

            print(f"{symbol:<10} {basic:<8.1f} {ml:<8.1f} {ultra:<8.1f} {diff:<8.1f}")

        print()
        print("実行時間比較:")
        print(f"  ルールベース: {basic_time:.2f}秒")
        print(f"  機械学習:     {ml_time:.2f}秒")
        print(f"  超高性能:     {ultra_time:.2f}秒")

        if ultra_time > 0:
            speedup_vs_basic = basic_time / ultra_time
            speedup_vs_ml = ml_time / ultra_time
            print(f"  高速化倍率: ルール比 {speedup_vs_basic:.1f}x, ML比 {speedup_vs_ml:.1f}x")

    except Exception as e:
        print(f"超高性能モードエラー: {str(e)}")
        print("深層学習ライブラリ（TensorFlow）が必要です")

    print()
    print("=" * 60)
    print("テスト完了")
    print("=" * 60)

def test_feature_engineering():
    """強化された特徴量エンジニアリングのテスト"""

    print("\n特徴量エンジニアリング強化テスト")
    print("-" * 50)

    try:
        from models.ml_models import MLStockPredictor

        ml_predictor = MLStockPredictor()
        data_provider = StockDataProvider()

        # テストデータ取得
        test_symbol = "7203"  # トヨタ
        data = data_provider.get_stock_data(test_symbol, "1y")

        print(f"テスト銘柄: {test_symbol}")
        print(f"データ期間: {len(data)}日")

        # 基本データ情報
        print(f"基本データ列数: {data.shape[1]}")

        # 強化された特徴量作成
        features = ml_predictor.prepare_features(data)

        print(f"強化特徴量列数: {features.shape[1]}")
        print(f"特徴量増加: {features.shape[1] - data.shape[1]}個")

        # 主要特徴量の表示
        print("\n主要特徴量（最初の20個）:")
        for i, col in enumerate(features.columns[:20]):
            print(f"  {i+1:2d}. {col}")

        if len(features.columns) > 20:
            print(f"  ... 他 {len(features.columns) - 20}個")

        print(f"\n特徴量エンジニアリング強化: {features.shape[1]}個の高度な特徴量を生成")

    except Exception as e:
        print(f"特徴量テストエラー: {str(e)}")

def test_caching_performance():
    """キャッシュ性能テスト"""

    print("\nキャッシュ性能テスト")
    print("-" * 30)

    data_provider = StockDataProvider()
    test_symbol = "7203"

    # 1回目（キャッシュなし）
    start_time = time.time()
    data1 = data_provider.get_stock_data(test_symbol, "6mo")
    data1 = data_provider.calculate_technical_indicators(data1)
    first_time = time.time() - start_time

    # 2回目（キャッシュあり）
    start_time = time.time()
    data2 = data_provider.get_stock_data(test_symbol, "6mo")
    data2 = data_provider.calculate_technical_indicators(data2)
    second_time = time.time() - start_time

    print(f"1回目（キャッシュなし）: {first_time:.3f}秒")
    print(f"2回目（キャッシュあり）: {second_time:.3f}秒")

    if second_time > 0:
        speedup = first_time / second_time
        print(f"キャッシュ効果: {speedup:.1f}x 高速化")

if __name__ == "__main__":
    test_ultra_performance_system()
    test_feature_engineering()
    test_caching_performance()