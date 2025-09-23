#!/usr/bin/env python3
"""
データ取得の修正をテスト
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

from data.stock_data import StockDataProvider


def test_data_provider():
    """データプロバイダーをテスト"""
    provider = StockDataProvider()

    # テスト対象銘柄
    test_symbols = ["6758", "7203", "8306"]

    print("=== データ取得テスト ===")
    for symbol in test_symbols:
        try:
            print(
                f"\n{symbol} ({provider.jp_stock_codes.get(symbol, 'Unknown')}) のデータ取得中..."
            )
            data = provider.get_stock_data(symbol, period="1mo")

            if not data.empty:
                print(f"[OK] 成功: {len(data)}件のデータを取得")
                print(f"   期間: {data.index[0]} ~ {data.index[-1]}")
                print(f"   終値: {data['Close'].iloc[-1]:.2f}")
                if "ActualTicker" in data.columns:
                    print(f"   使用チッカー: {data['ActualTicker'].iloc[0]}")
            else:
                print("[NG] データが空です")

        except Exception as e:
            print(f"[ERROR] エラー: {str(e)}")

    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    test_data_provider()
