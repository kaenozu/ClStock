# QStock コードスタイルと規約

## 命名規則
- クラス名: PascalCase（例：StockDataFetcher）
- 関数名と変数名: snake_case（例：get_stock_data）
- 定数: UPPER_SNAKE_CASE（例：API_KEY）

## 型ヒント
- 関数の引数と戻り値には型ヒントを付ける
- 例：def get_stock_data(self, symbol: str, days: int = 90) -> pd.DataFrame:

## ドキュメンテーション
- 各クラスと関数にはdocstringを記述
- docstringはGoogleスタイルで記述
- 各モジュールの先頭にはモジュールの説明を記述

## その他の規約
- インデントは4つのスペースを使用
- 行の最大長は88文字（blackフォーマッタのデフォルト）
- import文はアルファベット順に並べる