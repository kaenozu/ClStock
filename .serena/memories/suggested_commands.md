# QStock 推奨コマンド

## 開発
- `python main.py`: FastAPIサーバーの起動
- `python recommend.py --top N`: CUIクライアントで推奨銘柄を取得（Nは取得する件数）

## テスト
- `pytest`: ユニットテストの実行
- `pytest -v`: 詳細なテスト結果の表示

## コード品質
- `black .`: コードのフォーマット
- `flake8 .`: コードの静的解析
- `mypy .`: 型チェック

## 依存関係
- `pip install -r requirements.txt`: 依存関係のインストール