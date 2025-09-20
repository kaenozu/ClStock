# QStock 技術スタック

## コア技術
- **言語**: Python 3.11
- **フレームワーク**: FastAPI（APIサーバー）
- **機械学習ライブラリ**: 
  - XGBoost（銘柄スコアリング）
  - LightGBM（銘柄スコアリング）
  - TensorFlow/Keras（LSTMモデル）
- **データ処理**: pandas, numpy
- **可視化**: matplotlib, seaborn（将来のGUIで使用予定）

## 開発ツール
- **テスト**: pytest
- **コードフォーマット**: black
- **コード解析**: flake8
- **型チェック**: mypy
- **依存関係管理**: pip, requirements.txt

## その他のツール
- **HTTPクライアント**: requests
- **設定管理**: pyyaml
- **日付処理**: python-dateutil