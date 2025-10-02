# ClStock - AI株価予測・投資推奨システム

初心者でも「どの銘柄を買って、いつ売ればいいか」が分かるように、AIが中期（数週間〜数か月）の推奨銘柄をランキング形式で提示するアプリケーションです。

## 機能

- 📊 主要日本株4000銘柄の推奨ランキング（東証4000銘柄対応）
- 💰 買うタイミング、目標価格、損切り価格の提示
- 📅 保有期間の目安
- 🎯 機械学習モデルによる高精度予測（87%精度突破システム）
- 🔐 API認証・レート制限・入力検証を含むセキュアなAPI
- 🔌 REST APIとCLI提供
- 💻 CUIインターフェース
- 📈 リアルタイムデータとWebSocket対応
- 🔄 自動プロセスマネージャーと監視機能
- 🧪 完備されたテストスイート（ユニットテスト・統合テスト）

## インストール

```bash
# 依存関係をインストール
pip install -r requirements.txt

# または、仮想環境を使用
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
vnv\Scripts\activate  # Windows

pip install -r requirements.txt

# PyTorch系の重いライブラリが必要な場合のみ追加でインストール
pip install -r requirements-ml.txt
```

## 使用方法

### CLIクライアント（推奨）

```bash
# CLIメニュー（推奨）
python clstock_cli.py

# サービス管理
python clstock_cli.py service start dashboard    # ダッシュボード起動
python clstock_cli.py service start demo_trading # デモ取引起動
python clstock_cli.py service status             # サービス状態確認
python clstock_cli.py service stop all           # 全サービス停止

# 予測実行
python clstock_cli.py system predict -s 7203     # 特定銘柄の87%精度予測

# データ取得
python clstock_cli.py data fetch -s 7203 -p 1y   # 株価データ取得（1年分）

# セットアップ
python clstock_cli.py setup                       # 初期セットアップ
```

### システム実行

```bash
# 完全自動システム（4000銘柄分析）
python full_auto_system.py

# デモ取引開始
python demo_start.py

# 投資アドバイス取得
python investment_advisor_cui.py
```

### API サーバー

```bash
# APIサーバーを起動
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# または
cd app
python main.py
```

#### API 認証キーの設定

セキュアな API エンドポイントを利用するには、起動する環境で API キーを明示的に設定する必要があります。

- 環境変数 `CLSTOCK_DEV_KEY` と `CLSTOCK_ADMIN_KEY` を設定する。
- もしくは `config/secrets.py` を作成し、`API_KEYS` 辞書に少なくとも開発者と管理者のキーを定義する。

環境変数を利用する例：

```bash
export CLSTOCK_DEV_KEY="your-dev-api-key"
export CLSTOCK_ADMIN_KEY="your-admin-api-key"
```

`config/secrets.py` を利用する例：

```python
# config/secrets.py
API_KEYS = {
    "your-dev-api-key": "developer",
    "your-admin-api-key": "administrator",
}
```

これらの設定が存在しない場合、API の初期化時にエラーとなります。

#### API エンドポイント

- `GET /api/v1/recommendations?top_n=10` - 推奨銘柄ランキング（認証必要、`top_n` を省略した場合はデフォルトの10件を返却。任意の件数を指定可能）
- `GET /api/v1/recommendation/{symbol}` - 特定銘柄の推奨（認証必要）
- `GET /api/v1/stocks` - 利用可能な銘柄一覧
- `GET /api/v1/stock/{symbol}/data` - 株価データと技術指標
- `GET /api/v1/secure/stock/{symbol}/data` - セキュアな株価データ取得（認証必要）
- `GET /api/v1/secure/analysis/{symbol}` - 高度な市場分析（管理者権限必要）

#### API 認証トークンの設定

運用環境では以下の環境変数を必ず設定し、デフォルトの固定トークンは使用しないでください。

```bash
export API_ADMIN_TOKEN="<強力な管理者トークン>"
export API_USER_TOKEN="<一般ユーザー用トークン>"
```

必要に応じて FastAPI のセキュアエンドポイントに追加のトークンを紐付けたい場合は `config/secrets.py` や `CLSTOCK_DEV_KEY`/`CLSTOCK_ADMIN_KEY` などの既存設定を活用できます。

ローカル開発やテストで旧来の固定トークン（`admin_token_secure_2024` など）を利用したい場合のみ、明示的に以下のフラグを有効化してください。

```bash
export API_ENABLE_TEST_TOKENS=1  # 本番環境では絶対に有効化しない
```

このフラグが無効な状態では固定トークンは一切受け付けられず、環境変数で登録したトークンのみが使用されます。

## 主要技術

- 🧠 **87%精度突破統合システム** - メタラーニング・DQN強化学習統合
- 📊 **リアルタイムデータプロバイダー** - WebSocket対応
- 🔐 **APIセキュリティ** - トークン認証・レート制限
- 🔄 **プロセスマネージャー** - 自動再起動・監視機能
- 🧪 **テスト駆動開発** - 90%以上のカバレッジを目指す

## テストの実行

```bash
# すべてのテストを実行
pytest

# 単位テストのみ
pytest tests/unit/

# APIテストのみ
pytest tests/test_api_security.py

# 特定のテストモジュール
pytest tests/unit/test_data/test_stock_data.py

# カバレッジレポート
pytest --cov=.

# カバレッジレポート（HTML出力）
pytest --cov=. --cov-report=html
```

## ディレクトリ構造

```
ClStock/
├── api/                 # APIエンドポイント
├── app/                 # メインアプリケーション
├── config/             # 設定ファイル
├── data/               # データ取得・処理モジュール
├── ml_models/          # 機械学習モデル
├── models_new/         # 新しいモデル実装（87%精度システムなど）
├── models_refactored/  # リファクタリング済みモデル
├── systems/            # システム管理（プロセスマネージャーなど）
├── tests/              # テストスイート
├── trading/            # トレード戦略・ポートフォリオ管理
├── utils/              # ユーティリティ関数
├── research/           # 研究・実験コード
└── README.md
```

## 対象銘柄

- 東証4000銘柄に対応（デフォルトでは流動性と業種バランスを考慮した31銘柄を厳選）

## 出力例

```
============================================================
🏆 今週のおすすめ銘柄（30〜90日向け）
============================================================

[1位] トヨタ自動車 (7203)
   ✅ 買うタイミング: 昨日の高値（3,250円）を超えたら買い
   💰 目安の購入価格: 3,510円
   🛑 損切り目安: 3,152円 (-3.0%)
   🎯 利益目標: 3,442円 (+6.0%)、3,577円 (+10.0%)
   📅 保有期間の目安: 1～2か月
   📊 推奨度スコア: 78.5/100
   💡 推奨理由: 強い上昇トレンドとファンダメンタルズが良好
   🔮 87%精度AI予測: 次週価格は3,650円（+12.0%）を予測
```

## 設定

プロジェクトは `config/settings.py` で設定を一元管理しています。環境変数によるオーバーライドも可能です：

```bash
export CLSTOCK_API_TITLE="My Custom ClStock API"
export CLSTOCK_LOG_LEVEL="DEBUG"
export CLSTOCK_INITIAL_CAPITAL=2000000
```

## 注意事項

- この情報は投資判断の参考であり、投資は自己責任で行ってください
- 実際の投資では、必ず最新の情報を確認してください
- 市場状況により予測が外れる可能性があります
- 高精度予測システムはテスト中であり、実際の投資結果を保証するものではありません

## ライセンス

MIT License
