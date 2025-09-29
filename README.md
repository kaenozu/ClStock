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

#### API エンドポイント

- `GET /recommendations?top_n=5` - 推奨銘柄ランキング（認証必要）
- `GET /recommendation/{symbol}` - 特定銘柄の推奨（認証必要）
- `GET /stocks` - 利用可能な銘柄一覧
- `GET /stock/{symbol}/data` - 株価データと技術指標
- `GET /secure/stock/{symbol}/data` - セキュアな株価データ取得（認証必要）
- `GET /secure/analysis/{symbol}` - 高度な市場分析（管理者権限必要）

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

- 東証4000銘柄に対応（デフォルトで50銘柄が設定済み）

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
