# 📄 中期的な推奨銘柄予想アプリ 仕様書（Markdown版）

## 1. アプリ概要
- **目的**
  初心者でも「どの銘柄を買って、いつ売ればいいか」が分かるように、AIが中期（数週間〜数か月）の推奨銘柄をランキング形式で提示する。

- **設計方針**
  - **第1段階（CUI + API）**
    - ターミナルやコマンドラインで銘柄と推奨アクションを表示
    - APIを通じてデータ取得・予測結果を返却
  - **第2段階（GUI）**
    - Webアプリ/スマホアプリに拡張
    - APIの出力形式をそのまま利用してGUI表示

## 2. 利用者像
- 投資初心者（株歴0〜3年程度）
- 「難しい指標より、買う・売る・待つがすぐに知りたい人」
- 数週間〜数か月単位で投資する人（スイング〜中期投資向け）

## 3. 出力イメージ

### 🔹 CUI表示例
実行例: `python clstock_cli.py system predict -s 7203`
=== 今週のおすすめ銘柄（30〜90日向け / 上位3件） ===

[1位] トヨタ自動車 (7203)
   ✅ 買うタイミング: 今週中に「昨日の高値」を超えたら買い
   💰 目安の購入価格: 3,250円前後
   🛑 損切り目安: 3,150円（購入価格より約-3%）
   🎯 利益目標: 3,450円（+6%）、3,600円（+10%）
   📅 保有期間の目安: 1〜2か月

[2位] ソニーグループ (6758)
   ✅ 買うタイミング: 株価が20日平均線に戻ったら買い
   💰 目安の購入価格: 12,100円前後
   🛑 損切り目安: 11,700円（約-3%）
   🎯 利益目標: 12,900円（+6%）、13,300円（+10%）
   📅 保有期間の目安: 2〜3か月

[3位] 任天堂 (7974)
   ✅ 買うタイミング: 株価が週足サポートラインから反発したら買い
   💰 目安の購入価格: 6,200円前後
   🛑 損切り目安: 6,000円（約-3%）
   🎯 利益目標: 6,550円（+6%）、6,820円（+10%）
   📅 保有期間の目安: 1〜2か月

### 🔹 GUI表示イメージ
- **ランキングカード形式**
  - 「買うなら今週」「売るのは2か月以内」など、文章でアドバイス
- **チャート表示**
  - 緑ゾーン＝買いエリア
  - 赤ライン＝損切りライン
  - 青ライン＝利益目標

## 4. 機能概要
### データ取得
- 株価（日足データ）
- 財務指標（EPS, PER, ROEなど）
- ニュースやSNSのポジ/ネガ判定（将来的に追加予定）

### 予測モデル
- **XGBoost/LightGBM** → 銘柄スコアリング
- **LSTM/Transformer** → 株価の時系列予測（30〜90日先）
- 出力：推奨度スコア（0〜100）、買いサイン／売りサインの文章化

### 出力内容
- ランキング（デフォルトはTOP10、必要に応じて任意件数を指定可能（将来的に実装予定））
- 銘柄ごとの詳細アドバイス（買い条件・購入価格・損切り・利益目標・保有期間）

## 5. システム構成
### 第1段階（CUI + API）
- **バックエンド**: Python (FastAPI)

- **CUIクライアント**: `python clstock_cli.py`（メニュー起動）や `python clstock_cli.py system predict -s 7203`（銘柄指定予測）（デフォルト: 10銘柄）
  - 実行ディレクトリ: リポジトリルート（`ClStock/`）
  - 前提条件: `pip install -r requirements.txt` などで依存関係（特に NumPy などの科学計算ライブラリ）を事前にインストールしておく
  - 将来的には、ランキング件数を指定する機能が追加される可能性があります
- **出力形式**: JSON + テキスト
  - **JSONレスポンス仕様**
    - **キーと型**
      - `generated_at` *(string, ISO8601)*: 予測を作成した日時。
      - `universe` *(string)*: 対象市場や指数（例: "JPXプライム"）。
      - `ranking` *(array<object>)*: 推奨銘柄ごとの詳細情報。
        - `rank` *(integer)*: ランキング順位。
        - `symbol` *(string)*: 証券コード。
        - `name` *(string)*: 銘柄名（GUIカードのタイトルとして流用）。
        - `sector` *(string)*: 業種カテゴリ。
        - `score` *(number)*: 推奨スコア (0-100)。
        - `action` *(string)*: 推奨アクション（例: "buy", "hold", "sell"）。
        - `action_text` *(string)*: CUIで表示する文章化された推奨アクション。
        - `entry_condition` *(string)*: エントリー条件（例: 「昨日の高値を超えたら」）。
        - `entry_price` *(object)*: 購入価格目安。
          - `value` *(number)*: 価格（円）。
          - `text` *(string)*: GUIで表示するフォーマット済みテキスト（例: "3,250円前後"）。
        - `stop_loss` *(object)*: 損切りライン情報。
          - `value` *(number)*
          - `text` *(string)*
        - `targets` *(array<object>)*: 目標価格候補。
          - `value` *(number)*: 価格。
          - `text` *(string)*: 例: "3,450円 (+6%)"。
        - `holding_period` *(string)*: 保有期間の目安。
        - `confidence` *(string)*: モデル信頼度（例: "高", "中"）。
        - `notes` *(string)*: 補足コメント（ニュース・イベントなど）。
        - `risk_level` *(string)*: リスク指標（例: "低", "中", "高"）。
        - `chart_refs` *(object)*: GUIチャート連携用情報。
          - `support_levels` *(array<number>)*: サポートライン価格。
          - `resistance_levels` *(array<number>)*: レジスタンスライン価格。
          - `indicators` *(array<string>)*: 注釈に使う主要テクニカル指標。
    - **レスポンス例**
      ```json
      {
        "generated_at": "2024-05-01T09:00:00+09:00",
        "universe": "JPXプライム",
        "ranking": [
          {
            "rank": 1,
            "symbol": "7203",
            "name": "トヨタ自動車",
            "sector": "輸送用機器",
            "score": 92.5,
            "action": "buy",
            "action_text": "✅ 買うタイミング: 今週中に『昨日の高値』を超えたら買い",
            "entry_condition": "今週中に昨日の高値を上抜け",
            "entry_price": { "value": 3250, "text": "3,250円前後" },
            "stop_loss": { "value": 3150, "text": "3,150円（約-3%）" },
            "targets": [
              { "value": 3450, "text": "3,450円（+6%）" },
              { "value": 3600, "text": "3,600円（+10%）" }
            ],
            "holding_period": "1〜2か月",
            "confidence": "高",
            "notes": "決算発表を通過し需給改善の兆し",
            "risk_level": "中",
            "chart_refs": {
              "support_levels": [3150],
              "resistance_levels": [3600],
              "indicators": ["20日移動平均線", "ボリンジャーバンド"]
            }
          }
        ]
      }
      ```
  - **テキスト出力との対応関係**
    - `ranking[].rank` → CUIの「[1位]」などの順位表示、およびGUIカードの順位バッジ。
    - `ranking[].name` + `symbol` → CUIの銘柄行、GUIカードのタイトルとサブタイトル。
    - `ranking[].action_text` → CUI本文の推奨文そのもの。GUIでは `action` をボタン表示、`action_text` を補足説明に使用。
    - `ranking[].entry_price.text` / `stop_loss.text` / `targets[].text` → CUIの価格行と完全一致。GUIでは数値とラベルを分けて表示可能。
    - `ranking[].holding_period` → CUIの「保有期間の目安」行。GUIではバッジやタイムラインに転用。
    - `ranking[].notes` / `confidence` / `risk_level` → CUIの補足コメントに追記可能。GUIではツールチップやタグで提示。
    - `chart_refs` → CUIでは省略可能だが、GUIチャート描画時にサポート/レジスタンスや指標情報として直接利用。

#### APIレスポンス仕様
- 対象エンドポイント: `GET /recommendations?top_n=5`, `GET /recommendation/{symbol}`（いずれも README.md 110-117 行で定義）
- **ランキングレスポンス共通フィールド**
  - `generated_at` *(string, ISO 8601 datetime)*: 推奨結果の生成時刻。
  - `top_n` *(integer)*: レスポンスに含まれる銘柄数。
  - `items` *(array<object>)*: 銘柄ごとの推奨情報。
- **ランキングアイテム必須フィールド**
  - `symbol` *(string)*: 銘柄コード（例: "7203.T"）。
  - `score` *(number)*: 推奨スコア（0〜100）。
  - `action` *(string)*: 取るべきアクション（"buy"/"sell"/"hold" など）。
  - `entry_price` *(object)*: 推奨エントリー価格。
    - `min` *(number)*, `max` *(number)*: 価格帯を指定。
  - `targets` *(array<object>)*: 目標価格のリスト。
    - 各要素は `label` *(string)* と `price` *(number)* を含む。
  - `stop_loss` *(number)*: 損切りライン。
  - `holding_period_days` *(integer)*: 推奨保有日数の目安。
- **銘柄詳細レスポンス必須フィールド**
  - `symbol` *(string)*: 詳細情報の対象銘柄コード。
  - `score` *(number)*: 現在の推奨スコア。
  - `action` *(string)*: アクション種別。
  - `rationale` *(string)*: モデルの推奨理由サマリ。
  - `entry_price`, `targets`, `stop_loss`, `holding_period_days`: 上記ランキングアイテムと同様の構造。
  - `indicators` *(object)*: 主要な技術指標（例: `sma20` *(number)*, `rsi14` *(number)* など）。
- **オプション/拡張フィールド**
  - `confidence` *(number)*: 予測信頼度（将来の拡張で追加可能）。
  - `news_sentiment` *(object)*: ニュース/ SNS センチメント分析結果（「拡張性」方針に沿った発展要素）。
  - `backtest` *(object)*: 過去検証データ（勝率や平均リターンなど）。

##### JSONレスポンス例
**ランキング（`GET /recommendations?top_n=5`）成功例**
```json
{
  "generated_at": "2024-05-01T09:00:00Z",
  "top_n": 3,
  "items": [
    {
      "symbol": "7203.T",
      "score": 87.5,
      "action": "buy",
      "entry_price": {"min": 3200, "max": 3300},
      "targets": [
        {"label": "base", "price": 3450},
        {"label": "stretch", "price": 3600}
      ],
      "stop_loss": 3100,
      "holding_period_days": 45
    },
    {
      "symbol": "6758.T",
      "score": 82.1,
      "action": "buy",
      "entry_price": {"min": 12000, "max": 12200},
      "targets": [
        {"label": "base", "price": 12900},
        {"label": "stretch", "price": 13300}
      ],
      "stop_loss": 11700,
      "holding_period_days": 60
    },
    {
      "symbol": "9432.T",
      "score": 74.3,
      "action": "hold",
      "entry_price": {"min": 4000, "max": 4050},
      "targets": [{"label": "base", "price": 4200}],
      "stop_loss": 3900,
      "holding_period_days": 30
    }
  ]
}
```

**銘柄詳細（`GET /recommendation/{symbol}`）成功例**
```json
{
  "symbol": "7203.T",
  "score": 87.5,
  "action": "buy",
  "rationale": "国内販売台数の伸びとテクニカル指標の改善に基づく買いシグナル。",
  "entry_price": {"min": 3200, "max": 3300},
  "targets": [
    {"label": "base", "price": 3450},
    {"label": "stretch", "price": 3600}
  ],
  "stop_loss": 3100,
  "holding_period_days": 45,
  "indicators": {
    "sma20": 3150,
    "sma60": 3050,
    "rsi14": 58
  },
  "confidence": 0.78,
  "news_sentiment": {
    "score": 0.4,
    "sources": ["nikkei", "reuters"]
  }
}
```

> **Note:** `confidence`、`news_sentiment`、`backtest` などのフィールドは将来拡張向けのオプション項目であり、応答で未使用の場合は省略される。API クライアントは未知フィールドを無視できるように実装し、拡張性と後方互換性を担保する。

### 第2段階（GUI）
- **フロント**: React / Next.js
- **可視化**: D3.js / Plotly
- **機能**: ランキングカード表示、チャートへの買い/売りサイン描画

## 6. 開発ステップ
1. MVP（CUI+API）
2. バックテスト機能
3. GUI化
4. 拡張機能（LINE通知、SNSセンチメント分析など）

## ✅ まとめ
- 初心者向けに「買う・売る・待つ」を文章と価格目安で明示
- 開発は **CUI＋API → GUI拡張** の段階方式
- JSON仕様をベースにすることで拡張性を確保
