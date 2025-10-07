# ClStock デモ運用システム

ClStockプロジェクトに実装された1週間のデモ運用システムです。87%精度システムと統合され、実際の取引をしていたら得られた利益・損失を正確にトレースできます。
- 新モデルは命中実績（件数・平均リターン）と信頼度を同時に返すため、推奨根拠を明示できます。

## 🎯 システム概要

このデモ運用システムは、仮想資金を使用して実際の市場データに基づく高精度な取引シミュレーションを提供します。

### 主要機能

1. **デモ取引システム**
   - 仮想資金（デフォルト100万円）での売買シミュレーション
   - 87%精度システムによる自動売買判断
   - リアルタイムデータに基づく実行
   - 実際の取引時間・価格での約定シミュレーション

2. **バックテスト機能**
   - 過去データでの戦略検証
   - 取引コスト考慮（手数料0.1%、スプレッド0.05%、スリッページ0.02%）
   - ポートフォリオ管理とリスク分析

3. **パフォーマンス追跡**
   - 日次・週次・月次のP&L計算
   - シャープレシオ、最大ドローダウン、VaR計算
   - 勝率、平均利益/損失の分析
   - ベンチマーク（日経平均）との比較

4. **リスク管理**
   - Kelly基準によるポジションサイジング
   - 87%精度達成時の積極的ポジション管理
   - ストップロス自動実行（デフォルト5%）
   - VAR計算とリスクアラート

5. **レポート機能**
   - 取引履歴の詳細記録
   - パフォーマンスレポート生成
   - チャート・グラフ生成（資産曲線、月次リターン、リスク分析）
   - CSV/JSONエクスポート

## 🚀 クイックスタート

### 1. 基本的な使用方法

```python
from trading import DemoTrader

# デモトレーダー初期化
demo_trader = DemoTrader(
    initial_capital=1000000,  # 初期資金100万円
    target_symbols=[\"6758.T\", \"7203.T\", \"8306.T\"],  # 対象銘柄
    precision_threshold=87.0,  # 87%精度以上で取引
    confidence_threshold=0.8   # 80%信頼度以上で取引
)

# 1週間のデモ取引開始
session_id = demo_trader.start_demo_trading(session_duration_days=7)
print(f\"デモセッション開始: {session_id}\")

# 取引状況確認（実行中）
status = demo_trader.get_current_status()
print(f\"現在総資産: {status['total_equity']:,.0f}円\")
print(f\"リターン: {status['total_return']:.2f}%\")

# デモ取引終了
final_session = demo_trader.stop_demo_trading()
print(f\"最終リターン: {final_session.total_return:.2f}%\")
```

### 2. 統合テスト実行

```bash
# 統合テストの実行
python demo_trading_system_test.py
```

## 📁 システム構成

```
ClStock/
├── trading/                    # デモ運用システム
│   ├── __init__.py
│   ├── demo_trader.py         # メインデモ取引システム
│   ├── trading_strategy.py    # 87%精度統合取引戦略
│   ├── portfolio_manager.py   # ポートフォリオ管理
│   ├── risk_manager.py        # リスク管理（VaR、Kelly基準）
│   ├── trade_recorder.py      # 取引記録・レポート
│   ├── performance_tracker.py # パフォーマンス分析
│   └── backtest_engine.py     # バックテストエンジン
├── models_new/                # 87%精度システム
│   └── precision/
│       └── precision_87_system.py
├── data/                      # データ管理
│   ├── stock_data.py
│   └── personal_portfolio.db
└── demo_trading_system_test.py # 統合テスト
```

## 🎛️ システム設定

### デモトレーダー設定

```python
demo_trader = DemoTrader(
    initial_capital=1000000,       # 初期資金
    target_symbols=[               # 対象銘柄リスト
        \"6758.T\", \"7203.T\", \"8306.T\", \"9984.T\", \"6861.T\"
    ],
    precision_threshold=87.0,      # 取引実行精度閾値
    confidence_threshold=0.8,      # 信頼度閾値
    update_interval=300            # データ更新間隔（秒）
)
```

### 取引戦略設定

```python
strategy = TradingStrategy(
    max_position_size=0.1,         # 最大ポジションサイズ（10%）
    stop_loss_pct=0.05,           # ストップロス（5%）
    take_profit_pct=0.15,         # 利確（15%）
    min_expected_return=0.03       # 最小期待リターン（3%）
)
```

### リスク管理設定

```python
risk_manager = DemoRiskManager(
    max_position_size=0.1,         # 最大ポジション比率
    max_sector_exposure=0.3,       # 最大セクター集中度
    max_drawdown=0.2,             # 最大ドローダウン制限
    var_confidence=0.95           # VaR信頼水準
)
```

## 📊 87%精度システム統合

本システムは既存の87%精度システム（Precision87BreakthroughSystem）と完全統合されています。

### 統合ポイント

1. **高精度予測の活用**
   - 87%精度達成時の積極的ポジション管理
   - 精度に応じた動的ポジションサイジング
   - メタ学習とDQN強化学習の統合判断

2. **実価格予測**
   - 予測価格と現在価格の比較
   - 期待リターン計算
   - 取引タイミングの最適化

3. **信頼度ベース判断**
   - 80%以上の信頼度で取引実行
   - 信頼度に応じたリスク調整
   - 不確実性の高い相場での取引回避

## 📈 パフォーマンス指標

### 基本指標
- **総リターン**: 期間中の総収益率
- **年率リターン**: 年率換算収益率
- **シャープレシオ**: リスク調整後リターン
- **最大ドローダウン**: 最大下落率

### リスク指標
- **VaR (95%)**: 95%信頼区間での最大損失
- **期待ショートフォール**: VaR超過時の平均損失
- **ベータ**: 日経平均に対する感応度
- **ボラティリティ**: 価格変動の標準偏差

### 取引指標
- **勝率**: 利益取引の比率
- **プロフィットファクター**: 総利益/総損失
- **87%精度取引成功率**: 87%精度時の勝率
- **平均保有期間**: 平均ポジション保有日数

## 💰 コスト計算

実際の取引コストを正確に反映：

### 取引コスト
- **手数料**: 0.1%（position_value × 0.001）
- **スプレッド**: 0.05%（position_value × 0.0005）
- **スリッページ**: 0.02%（position_value × 0.0002）

### 税金計算
- **申告分離課税**: 20.315%
- **短期/長期の区分**: 1年未満/以上
- **控除可能経費**: 取引手数料等

## 📋 使用例

### 1. 基本的なデモ取引

```python
from trading import DemoTrader

# 1週間のデモ運用
demo_trader = DemoTrader(initial_capital=1000000)
session_id = demo_trader.start_demo_trading(session_duration_days=7)

# 運用中の状況確認
import time
time.sleep(60)  # 1分待機
status = demo_trader.get_current_status()
print(f\"現在のリターン: {status['total_return']:.2f}%\")

# 運用終了
final_session = demo_trader.stop_demo_trading()
```

### 2. バックテスト実行

```python
from trading import BacktestEngine, BacktestConfig
from datetime import datetime, timedelta

# バックテスト設定
config = BacktestConfig(
    start_date=datetime.now() - timedelta(days=90),
    end_date=datetime.now() - timedelta(days=1),
    initial_capital=1000000,
    target_symbols=[\"6758.T\", \"7203.T\", \"8306.T\"]
)

# バックテスト実行
engine = BacktestEngine(config)
result = engine.run_backtest()

print(f\"バックテスト結果:\")
print(f\"総リターン: {result.total_return:.2%}\")
print(f\"シャープレシオ: {result.sharpe_ratio:.2f}\")
print(f\"最大ドローダウン: {result.max_drawdown:.2%}\")
```

### 3. パフォーマンス分析

```python
from trading import PerformanceTracker

tracker = PerformanceTracker(initial_capital=1000000)

# 日次更新
tracker.update_performance(
    current_portfolio_value=1050000,
    active_positions=5,
    trades_count=2
)

# 期間パフォーマンス取得
period_perf = tracker.get_period_performance()
print(f\"シャープレシオ: {period_perf.sharpe_ratio:.2f}\")

# ベンチマーク比較
benchmark_comp = tracker.get_benchmark_comparison()
print(f\"超過リターン: {benchmark_comp.excess_return:.2%}\")
```

### 4. レポート生成

```python
from trading import TradeRecorder

recorder = TradeRecorder()

# CSV エクスポート
recorder.export_to_csv(\"取引履歴.csv\")

# JSON エクスポート
recorder.export_to_json(\"取引データ.json\")

# 税務計算
tax_calc = recorder.calculate_tax_implications()
print(f\"推定税額: {tax_calc.estimated_tax_liability:,.0f}円\")
```

## ⚠️ 注意事項

1. **デモ取引について**
   - これは仮想取引であり、実際の資金は使用されません
   - 実際の市場データを使用しますが、取引は実行されません
   - パフォーマンスは参考値であり、実取引での結果を保証するものではありません

2. **87%精度システムについて**
   - 精度は過去データに基づく推定値です
   - 市場環境の変化により精度が変動する可能性があります
   - リスク管理を適切に行うことが重要です

3. **技術的制限**
   - リアルタイムデータの取得には制限があります
   - 市場休場時は取引シミュレーションは停止します
   - システムの負荷により更新間隔が延長される場合があります

## 🔧 トラブルシューティング

### よくある問題

1. **データ取得エラー**
   ```
   エラー: データ取得失敗
   解決: インターネット接続とAPI制限を確認
   ```

2. **精度システムエラー**
   ```
   エラー: 87%精度システム初期化失敗
   解決: models_new/precisionディレクトリのファイルを確認
   ```

3. **メモリ不足**
   ```
   エラー: メモリ不足
   解決: 対象銘柄数を減らすか、update_intervalを増加
   ```

### ログの確認

```python
import logging
logging.basicConfig(level=logging.INFO)

# デバッグレベルでの詳細ログ
logging.basicConfig(level=logging.DEBUG)
```

## 📞 サポート

問題が発生した場合：

1. `demo_trading_system_test.py`で統合テストを実行
2. ログファイルを確認
3. 設定パラメータを見直し
4. 必要に応じてコンポーネントを個別テスト

## 🎯 実用的な活用方法

### 1週間のデモ運用での学習ポイント

1. **取引タイミングの理解**
   - 87%精度システムのシグナル精度確認
   - 市場環境と取引成果の関係分析

2. **リスク管理の実践**
   - ポジションサイズの最適化
   - ドローダウン制御の効果確認

3. **コスト影響の把握**
   - 取引コストが収益に与える影響
   - 税金計算の実際

4. **パフォーマンス評価**
   - ベンチマークとの比較
   - リスク調整後リターンの評価

この1週間のデモ運用により、実際の投資判断に必要な全ての要素を安全に体験・検証できます。
