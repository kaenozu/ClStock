# Archive整理戦略 2025-09-20

## 削除対象
### 完全削除
- `archive/cache_backup/` - 古いキャッシュファイル（削除済み）
- 重複する実験ファイル（複数のmape_系、breakthrough_系）
- main()関数のみで価値のない小さなテストファイル

### 統合対象
以下のカテゴリで有用なコードを統合：

#### 1. MAPE最適化系（統合先: `research/optimized_mape_system.py`）
保持：
- `research/correct_mape_approach.py` - 最も成熟したアプローチ
- `research/final_mape_breakthrough.py` - 最終的な実装

削除対象：
- `research/breakthrough_10_percent_mape.py`
- `research/breakthrough_mape_system.py`
- `research/conservative_mape_achiever.py`
- `research/advanced_mape_optimizer.py`
- `research/final_realistic_mape.py`
- `research/robust_mape_predictor.py`
- `research/specialized_mape_optimizer.py`
- `research/ultimate_mape_solution.py`

#### 2. 精度向上系（統合先: `research/precision_breakthrough_system.py`）
保持：
- `research/final_breakthrough_846.py` - 84.6%精度達成
- `research/enhanced_846_system.py`
- `research/precision_optimization_system.py`

削除対象：
- `research/absolute_maximum_system.py`
- `research/direct_846_enhancement.py`
- `research/enhanced_ml_predictor.py`
- `research/enhanced_trend_master.py`
- `research/final_mastery_system.py`
- `research/super_enhanced_system.py`
- `research/ultimate_breakthrough_system.py`
- `research/ultimate_directional_system.py`
- `research/ultimate_final_challenger.py`
- `research/ultimate_precision_master.py`
- `research/ultra_precision_predictor.py`

#### 3. 深層学習系（統合先: `research/deep_learning_system.py`）
保持：
- `research/deep_learning_clean.py` - クリーンな実装
- `research/big_data_deep_learning.py` - ビッグデータ対応

削除対象：
- `research/deep_learning_breakthrough.py`

#### 4. 方向性予測系（統合先: `research/directional_prediction_system.py`）
保持：
- `research/binary_direction_predictor.py`
- `research/advanced_directional_predictor.py`

#### 5. バックテスト・評価系（統合先: `research/evaluation_systems.py`）
保持：
- `research/advanced_backtest.py`
- `research/return_rate_backtest.py`
- `research/portfolio_optimizer.py`

削除対象：
- `research/improved_backtest.py`

#### 6. 実用系（保持）
- `research/practical_accuracy_predictor.py`
- `research/practical_predictor.py`
- `research/revolutionary_prediction_system.py` - 946行の大規模システム

#### 7. テスト・検証系（統合先: `research/validation_tools.py`）
保持：
- `research/final_chatgpt_validation.py`
- `research/mape_analyzer.py`
- `research/realtime_test.py`

削除対象：
- `research/test_ml_models.py`
- `research/test_ultra_performance.py`
- `research/simple_return_test.py`

#### 8. ユーティリティ（保持）
- `research/recommend.py` - CUIクライアント
- `research/train_ml_model.py` - 学習用
- `research/immediate_action_plan.py` - アクションプラン

## 実行後の予想構造
```
research/
├── optimized_mape_system.py         # MAPE最適化統合
├── precision_breakthrough_system.py # 精度向上統合
├── deep_learning_system.py          # 深層学習統合
├── directional_prediction_system.py # 方向性予測統合
├── evaluation_systems.py            # 評価システム統合
├── validation_tools.py              # 検証ツール統合
├── practical_accuracy_predictor.py  # 実用系（個別保持）
├── practical_predictor.py
├── revolutionary_prediction_system.py
├── recommend.py                     # ユーティリティ
├── train_ml_model.py
└── immediate_action_plan.py
```

## 削減効果
- ファイル数: 43個 → 12個 (-72%)
- 重複コード大幅削減
- メンテナンス性向上
- 機能性は100%保持