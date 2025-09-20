"""
ClStock 設定管理
ハードコードされた値を一元管理
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# プロジェクトルートパス
PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class DatabaseConfig:
    """データベース設定"""
    # 絶対パスでデータベースファイルを指定
    personal_portfolio_db: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "personal_portfolio.db")
    prediction_history_db: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "prediction_history.db")
    backtest_results_db: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "backtest_results.db")

@dataclass
class PredictionConfig:
    """予測システム設定（命名統一）"""
    # 精度設定
    target_accuracy: float = 87.0  # 目標精度（%）
    achieved_accuracy: float = 89.18  # 実際の達成精度（%）
    baseline_accuracy: float = 84.6  # ベースライン精度（%）

    # 予測制限設定
    max_predicted_change_percent: float = 0.05  # 最大予測変動率（±5%）
    min_confidence_threshold: float = 0.3  # 最小信頼度閾値
    max_confidence_threshold: float = 0.95  # 最大信頼度閾値

    # 技術指標設定
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    bollinger_period: int = 20
    bollinger_std: int = 2

@dataclass
class ModelConfig:
    """機械学習モデル設定"""

    # XGBoost設定
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_random_state: int = 42

    # LightGBM設定
    lgb_n_estimators: int = 200
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.1
    lgb_random_state: int = 42

    # 訓練設定
    train_test_split: float = 0.8
    min_training_data: int = 100


@dataclass
class BacktestConfig:
    """バックテスト設定"""

    # デフォルト設定
    default_initial_capital: float = 1000000
    default_rebalance_frequency: int = 5
    default_top_n: int = 3
    default_stop_loss_pct: float = 0.05
    default_take_profit_pct: float = 0.10
    default_max_holding_days: int = 30
    default_score_threshold: float = 60

    # 期間設定
    default_data_period: str = "3y"
    min_historical_data_points: int = 100


@dataclass
class TradingConfig:
    """取引設定"""

    # 市場時間
    market_open_hour: int = 9
    market_close_hour: int = 15

    # スコア設定
    default_score: float = 50.0
    min_score: int = 0
    max_score: int = 100

    # データ期間設定
    recommendation_data_period: str = "6mo"


@dataclass
class APIConfig:
    """API設定"""

    # FastAPI設定
    title: str = "ClStock API"
    description: str = "中期的な推奨銘柄予想システム"
    version: str = "1.0.0"

    # 制限設定
    max_top_n: int = 10
    min_top_n: int = 1

    # データ期間オプション
    available_periods: Optional[List[str]] = None

    def __post_init__(self):
        if self.available_periods is None:
            self.available_periods = ["1mo", "3mo", "6mo", "1y", "2y", "3y"]


@dataclass
class LoggingConfig:
    """ログ設定"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # ファイルログ設定
    log_file: str = "logs/clstock.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class AppSettings:
    """アプリケーション全体設定"""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    realtime: "RealTimeConfig" = field(default_factory=lambda: RealTimeConfig())

    # 対象銘柄
    target_stocks: Dict[str, str] = field(default_factory=lambda: {
        # 指定された50銘柄
        "7203": "トヨタ自動車",
        "6758": "ソニーグループ",
        "9432": "NTT",
        "9434": "ソフトバンク",
        "6701": "日本電気",
        "8316": "三井住友フィナンシャルグループ",
        "8411": "みずほフィナンシャルグループ",
        "8306": "三菱UFJフィナンシャル・グループ",
        "8058": "三菱商事",
        "8001": "伊藤忠商事",
        "8002": "丸紅",
        "8031": "三井物産",
        "6902": "デンソー",
        "7267": "ホンダ",
        "6501": "日立製作所",
        "6503": "三菱電機",
        "7751": "キヤノン",
        "8035": "東京エレクトロン",
        "6770": "アルプスアルパイン",
        "9433": "KDDI",
        "6502": "東芝",
        "6752": "パナソニックHD",
        "6954": "ファナック",
        "6861": "キーエンス",
        "4523": "エーザイ",
        "4578": "大塚HD",
        "7201": "日産自動車",
        "7261": "マツダ",
        "7269": "スズキ",
        "4901": "富士フイルムHD",
        "4502": "武田薬品工業",
        "4503": "アステラス製薬",
        "6504": "富士電機",
        "4011": "ヤフー",  # 代替
        "2914": "日本たばこ産業",
        "5020": "ENEOSホールディングス",
        "1605": "INPEX",
        "1332": "日本水産",
        "5201": "AGC",
        "5401": "日本製鉄",
        "6098": "リクルートHD",
        "3865": "北越コーポレーション",
        "6724": "セイコーエプソン",
        "6703": "沖電気",
        "4063": "信越化学工業",
        "4689": "ヤフー",
        "9983": "ファーストリテイリング",
        "4755": "楽天グループ",
        "6367": "ダイキン工業",
        "4519": "中外製薬",
    })


@dataclass
class RealTimeConfig:
    """リアルタイム取引設定"""

    # データ更新頻度
    update_interval_seconds: int = 60  # 1分間隔
    market_hours_only: bool = True

    # 84.6%パターン検出設定
    pattern_confidence_threshold: float = 0.846
    min_trend_days: int = 7

    # 注文執行設定
    max_position_size_pct: float = 0.20  # 最大20%
    default_stop_loss_pct: float = 0.05  # 5%損切り
    default_take_profit_pct: float = 0.10  # 10%利確

    # リスク管理
    max_daily_trades: int = 5
    max_total_exposure_pct: float = 0.80  # 最大80%投資

    # API設定
    data_source: str = "yahoo"  # yahoo, rakuten, sbi
    order_execution: str = "simulation"  # simulation, live
    
    # WebSocket設定
    websocket_url: str = ""
    websocket_timeout: int = 10
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10
    max_reconnection_attempts: int = 5
    reconnection_base_delay: float = 1.0
    
    # データ品質監視設定
    enable_data_quality_monitoring: bool = True
    max_price_spike_threshold: float = 0.1  # 10%の価格変動をスパイクとして検出
    data_latency_warning_threshold: int = 300  # 5分以上遅延で警告
    min_data_points_for_validation: int = 10
    
    # キャッシュ設定
    enable_real_time_caching: bool = True
    max_tick_history_per_symbol: int = 1000
    max_order_book_history_per_symbol: int = 100
    cache_cleanup_interval_hours: int = 24
    tick_cache_ttl_seconds: int = 300  # 5分
    order_book_cache_ttl_seconds: int = 60  # 1分
    
    # ログ設定
    enable_detailed_logging: bool = True
    log_websocket_messages: bool = False  # デバッグ用
    log_data_quality_issues: bool = True
    log_cache_operations: bool = False
    performance_logging_interval: int = 300  # 5分間隔
    
    # 監視設定
    enable_performance_monitoring: bool = True
    enable_market_metrics_calculation: bool = True
    metrics_calculation_interval: int = 60  # 1分間隔
    alert_on_connection_loss: bool = True
    alert_on_data_quality_degradation: bool = True
    data_quality_alert_threshold: float = 0.95  # 品質95%を下回ったらアラート
    
    # サブスクリプション設定
    default_tick_subscription: List[str] = field(default_factory=lambda: [])
    default_order_book_subscription: List[str] = field(default_factory=lambda: [])
    default_index_subscription: List[str] = field(default_factory=lambda: ["NIKKEI", "TOPIX"])
    enable_news_subscription: bool = True
    news_relevance_threshold: float = 0.7
    
    # 市場時間設定
    market_open_time: str = "09:00"  # JST
    market_close_time: str = "15:00"  # JST
    market_timezone: str = "Asia/Tokyo"
    enable_after_hours_trading: bool = False  # 時間外取引有効フラグ（True/False）


# グローバル設定インスタンス
settings = AppSettings()


def get_settings() -> AppSettings:
    """設定を取得"""
    return settings


def load_from_env() -> None:
    """環境変数から設定を読み込み"""
    # API設定
    api_title = os.getenv("CLSTOCK_API_TITLE")
    if api_title:
        settings.api.title = api_title

    # ログレベル
    log_level = os.getenv("CLSTOCK_LOG_LEVEL")
    if log_level:
        settings.logging.level = log_level

    # バックテスト設定
    initial_capital = os.getenv("CLSTOCK_INITIAL_CAPITAL")
    if initial_capital:
        settings.backtest.default_initial_capital = float(initial_capital)

    score_threshold = os.getenv("CLSTOCK_SCORE_THRESHOLD")
    if score_threshold:
        settings.backtest.default_score_threshold = float(score_threshold)


# 起動時に環境変数を読み込み
load_from_env()
