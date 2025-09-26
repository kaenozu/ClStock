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
    personal_portfolio_db: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "personal_portfolio.db"
    )
    prediction_history_db: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "prediction_history.db"
    )
    backtest_results_db: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "backtest_results.db"
    )


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

    # 予測履歴設定
    max_prediction_history_size: int = 1000  # 予測履歴の最大保持数

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
class ProcessConfig:
    """プロセス管理設定"""

    # プロセス制限設定
    max_concurrent_processes: int = 10
    max_memory_per_process_mb: int = 1000
    max_cpu_percent_per_process: int = 50

    # 自動再起動設定
    auto_restart_failed: bool = True
    max_restart_attempts: int = 3
    restart_delay_seconds: int = 5

    # 監視設定
    health_check_interval_seconds: int = 5
    process_timeout_seconds: int = 300
    memory_warning_threshold_mb: int = 800
    cpu_warning_threshold_percent: int = 80

    # ログ設定
    enable_process_logging: bool = True
    log_process_output: bool = False  # デバッグ用
    max_log_lines_per_process: int = 1000

    # 優先度設定
    process_priorities: Dict[str, int] = field(
        default_factory=lambda: {
            "dashboard": 10,
            "demo_trading": 5,
            "investment_system": 8,
            "deep_learning": 3,
            "ensemble_test": 2,
            "clstock_main": 7,
            "optimized_system": 6,
            "selective_system": 4,
        }
    )

    # ポート管理設定
    port_range_start: int = 8000
    port_range_end: int = 8100
    auto_assign_ports: bool = True


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
    default_index_subscription: List[str] = field(
        default_factory=lambda: ["NIKKEI", "TOPIX"]
    )
    enable_news_subscription: bool = True
    news_relevance_threshold: float = 0.7

    # 市場時間設定
    market_open_time: str = "09:00"  # JST
    market_close_time: str = "15:00"  # JST
    market_timezone: str = "Asia/Tokyo"
    enable_after_hours_trading: bool = False  # 時間外取引有効フラグ（True/False）


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
    realtime: RealTimeConfig = field(default_factory=RealTimeConfig)
    process: ProcessConfig = field(default_factory=ProcessConfig)

    # 対象銘柄
    target_stocks: Dict[str, str] = field(
        default_factory=lambda: {
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
        }
    )


# グローバル設定インスタンス
settings = AppSettings()


def get_settings() -> AppSettings:
    """設定を取得"""
    return settings


def load_from_env() -> None:
    """環境変数から設定を読み込み"""
    # データベース設定
    personal_portfolio_db = os.getenv("CLSTOCK_PERSONAL_PORTFOLIO_DB")
    if personal_portfolio_db:
        settings.database.personal_portfolio_db = Path(personal_portfolio_db)
    
    prediction_history_db = os.getenv("CLSTOCK_PREDICTION_HISTORY_DB")
    if prediction_history_db:
        settings.database.prediction_history_db = Path(prediction_history_db)
    
    backtest_results_db = os.getenv("CLSTOCK_BACKTEST_RESULTS_DB")
    if backtest_results_db:
        settings.backtest.backtest_results_db = Path(backtest_results_db)

    # 予測設定
    target_accuracy = os.getenv("CLSTOCK_TARGET_ACCURACY")
    if target_accuracy:
        settings.prediction.target_accuracy = float(target_accuracy)
    
    achieved_accuracy = os.getenv("CLSTOCK_ACHIEVED_ACCURACY")
    if achieved_accuracy:
        settings.prediction.achieved_accuracy = float(achieved_accuracy)
    
    baseline_accuracy = os.getenv("CLSTOCK_BASELINE_ACCURACY")
    if baseline_accuracy:
        settings.prediction.baseline_accuracy = float(baseline_accuracy)
    
    max_predicted_change = os.getenv("CLSTOCK_MAX_PREDICTED_CHANGE")
    if max_predicted_change:
        settings.prediction.max_predicted_change_percent = float(max_predicted_change)
    
    min_confidence_threshold = os.getenv("CLSTOCK_MIN_CONFIDENCE_THRESHOLD")
    if min_confidence_threshold:
        settings.prediction.min_confidence_threshold = float(min_confidence_threshold)
    
    max_confidence_threshold = os.getenv("CLSTOCK_MAX_CONFIDENCE_THRESHOLD")
    if max_confidence_threshold:
        settings.prediction.max_confidence_threshold = float(max_confidence_threshold)

    # モデル設定
    xgb_estimators = os.getenv("CLSTOCK_XGB_N_ESTIMATORS")
    if xgb_estimators:
        settings.model.xgb_n_estimators = int(xgb_estimators)
    
    xgb_max_depth = os.getenv("CLSTOCK_XGB_MAX_DEPTH")
    if xgb_max_depth:
        settings.model.xgb_max_depth = int(xgb_max_depth)
    
    xgb_learning_rate = os.getenv("CLSTOCK_XGB_LEARNING_RATE")
    if xgb_learning_rate:
        settings.model.xgb_learning_rate = float(xgb_learning_rate)
    
    lgb_estimators = os.getenv("CLSTOCK_LGB_N_ESTIMATORS")
    if lgb_estimators:
        settings.model.lgb_n_estimators = int(lgb_estimators)
    
    lgb_max_depth = os.getenv("CLSTOCK_LGB_MAX_DEPTH")
    if lgb_max_depth:
        settings.model.lgb_max_depth = int(lgb_max_depth)
    
    lgb_learning_rate = os.getenv("CLSTOCK_LGB_LEARNING_RATE")
    if lgb_learning_rate:
        settings.model.lgb_learning_rate = float(lgb_learning_rate)
    
    train_test_split = os.getenv("CLSTOCK_TRAIN_TEST_SPLIT")
    if train_test_split:
        settings.model.train_test_split = float(train_test_split)
    
    min_training_data = os.getenv("CLSTOCK_MIN_TRAINING_DATA")
    if min_training_data:
        settings.model.min_training_data = int(min_training_data)

    # バックテスト設定
    initial_capital = os.getenv("CLSTOCK_INITIAL_CAPITAL")
    if initial_capital:
        settings.backtest.default_initial_capital = float(initial_capital)
    
    rebalance_freq = os.getenv("CLSTOCK_REBALANCE_FREQUENCY")
    if rebalance_freq:
        settings.backtest.default_rebalance_frequency = int(rebalance_freq)
    
    top_n = os.getenv("CLSTOCK_TOP_N")
    if top_n:
        settings.backtest.default_top_n = int(top_n)
    
    stop_loss_pct = os.getenv("CLSTOCK_STOP_LOSS_PCT")
    if stop_loss_pct:
        settings.backtest.default_stop_loss_pct = float(stop_loss_pct)
    
    take_profit_pct = os.getenv("CLSTOCK_TAKE_PROFIT_PCT")
    if take_profit_pct:
        settings.backtest.default_take_profit_pct = float(take_profit_pct)
    
    max_holding_days = os.getenv("CLSTOCK_MAX_HOLDING_DAYS")
    if max_holding_days:
        settings.backtest.default_max_holding_days = int(max_holding_days)
    
    score_threshold = os.getenv("CLSTOCK_SCORE_THRESHOLD")
    if score_threshold:
        settings.backtest.default_score_threshold = float(score_threshold)
    
    data_period = os.getenv("CLSTOCK_DATA_PERIOD")
    if data_period:
        settings.backtest.default_data_period = data_period

    # トレード設定
    market_open_hour = os.getenv("CLSTOCK_MARKET_OPEN_HOUR")
    if market_open_hour:
        settings.trading.market_open_hour = int(market_open_hour)
    
    market_close_hour = os.getenv("CLSTOCK_MARKET_CLOSE_HOUR")
    if market_close_hour:
        settings.trading.market_close_hour = int(market_close_hour)

    # API設定
    api_title = os.getenv("CLSTOCK_API_TITLE")
    if api_title:
        settings.api.title = api_title
    
    api_description = os.getenv("CLSTOCK_API_DESCRIPTION")
    if api_description:
        settings.api.description = api_description
    
    api_version = os.getenv("CLSTOCK_API_VERSION")
    if api_version:
        settings.api.version = api_version
    
    max_top_n = os.getenv("CLSTOCK_API_MAX_TOP_N")
    if max_top_n:
        settings.api.max_top_n = int(max_top_n)
    
    min_top_n = os.getenv("CLSTOCK_API_MIN_TOP_N")
    if min_top_n:
        settings.api.min_top_n = int(min_top_n)
    
    available_periods = os.getenv("CLSTOCK_AVAILABLE_PERIODS")
    if available_periods:
        settings.api.available_periods = available_periods.split(",")

    # ログ設定
    log_level = os.getenv("CLSTOCK_LOG_LEVEL")
    if log_level:
        settings.logging.level = log_level
    
    log_format = os.getenv("CLSTOCK_LOG_FORMAT")
    if log_format:
        settings.logging.format = log_format
    
    log_file = os.getenv("CLSTOCK_LOG_FILE")
    if log_file:
        settings.logging.log_file = log_file

    # リアルタイム設定
    update_interval = os.getenv("CLSTOCK_RT_UPDATE_INTERVAL")
    if update_interval:
        settings.realtime.update_interval_seconds = int(update_interval)
    
    pattern_confidence = os.getenv("CLSTOCK_RT_PATTERN_CONFIDENCE")
    if pattern_confidence:
        settings.realtime.pattern_confidence_threshold = float(pattern_confidence)
    
    min_trend_days = os.getenv("CLSTOCK_RT_MIN_TREND_DAYS")
    if min_trend_days:
        settings.realtime.min_trend_days = int(min_trend_days)
    
    max_position_size = os.getenv("CLSTOCK_RT_MAX_POSITION_SIZE")
    if max_position_size:
        settings.realtime.max_position_size_pct = float(max_position_size)
    
    stop_loss_pct_rt = os.getenv("CLSTOCK_RT_STOP_LOSS_PCT")
    if stop_loss_pct_rt:
        settings.realtime.default_stop_loss_pct = float(stop_loss_pct_rt)
    
    take_profit_pct_rt = os.getenv("CLSTOCK_RT_TAKE_PROFIT_PCT")
    if take_profit_pct_rt:
        settings.realtime.default_take_profit_pct = float(take_profit_pct_rt)
    
    max_daily_trades = os.getenv("CLSTOCK_RT_MAX_DAILY_TRADES")
    if max_daily_trades:
        settings.realtime.max_daily_trades = int(max_daily_trades)
    
    max_total_exposure = os.getenv("CLSTOCK_RT_MAX_TOTAL_EXPOSURE")
    if max_total_exposure:
        settings.realtime.max_total_exposure_pct = float(max_total_exposure)
    
    data_source = os.getenv("CLSTOCK_RT_DATA_SOURCE")
    if data_source:
        settings.realtime.data_source = data_source
    
    order_execution = os.getenv("CLSTOCK_RT_ORDER_EXECUTION")
    if order_execution:
        settings.realtime.order_execution = order_execution

    # プロセス設定
    max_concurrent_processes = os.getenv("CLSTOCK_MAX_CONCURRENT_PROCESSES")
    if max_concurrent_processes:
        settings.process.max_concurrent_processes = int(max_concurrent_processes)
    
    max_memory_per_process = os.getenv("CLSTOCK_MAX_MEMORY_PER_PROCESS")
    if max_memory_per_process:
        settings.process.max_memory_per_process_mb = int(max_memory_per_process)
    
    max_cpu_percent = os.getenv("CLSTOCK_MAX_CPU_PERCENT")
    if max_cpu_percent:
        settings.process.max_cpu_percent_per_process = float(max_cpu_percent)
    
    auto_restart_failed = os.getenv("CLSTOCK_AUTO_RESTART_FAILED")
    if auto_restart_failed:
        settings.process.auto_restart_failed = auto_restart_failed.lower() == 'true'
    
    max_restart_attempts = os.getenv("CLSTOCK_MAX_RESTART_ATTEMPTS")
    if max_restart_attempts:
        settings.process.max_restart_attempts = int(max_restart_attempts)


def validate_settings(settings: 'AppSettings') -> bool:
    """設定のバリデーション"""
    errors = []
    
    # 予測設定のバリデーション
    if not 0 <= settings.prediction.min_confidence_threshold <= 1:
        errors.append(f"min_confidence_threshold must be between 0 and 1, got {settings.prediction.min_confidence_threshold}")
    
    if not 0 <= settings.prediction.max_confidence_threshold <= 1:
        errors.append(f"max_confidence_threshold must be between 0 and 1, got {settings.prediction.max_confidence_threshold}")
    
    if settings.prediction.min_confidence_threshold > settings.prediction.max_confidence_threshold:
        errors.append("min_confidence_threshold cannot be greater than max_confidence_threshold")
    
    if settings.prediction.max_predicted_change_percent <= 0:
        errors.append(f"max_predicted_change_percent must be positive, got {settings.prediction.max_predicted_change_percent}")
    
    # モデル設定のバリデーション
    if settings.model.train_test_split <= 0 or settings.model.train_test_split >= 1:
        errors.append(f"train_test_split must be between 0 and 1 (exclusive), got {settings.model.train_test_split}")
    
    if settings.model.min_training_data <= 0:
        errors.append(f"min_training_data must be positive, got {settings.model.min_training_data}")
    
    # バックテスト設定のバリデーション
    if settings.backtest.default_initial_capital <= 0:
        errors.append(f"default_initial_capital must be positive, got {settings.backtest.default_initial_capital}")
    
    if settings.backtest.default_score_threshold < 0:
        errors.append(f"default_score_threshold must be non-negative, got {settings.backtest.default_score_threshold}")
    
    # トレード設定のバリデーション
    if settings.trading.min_score < 0 or settings.trading.max_score > 100 or settings.trading.min_score >= settings.trading.max_score:
        errors.append("Invalid trading score range")
    
    # API設定のバリデーション
    if settings.api.max_top_n <= 0:
        errors.append(f"max_top_n must be positive, got {settings.api.max_top_n}")
    
    if settings.api.min_top_n <= 0:
        errors.append(f"min_top_n must be positive, got {settings.api.min_top_n}")
    
    # ログ設定
    if settings.api.max_top_n < settings.api.min_top_n:
        errors.append(f"max_top_n ({settings.api.max_top_n}) cannot be less than min_top_n ({settings.api.min_top_n})")
    
    if errors:
        for error in errors:
            logger.error(f"Settings validation error: {error}")
        return False
    
    return True

# 起動時に環境変数を読み込み
load_from_env()

# 設定のバリデーション
if not validate_settings(settings):
    logger.error("Settings validation failed. Please check the configuration.")
