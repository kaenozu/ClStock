"""Configuration schema and environment overrides for ClStock."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from utils.logger_config import setup_logger

from .target_universe import get_target_universe

logger = setup_logger(__name__)


def _apply_env_value(
    env: Mapping[str, str],
    env_name: str,
    caster: Callable[[str], Any],
    apply: Callable[[Any], None],
) -> None:
    """Parse an environment variable with caster; preserve defaults if invalid."""
    raw = env.get(env_name)
    if raw is None or raw == "":
        return
    if not isinstance(raw, str):
        raw = str(raw)
    try:
        value = caster(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; keeping default configuration", env_name, raw)
        return
    apply(value)


# 繝励Ο繧ｸ繧ｧ繧ｯ繝医Ν繝ｼ繝医ヱ繧ｹ
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class DatabaseConfig:
    """Database file locations."""

    # 邨ｶ蟇ｾ繝代せ縺ｧ繝・・繧ｿ繝吶・繧ｹ繝輔ぃ繧､繝ｫ繧呈欠螳・
    personal_portfolio_db: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "personal_portfolio.db",
    )
    prediction_history_db: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "prediction_history.db",
    )
    backtest_results_db: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "backtest_results.db",
    )


@dataclass
class MarketDataConfig:
    """Market data provider settings and credentials."""

    provider: str = "local_csv"
    local_cache_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "historical",
    )
    extra_cache_dirs: List[Path] = field(default_factory=list)
    api_base_url: Optional[str] = None
    api_token: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_timeout: float = 10.0
    verify_ssl: bool = True


@dataclass
class PredictionConfig:
    """Prediction thresholds and metrics."""

    # 邊ｾ蠎ｦ險ｭ螳・
    target_accuracy: float = 87.0  # 逶ｮ讓咏ｲｾ蠎ｦ・・・・
    achieved_accuracy: float = 89.18  # 螳滄圀縺ｮ驕疲・邊ｾ蠎ｦ・・・・
    baseline_accuracy: float = 84.6  # 繝吶・繧ｹ繝ｩ繧､繝ｳ邊ｾ蠎ｦ・・・・

    # 莠域ｸｬ蛻ｶ髯占ｨｭ螳・
    max_predicted_change_percent: float = 0.05  # 譛螟ｧ莠域ｸｬ螟牙虚邇・ｼ按ｱ5%・・
    min_confidence_threshold: float = 0.3  # 譛蟆丈ｿ｡鬆ｼ蠎ｦ髢ｾ蛟､
    max_confidence_threshold: float = 0.95  # 譛螟ｧ菫｡鬆ｼ蠎ｦ髢ｾ蛟､

    # 莠域ｸｬ螻･豁ｴ險ｭ螳・
    max_prediction_history_size: int = 1000  # 莠域ｸｬ螻･豁ｴ縺ｮ譛螟ｧ菫晄戟謨ｰ

    # 謚陦捺欠讓呵ｨｭ螳・
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    bollinger_period: int = 20
    bollinger_std: int = 2


@dataclass
class ModelConfig:
    """Model training hyper-parameters."""

    # XGBoost險ｭ螳・
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_random_state: int = 42

    # LightGBM險ｭ螳・
    lgb_n_estimators: int = 200
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.1
    lgb_random_state: int = 42

    # 險鍋ｷｴ險ｭ螳・
    train_test_split: float = 0.8
    min_training_data: int = 100


@dataclass
class BacktestConfig:
    """Backtest defaults and constraints."""

    # 繝・ヵ繧ｩ繝ｫ繝郁ｨｭ螳・
    default_initial_capital: float = 1000000
    default_rebalance_frequency: int = 5
    default_top_n: int = 3
    default_stop_loss_pct: float = 0.05
    default_take_profit_pct: float = 0.10
    default_max_holding_days: int = 30
    default_score_threshold: float = 60

    # 譛滄俣險ｭ螳・
    default_data_period: str = "3y"
    min_historical_data_points: int = 100


@dataclass
class TradingConfig:
    """Trading score and scheduling settings."""

    # 蟶ょｴ譎る俣
    market_open_hour: int = 9
    market_close_hour: int = 15

    # 繧ｹ繧ｳ繧｢險ｭ螳・
    default_score: float = 50.0
    min_score: int = 0
    max_score: int = 100

    # 繝・・繧ｿ譛滄俣險ｭ螳・
    recommendation_data_period: str = "6mo"


@dataclass
class APIConfig:
    """Public API metadata and limits."""

    # FastAPI險ｭ螳・
    title: str = "ClStock API"
    description: str = "荳ｭ譛溽噪縺ｪ謗ｨ螂ｨ驫俶氛莠域Φ繧ｷ繧ｹ繝・Β"
    version: str = "1.0.0"

    # 蛻ｶ髯占ｨｭ螳・
    max_top_n: int = 10
    min_top_n: int = 1

    # 繝・・繧ｿ譛滄俣繧ｪ繝励す繝ｧ繝ｳ
    available_periods: Optional[List[str]] = None

    def __post_init__(self):
        if self.available_periods is None:
            self.available_periods = ["1mo", "3mo", "6mo", "1y", "2y", "3y"]


@dataclass
class LoggingConfig:
    """Logging destinations and rotation settings."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 繝輔ぃ繧､繝ｫ繝ｭ繧ｰ險ｭ螳・
    log_file: str = "logs/clstock.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class ProcessConfig:
    """Process management limits."""

    # 繝励Ο繧ｻ繧ｹ蛻ｶ髯占ｨｭ螳・
    max_concurrent_processes: int = 10
    max_memory_per_process_mb: int = 1000
    max_cpu_percent_per_process: int = 50

    # 閾ｪ蜍募・襍ｷ蜍戊ｨｭ螳・
    auto_restart_failed: bool = True
    max_restart_attempts: int = 3
    restart_delay_seconds: int = 5

    # 逶｣隕冶ｨｭ螳・
    health_check_interval_seconds: int = 5
    process_timeout_seconds: int = 300
    memory_warning_threshold_mb: int = 800
    cpu_warning_threshold_percent: int = 80

    # 繝ｭ繧ｰ險ｭ螳・
    enable_process_logging: bool = True
    log_process_output: bool = False  # 繝・ヰ繝・げ逕ｨ
    max_log_lines_per_process: int = 1000

    # 蜆ｪ蜈亥ｺｦ險ｭ螳・
    process_priorities: Dict[str, int] = field(
        default_factory=lambda: {
            "dashboard": 10,
            "demo_trading": 5,
            "investment_system": 8,
            "deep_learning": 3,
            "integration_test": 2,
            "optimized_system": 6,
            "selective_system": 4,
        },
    )

    # 繝昴・繝育ｮ｡逅・ｨｭ螳・
    port_range_start: int = 8000
    port_range_end: int = 8100
    auto_assign_ports: bool = True


@dataclass
class RealTimeConfig:
    """Real-time execution tuning options."""

    # 繝・・繧ｿ譖ｴ譁ｰ鬆ｻ蠎ｦ
    update_interval_seconds: int = 60  # 1蛻・俣髫・
    market_hours_only: bool = True

    # 84.6%繝代ち繝ｼ繝ｳ讀懷・險ｭ螳・
    pattern_confidence_threshold: float = 0.846
    min_trend_days: int = 7

    # 豕ｨ譁・濤陦瑚ｨｭ螳・
    max_position_size_pct: float = 0.20  # 譛螟ｧ20%
    default_stop_loss_pct: float = 0.05  # 5%謳榊・繧・
    default_take_profit_pct: float = 0.10  # 10%蛻ｩ遒ｺ

    # 繝ｪ繧ｹ繧ｯ邂｡逅・
    max_daily_trades: int = 5
    max_total_exposure_pct: float = 0.80  # 譛螟ｧ80%謚戊ｳ・

    # API險ｭ螳・
    data_source: str = "yahoo"  # yahoo, rakuten, sbi
    order_execution: str = "simulation"  # simulation, live

    # WebSocket險ｭ螳・
    websocket_url: str = ""
    websocket_timeout: int = 10
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10
    max_reconnection_attempts: int = 5
    reconnection_base_delay: float = 1.0

    # 繝・・繧ｿ蜩∬ｳｪ逶｣隕冶ｨｭ螳・
    enable_data_quality_monitoring: bool = True
    max_price_spike_threshold: float = (
        0.1  # 10%縺ｮ萓｡譬ｼ螟牙虚繧偵せ繝代う繧ｯ縺ｨ縺励※讀懷・
    )
    data_latency_warning_threshold: int = 300  # 5蛻・ｻ･荳企≦蟒ｶ縺ｧ隴ｦ蜻・
    min_data_points_for_validation: int = 10

    # 繧ｭ繝｣繝・す繝･險ｭ螳・
    enable_real_time_caching: bool = True
    max_tick_history_per_symbol: int = 1000
    max_order_book_history_per_symbol: int = 100
    cache_cleanup_interval_hours: int = 24
    tick_cache_ttl_seconds: int = 300  # 5蛻・
    order_book_cache_ttl_seconds: int = 60  # 1蛻・

    # 繝ｭ繧ｰ險ｭ螳・
    enable_detailed_logging: bool = True
    log_websocket_messages: bool = False  # 繝・ヰ繝・げ逕ｨ
    log_data_quality_issues: bool = True
    log_cache_operations: bool = False
    performance_logging_interval: int = 300  # 5蛻・俣髫・

    # 逶｣隕冶ｨｭ螳・
    enable_performance_monitoring: bool = True
    enable_market_metrics_calculation: bool = True
    metrics_calculation_interval: int = 60  # 1蛻・俣髫・
    alert_on_connection_loss: bool = True
    alert_on_data_quality_degradation: bool = True
    data_quality_alert_threshold: float = (
        0.95  # 蜩∬ｳｪ95%繧剃ｸ句屓縺｣縺溘ｉ繧｢繝ｩ繝ｼ繝・
    )

    # 繧ｵ繝悶せ繧ｯ繝ｪ繝励す繝ｧ繝ｳ險ｭ螳・
    default_tick_subscription: List[str] = field(default_factory=list)
    default_order_book_subscription: List[str] = field(default_factory=list)
    default_index_subscription: List[str] = field(
        default_factory=lambda: ["NIKKEI", "TOPIX"],
    )
    enable_news_subscription: bool = True
    news_relevance_threshold: float = 0.7

    # 蟶ょｴ譎る俣險ｭ螳・
    market_open_time: str = "09:00"  # JST
    market_close_time: str = "15:00"  # JST
    market_timezone: str = "Asia/Tokyo"
    enable_after_hours_trading: bool = (
        False  # 譎る俣螟門叙蠑墓怏蜉ｹ繝輔Λ繧ｰ・・rue/False・・
    )


@dataclass
class AppSettings:
    """Configuration section."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    realtime: RealTimeConfig = field(default_factory=RealTimeConfig)
    process: ProcessConfig = field(default_factory=ProcessConfig)

    # 蟇ｾ雎｡驫俶氛
    target_stocks: Dict[str, str] = field(
        default_factory=lambda: get_target_universe().english_names,
    )


# Lazy singleton storage for application settings.
_SETTINGS_SINGLETON: Optional["AppSettings"] = None


def load_settings(env: Optional[Mapping[str, str]] = None) -> AppSettings:
    """Create a new AppSettings instance with environment overrides applied."""
    env_map: Mapping[str, str] = os.environ if env is None else env
    settings = AppSettings()
    load_database_settings_from_env(settings, env_map)
    load_market_data_settings_from_env(settings, env_map)
    load_prediction_settings_from_env(settings, env_map)
    load_model_settings_from_env(settings, env_map)
    load_backtest_settings_from_env(settings, env_map)
    load_trading_settings_from_env(settings, env_map)
    load_api_settings_from_env(settings, env_map)
    load_logging_settings_from_env(settings, env_map)
    load_realtime_settings_from_env(settings, env_map)
    load_process_settings_from_env(settings, env_map)
    return settings


def create_settings(env: Optional[Mapping[str, str]] = None) -> AppSettings:
    """Return a validated AppSettings instance based on the given environment."""
    settings = load_settings(env=env)
    if not validate_settings(settings):
        logger.error("Settings validation failed. Please check the configuration.")
    return settings


def get_settings() -> AppSettings:
    """Return the lazily instantiated global AppSettings singleton."""
    global _SETTINGS_SINGLETON
    if _SETTINGS_SINGLETON is None:
        _SETTINGS_SINGLETON = create_settings()
    return _SETTINGS_SINGLETON


def load_from_env(env: Optional[Mapping[str, str]] = None) -> AppSettings:
    """Refresh the global settings singleton using the provided environment mapping."""
    global _SETTINGS_SINGLETON
    _SETTINGS_SINGLETON = create_settings(env=env)
    return _SETTINGS_SINGLETON


def load_database_settings_from_env(
    settings: "AppSettings", env: Mapping[str, str],
) -> None:
    """Apply database overrides from the environment mapping."""
    _apply_env_value(
        env,
        "CLSTOCK_PERSONAL_PORTFOLIO_DB",
        lambda s: Path(s),
        lambda x: setattr(settings.database, "personal_portfolio_db", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_PREDICTION_HISTORY_DB",
        lambda s: Path(s),
        lambda x: setattr(settings.database, "prediction_history_db", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_BACKTEST_RESULTS_DB",
        lambda s: Path(s),
        lambda x: setattr(settings.database, "backtest_results_db", x),
    )


def load_market_data_settings_from_env(
    settings: "AppSettings", env: Mapping[str, str],
) -> None:
    """Apply market data overrides from the environment mapping."""
    _apply_env_value(
        env,
        "CLSTOCK_MARKET_DATA_PROVIDER",
        str,
        lambda x: setattr(settings.market_data, "provider", x.lower()),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MARKET_DATA_LOCAL_CACHE",
        lambda s: Path(s),
        lambda x: setattr(settings.market_data, "local_cache_dir", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MARKET_DATA_EXTRA_CACHES",
        lambda s: [Path(part.strip()) for part in s.split(os.pathsep) if part.strip()],
        lambda x: setattr(settings.market_data, "extra_cache_dirs", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MARKET_DATA_API_BASE",
        str,
        lambda x: setattr(settings.market_data, "api_base_url", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MARKET_DATA_API_TOKEN",
        str,
        lambda x: setattr(settings.market_data, "api_token", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MARKET_DATA_API_KEY",
        str,
        lambda x: setattr(settings.market_data, "api_key", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MARKET_DATA_API_SECRET",
        str,
        lambda x: setattr(settings.market_data, "api_secret", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MARKET_DATA_API_TIMEOUT",
        float,
        lambda x: setattr(settings.market_data, "api_timeout", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MARKET_DATA_VERIFY_SSL",
        lambda s: s.lower() in {"1", "true", "yes"},
        lambda x: setattr(settings.market_data, "verify_ssl", x),
    )


def load_prediction_settings_from_env(
    settings: "AppSettings", env: Mapping[str, str],
) -> None:
    """Apply prediction overrides from the environment mapping."""
    _apply_env_value(
        env,
        "CLSTOCK_TARGET_ACCURACY",
        float,
        lambda x: setattr(settings.prediction, "target_accuracy", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_ACHIEVED_ACCURACY",
        float,
        lambda x: setattr(settings.prediction, "achieved_accuracy", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_BASELINE_ACCURACY",
        float,
        lambda x: setattr(settings.prediction, "baseline_accuracy", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MAX_PREDICTED_CHANGE",
        float,
        lambda x: setattr(settings.prediction, "max_predicted_change_percent", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MIN_CONFIDENCE_THRESHOLD",
        float,
        lambda x: setattr(settings.prediction, "min_confidence_threshold", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MAX_CONFIDENCE_THRESHOLD",
        float,
        lambda x: setattr(settings.prediction, "max_confidence_threshold", x),
    )


def load_model_settings_from_env(
    settings: "AppSettings", env: Mapping[str, str],
) -> None:
    """Apply model overrides from the environment mapping."""
    _apply_env_value(
        env,
        "CLSTOCK_XGB_N_ESTIMATORS",
        int,
        lambda x: setattr(settings.model, "xgb_n_estimators", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_XGB_MAX_DEPTH",
        int,
        lambda x: setattr(settings.model, "xgb_max_depth", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_XGB_LEARNING_RATE",
        float,
        lambda x: setattr(settings.model, "xgb_learning_rate", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_LGB_N_ESTIMATORS",
        int,
        lambda x: setattr(settings.model, "lgb_n_estimators", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_LGB_MAX_DEPTH",
        int,
        lambda x: setattr(settings.model, "lgb_max_depth", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_LGB_LEARNING_RATE",
        float,
        lambda x: setattr(settings.model, "lgb_learning_rate", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_TRAIN_TEST_SPLIT",
        float,
        lambda x: setattr(settings.model, "train_test_split", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MIN_TRAINING_DATA",
        int,
        lambda x: setattr(settings.model, "min_training_data", x),
    )


def load_backtest_settings_from_env(
    settings: "AppSettings", env: Mapping[str, str],
) -> None:
    """Apply backtest overrides from the environment mapping."""
    _apply_env_value(
        env,
        "CLSTOCK_INITIAL_CAPITAL",
        float,
        lambda x: setattr(settings.backtest, "default_initial_capital", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_REBALANCE_FREQUENCY",
        int,
        lambda x: setattr(settings.backtest, "default_rebalance_frequency", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_TOP_N",
        int,
        lambda x: setattr(settings.backtest, "default_top_n", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_STOP_LOSS_PCT",
        float,
        lambda x: setattr(settings.backtest, "default_stop_loss_pct", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_TAKE_PROFIT_PCT",
        float,
        lambda x: setattr(settings.backtest, "default_take_profit_pct", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MAX_HOLDING_DAYS",
        int,
        lambda x: setattr(settings.backtest, "default_max_holding_days", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_SCORE_THRESHOLD",
        float,
        lambda x: setattr(settings.backtest, "default_score_threshold", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_DATA_PERIOD",
        str,
        lambda x: setattr(settings.backtest, "default_data_period", x),
    )


def load_trading_settings_from_env(
    settings: "AppSettings", env: Mapping[str, str],
) -> None:
    """Apply trading overrides from the environment mapping."""
    _apply_env_value(
        env,
        "CLSTOCK_MARKET_OPEN_HOUR",
        int,
        lambda x: setattr(settings.trading, "market_open_hour", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MARKET_CLOSE_HOUR",
        int,
        lambda x: setattr(settings.trading, "market_close_hour", x),
    )


def load_api_settings_from_env(settings: "AppSettings", env: Mapping[str, str]) -> None:
    """Apply API overrides from the environment mapping."""
    _apply_env_value(
        env, "CLSTOCK_API_TITLE", str, lambda x: setattr(settings.api, "title", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_API_DESCRIPTION",
        str,
        lambda x: setattr(settings.api, "description", x),
    )
    _apply_env_value(
        env, "CLSTOCK_API_VERSION", str, lambda x: setattr(settings.api, "version", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_API_MAX_TOP_N",
        int,
        lambda x: setattr(settings.api, "max_top_n", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_API_MIN_TOP_N",
        int,
        lambda x: setattr(settings.api, "min_top_n", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_AVAILABLE_PERIODS",
        lambda s: s.split(","),
        lambda x: setattr(settings.api, "available_periods", x),
    )


def load_logging_settings_from_env(
    settings: "AppSettings", env: Mapping[str, str],
) -> None:
    """Apply logging overrides from the environment mapping."""
    _apply_env_value(
        env, "CLSTOCK_LOG_LEVEL", str, lambda x: setattr(settings.logging, "level", x),
    )
    _apply_env_value(
        env, "CLSTOCK_LOG_FORMAT", str, lambda x: setattr(settings.logging, "format", x),
    )
    _apply_env_value(
        env, "CLSTOCK_LOG_FILE", str, lambda x: setattr(settings.logging, "log_file", x),
    )


def load_realtime_settings_from_env(
    settings: "AppSettings", env: Mapping[str, str],
) -> None:
    """Apply realtime overrides from the environment mapping."""
    _apply_env_value(
        env,
        "CLSTOCK_RT_UPDATE_INTERVAL",
        int,
        lambda x: setattr(settings.realtime, "update_interval_seconds", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_RT_PATTERN_CONFIDENCE",
        float,
        lambda x: setattr(settings.realtime, "pattern_confidence_threshold", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_RT_MIN_TREND_DAYS",
        int,
        lambda x: setattr(settings.realtime, "min_trend_days", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_RT_MAX_POSITION_SIZE",
        float,
        lambda x: setattr(settings.realtime, "max_position_size_pct", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_RT_STOP_LOSS_PCT",
        float,
        lambda x: setattr(settings.realtime, "default_stop_loss_pct", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_RT_TAKE_PROFIT_PCT",
        float,
        lambda x: setattr(settings.realtime, "default_take_profit_pct", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_RT_MAX_DAILY_TRADES",
        int,
        lambda x: setattr(settings.realtime, "max_daily_trades", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_RT_MAX_TOTAL_EXPOSURE",
        float,
        lambda x: setattr(settings.realtime, "max_total_exposure_pct", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_RT_DATA_SOURCE",
        str,
        lambda x: setattr(settings.realtime, "data_source", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_RT_ORDER_EXECUTION",
        str,
        lambda x: setattr(settings.realtime, "order_execution", x),
    )


def load_process_settings_from_env(
    settings: "AppSettings", env: Mapping[str, str],
) -> None:
    """Apply process overrides from the environment mapping."""
    _apply_env_value(
        env,
        "CLSTOCK_MAX_CONCURRENT_PROCESSES",
        int,
        lambda x: setattr(settings.process, "max_concurrent_processes", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MAX_MEMORY_PER_PROCESS",
        int,
        lambda x: setattr(settings.process, "max_memory_per_process_mb", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MAX_CPU_PERCENT",
        float,
        lambda x: setattr(settings.process, "max_cpu_percent_per_process", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_AUTO_RESTART_FAILED",
        lambda s: s.lower() == "true",
        lambda x: setattr(settings.process, "auto_restart_failed", x),
    )
    _apply_env_value(
        env,
        "CLSTOCK_MAX_RESTART_ATTEMPTS",
        int,
        lambda x: setattr(settings.process, "max_restart_attempts", x),
    )


def validate_settings(settings: "AppSettings") -> bool:
    """Configuration section."""
    errors = []

    # 莠域ｸｬ險ｭ螳壹・繝舌Μ繝・・繧ｷ繝ｧ繝ｳ
    if not 0 <= settings.prediction.min_confidence_threshold <= 1:
        errors.append(
            f"min_confidence_threshold must be between 0 and 1, got {settings.prediction.min_confidence_threshold}",
        )

    if not 0 <= settings.prediction.max_confidence_threshold <= 1:
        errors.append(
            f"max_confidence_threshold must be between 0 and 1, got {settings.prediction.max_confidence_threshold}",
        )

    if (
        settings.prediction.min_confidence_threshold
        > settings.prediction.max_confidence_threshold
    ):
        errors.append(
            "min_confidence_threshold cannot be greater than max_confidence_threshold",
        )

    if settings.prediction.max_predicted_change_percent <= 0:
        errors.append(
            f"max_predicted_change_percent must be positive, got {settings.prediction.max_predicted_change_percent}",
        )

    # 繝｢繝・Ν險ｭ螳壹・繝舌Μ繝・・繧ｷ繝ｧ繝ｳ
    if settings.model.train_test_split <= 0 or settings.model.train_test_split >= 1:
        errors.append(
            f"train_test_split must be between 0 and 1 (exclusive), got {settings.model.train_test_split}",
        )

    if settings.model.min_training_data <= 0:
        errors.append(
            f"min_training_data must be positive, got {settings.model.min_training_data}",
        )

    # 繝舌ャ繧ｯ繝・せ繝郁ｨｭ螳壹・繝舌Μ繝・・繧ｷ繝ｧ繝ｳ
    if settings.backtest.default_initial_capital <= 0:
        errors.append(
            f"default_initial_capital must be positive, got {settings.backtest.default_initial_capital}",
        )

    if settings.backtest.default_score_threshold < 0:
        errors.append(
            f"default_score_threshold must be non-negative, got {settings.backtest.default_score_threshold}",
        )

    # 繝医Ξ繝ｼ繝芽ｨｭ螳壹・繝舌Μ繝・・繧ｷ繝ｧ繝ｳ
    if (
        settings.trading.min_score < 0
        or settings.trading.max_score > 100
        or settings.trading.min_score >= settings.trading.max_score
    ):
        errors.append("Invalid trading score range")

    # API險ｭ螳壹・繝舌Μ繝・・繧ｷ繝ｧ繝ｳ
    if settings.api.max_top_n <= 0:
        errors.append(f"max_top_n must be positive, got {settings.api.max_top_n}")

    if settings.api.min_top_n <= 0:
        errors.append(f"min_top_n must be positive, got {settings.api.min_top_n}")

    # 繝ｭ繧ｰ險ｭ螳・
    if settings.api.max_top_n < settings.api.min_top_n:
        errors.append(
            f"max_top_n ({settings.api.max_top_n}) cannot be less than min_top_n ({settings.api.min_top_n})",
        )

    valid_providers = {"local_csv", "http_api", "hybrid"}
    provider = settings.market_data.provider.lower()
    if provider not in valid_providers:
        errors.append(
            f"Unsupported market data provider: {settings.market_data.provider}",
        )
    if provider in {"http_api", "hybrid"} and not settings.market_data.api_base_url:
        errors.append(
            "api_base_url must be configured when using remote market data providers",
        )

    if errors:
        for error in errors:
            logger.error(f"Settings validation error: {error}")
        return False

    return True
