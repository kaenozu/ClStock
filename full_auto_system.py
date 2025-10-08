"""互換性維持のためのフルオートシステムラッパーモジュール。"""

from __future__ import annotations

from analysis.sentiment_analyzer import MarketSentimentAnalyzer
from archive.old_systems.medium_term_prediction import MediumTermPredictionSystem
from config.settings import get_settings
from data.stock_data import StockDataProvider
from data_retrieval_script_generator import generate_colab_data_retrieval_script
from models.advanced.risk_management_framework import (
    PortfolioRisk,
    RiskLevel,
    RiskManager,
)
from models.advanced.trading_strategy_generator import (
    ActionType,
    SignalGenerator,
    StrategyGenerator,
)
from models.base.interfaces import PredictionResult
from models.hybrid.hybrid_predictor import HybridStockPredictor
from trading.tse import PortfolioBacktester
from trading.tse.analysis import StockProfile
from trading.tse.optimizer import PortfolioOptimizer

from systems.full_auto import (
    AutoRecommendation,
    FullAutoInvestmentSystem,
    HybridPredictorAdapter,
    RiskAssessment,
    RiskManagerAdapter,
    SentimentAnalyzerAdapter,
    StrategyGeneratorAdapter,
    build_cli_parser,
    main,
    run_full_auto,
)

__all__ = [
    "AutoRecommendation",
    "FullAutoInvestmentSystem",
    "HybridPredictorAdapter",
    "RiskAssessment",
    "RiskManagerAdapter",
    "SentimentAnalyzerAdapter",
    "StrategyGeneratorAdapter",
    "build_cli_parser",
    "main",
    "run_full_auto",
    # 依存モジュール（既存コードとの互換性のため）
    "MarketSentimentAnalyzer",
    "MediumTermPredictionSystem",
    "get_settings",
    "StockDataProvider",
    "generate_colab_data_retrieval_script",
    "PortfolioRisk",
    "RiskLevel",
    "RiskManager",
    "ActionType",
    "SignalGenerator",
    "StrategyGenerator",
    "PredictionResult",
    "HybridStockPredictor",
    "PortfolioBacktester",
    "StockProfile",
    "PortfolioOptimizer",
]


if __name__ == "__main__":  # pragma: no cover - CLI エントリポイント
    raise SystemExit(main())
