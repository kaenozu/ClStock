"""完全自動投資システムパッケージ。"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AutoRecommendation",
    "HybridPredictorAdapter",
    "RiskAssessment",
    "RiskManagerAdapter",
    "SentimentAnalyzerAdapter",
    "StrategyGeneratorAdapter",
    "FullAutoInvestmentSystem",
    "run_full_auto",
    "build_cli_parser",
    "main",
]


_ADAPTER_EXPORTS = {
    "AutoRecommendation",
    "HybridPredictorAdapter",
    "RiskAssessment",
    "RiskManagerAdapter",
    "SentimentAnalyzerAdapter",
    "StrategyGeneratorAdapter",
}

_SYSTEM_EXPORTS = {"FullAutoInvestmentSystem", "run_full_auto"}
_CLI_EXPORTS = {"build_cli_parser", "main"}


def __getattr__(name: str) -> Any:
    if name in _ADAPTER_EXPORTS:
        module = importlib.import_module(".adapters", __name__)
        return getattr(module, name)
    if name in _SYSTEM_EXPORTS:
        module = importlib.import_module(".system", __name__)
        return getattr(module, name)
    if name in _CLI_EXPORTS:
        module = importlib.import_module(".cli", __name__)
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | _ADAPTER_EXPORTS | _SYSTEM_EXPORTS | _CLI_EXPORTS)
