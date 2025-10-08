"""Parameter optimization helpers for the backtest engine."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from trading.backtest_engine import BacktestConfig, BacktestEngine, BacktestResult


class BacktestOptimizer:
    """Run parameter grid searches for :class:`BacktestEngine`."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def optimize_parameters(
        self,
        base_config: BacktestConfig,
        parameter_ranges: Dict[str, List[float]],
        optimization_metric: str,
        engine_factory: Callable[[BacktestConfig], BacktestEngine],
    ) -> Dict[str, Any]:
        """Perform a brute-force parameter search."""
        try:
            best_params: Dict[str, float] = {}
            best_score = float("-inf")
            all_results: List[Tuple[Dict[str, float], float, BacktestResult]] = []

            param_combinations = self._generate_parameter_combinations(parameter_ranges)
            self.logger.info(
                f"パラメータ最適化開始: {len(param_combinations)}通りの組み合わせ",
            )

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(
                        self._test_parameter_combination,
                        base_config,
                        params,
                        optimization_metric,
                        engine_factory,
                    ): params
                    for params in param_combinations[:50]
                }

                for future in as_completed(futures):
                    params = futures[future]
                    try:
                        score, result = future.result()
                        all_results.append((params, score, result))

                        if score > best_score:
                            best_score = score
                            best_params = params

                        self.logger.info(
                            f"パラメータテスト完了: {params} スコア: {score:.4f}",
                        )
                    except Exception as exc:  # pragma: no cover - defensive logging
                        self.logger.error(f"パラメータテストエラー {params}: {exc}")

            return {
                "best_parameters": best_params,
                "best_score": best_score,
                "all_results": all_results,
                "optimization_metric": optimization_metric,
            }

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(f"パラメータ最適化エラー: {exc}")
            return {}

    def _generate_parameter_combinations(
        self,
        parameter_ranges: Dict[str, List[float]],
    ) -> List[Dict[str, float]]:
        import itertools

        keys = list(parameter_ranges.keys())
        values = [parameter_ranges[k] for k in keys]

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _test_parameter_combination(
        self,
        base_config: BacktestConfig,
        params: Dict[str, float],
        metric: str,
        engine_factory: Callable[[BacktestConfig], BacktestEngine],
    ) -> Tuple[float, BacktestResult]:
        test_config = replace(
            base_config,
            precision_threshold=params.get(
                "precision_threshold",
                base_config.precision_threshold,
            ),
            confidence_threshold=params.get(
                "confidence_threshold",
                base_config.confidence_threshold,
            ),
            max_position_size=params.get(
                "max_position_size",
                base_config.max_position_size,
            ),
            target_symbols=base_config.target_symbols,
        )

        test_engine = engine_factory(test_config)
        result = test_engine.run_backtest()

        score = getattr(result, metric, 0.0)
        return score, result
