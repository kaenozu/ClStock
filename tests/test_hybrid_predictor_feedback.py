import asyncio
from collections import deque
from datetime import datetime

import pandas as pd
import pytest

from models_refactored.core.interfaces import PredictionResult, PredictionMode
from models_refactored.hybrid.hybrid_predictor import RefactoredHybridPredictor


class _StubLearner:
    def __init__(self):
        self.calls = []

    def add_prediction_feedback(self, prediction: float, actual: float, symbol: str) -> None:
        self.calls.append((prediction, actual, symbol))


class _StubDataProvider:
    def __init__(self, close_price: float | None = None):
        self._close_price = close_price

    def get_stock_data(self, symbol: str, period: str):
        if self._close_price is None:
            return pd.DataFrame()
        return pd.DataFrame({"Close": [self._close_price]})


# MultiGPUParallelPredictorはRefactoredHybridPredictorに統合されました
# このテストは一時的にコメントアウト
# @pytest.mark.asyncio
# async def test_multi_gpu_predictor_uses_callback_results():
#     # TODO: RefactoredHybridPredictor内のマルチGPU機能でテストを置き換える必要があります
#     pass


def _build_hybrid_for_feedback_testing(close_price: float | None):
    hybrid = RefactoredHybridPredictor(
        enable_cache=False,
        enable_adaptive_optimization=False,
        enable_streaming=False,
        enable_multi_gpu=False,
        enable_real_time_learning=False
    )
    hybrid.prediction_history = deque(maxlen=1000)
    hybrid.data_provider = _StubDataProvider(close_price)
    hybrid.real_time_learning_enabled = True
    hybrid.real_time_learner = _StubLearner()
    return hybrid


def test_record_prediction_uses_actual_price_from_metadata():
    hybrid = _build_hybrid_for_feedback_testing(close_price=None)
    stub_learner = hybrid.real_time_learner

    result = PredictionResult(
        prediction=1200.0,
        confidence=0.8,
        accuracy=85.0,
        timestamp=datetime.now(),
        symbol='6758.T',
        metadata={'current_price': 1180.5, 'system_used': 'test'}
    )

    hybrid._record_prediction('6758.T', result, PredictionMode.BALANCED, 0.01)

    assert stub_learner.calls == [(1200.0, 1180.5, '6758.T')]


def test_record_prediction_skips_when_no_actual_price():
    hybrid = _build_hybrid_for_feedback_testing(close_price=None)
    stub_learner = hybrid.real_time_learner

    result = PredictionResult(
        prediction=950.0,
        confidence=0.6,
        accuracy=80.0,
        timestamp=datetime.now(),
        symbol='7203.T',
        metadata={'system_used': 'test'}
    )

    hybrid._record_prediction('7203.T', result, PredictionMode.BALANCED, 0.02)

    assert stub_learner.calls == []

def test_record_prediction_falls_back_to_data_provider():
    hybrid = _build_hybrid_for_feedback_testing(close_price=1505.25)
    stub_learner = hybrid.real_time_learner

    result = PredictionResult(
        prediction=1500.0,
        confidence=0.7,
        accuracy=82.0,
        timestamp=datetime.now(),
        symbol='8031.T',
        metadata={'system_used': 'test'}
    )

    hybrid._record_prediction('8031.T', result, PredictionMode.BALANCED, 0.03)

    assert stub_learner.calls == [(1500.0, 1505.25, '8031.T')]
