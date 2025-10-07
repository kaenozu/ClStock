"""アテンション機構の軽量な NumPy ベース実装。"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class TemporalAttention:
    """時系列データ用カスタムアテンション機構 (NumPy 版)。"""

    def __init__(self, attention_units: int):
        self.attention_units = attention_units
        self._projection = None

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        data = np.asarray(inputs, dtype=float)
        if data.ndim != 3:
            raise ValueError("inputs は (batch, time, features) 形式である必要があります。")

        batch, time_steps, features = data.shape
        if time_steps == 0:
            return np.zeros((batch, self.attention_units))

        time_weights = np.linspace(1.0, 2.0, time_steps, dtype=float)
        time_weights = time_weights / time_weights.sum()
        aggregated = np.tensordot(time_weights, data, axes=(0, 1))

        if self._projection is None:
            scale = 1.0 / max(features, 1)
            self._projection = np.full((features, self.attention_units), scale)

        output = aggregated @ self._projection
        return np.asarray(output, dtype=float)


class MultiHeadTemporalAttention:
    """Multi-Head Temporal Attention (NumPy 版)。"""

    def __init__(self, num_heads: int, attention_units_per_head: int):
        self.num_heads = num_heads
        self.attention_units_per_head = attention_units_per_head
        self.total_units = num_heads * attention_units_per_head
        self.attention_layers: List[TemporalAttention] = [
            TemporalAttention(attention_units_per_head) for _ in range(num_heads)
        ]
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        head_outputs = [layer(inputs) for layer in self.attention_layers]
        return np.concatenate(head_outputs, axis=-1)


class AdaptiveTemporalAttention:
    """適応的時系列アテンション（短期・中期・長期の重み調整可能）。"""

    def __init__(self, attention_units: int):
        if attention_units % 3 != 0:
            raise ValueError("attention_units は 3 で割り切れる必要があります。")

        self.attention_units = attention_units
        per_segment = attention_units // 3
        self.short_term_attention = TemporalAttention(per_segment)
        self.medium_term_attention = TemporalAttention(per_segment)
        self.long_term_attention = TemporalAttention(per_segment)
        self._projection = None

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        data = np.asarray(inputs, dtype=float)
        batch, time_steps, _ = data.shape
        segment = max(time_steps // 3, 1)

        short_inputs = data[:, -segment:, :]
        mid_start = max((time_steps - segment) // 2, 0)
        mid_inputs = data[:, mid_start : mid_start + segment, :]
        long_inputs = data[:, :segment, :]

        short_attention = self.short_term_attention(short_inputs)
        mid_attention = self.medium_term_attention(mid_inputs)
        long_attention = self.long_term_attention(long_inputs)

        stacked = np.stack([short_attention, mid_attention, long_attention], axis=1)
        weights = np.mean(stacked, axis=2, keepdims=False)
        weights = np.exp(weights - weights.max(axis=1, keepdims=True))
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        combined = (
            weights[:, 0:1] * short_attention
            + weights[:, 1:2] * mid_attention
            + weights[:, 2:3] * long_attention
        )

        if self._projection is None:
            scale = 1.0 / max(combined.shape[-1], 1)
            self._projection = np.full((combined.shape[-1], self.attention_units), scale)

        return combined @ self._projection

