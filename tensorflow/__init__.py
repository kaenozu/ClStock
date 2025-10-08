"""テスト用の極小 TensorFlow 互換スタブ。"""

from __future__ import annotations

import sys
import types

import numpy as np

__all__ = [
    "keras",
    "nn",
    "reduce_sum",
    "reduce_mean",
    "concat",
    "stack",
    "shape",
    "squeeze",
]


class _Layer:  # pragma: no cover - 単純スタブ
    pass


class _Dense:  # pragma: no cover - 単純スタブ
    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation
        self.kernel = None
        self.bias = None

    def __call__(self, inputs):
        array = np.asarray(inputs, dtype=float)
        in_features = array.shape[-1]

        if self.kernel is None:
            scale = 1.0 / max(in_features, 1)
            self.kernel = np.full((in_features, self.units), scale, dtype=float)
            self.bias = np.zeros((self.units,), dtype=float)

        output = np.matmul(array, self.kernel) + self.bias

        if self.activation == "tanh":
            output = np.tanh(output)
        elif self.activation == "softmax":
            output = _softmax(output, axis=-1)

        return output


# keras サブモジュールを sys.modules に登録
keras = types.ModuleType("tensorflow.keras")
layers = types.ModuleType("tensorflow.keras.layers")
layers.Dense = _Dense
layers.Layer = _Layer
keras.layers = layers
sys.modules[__name__ + ".keras"] = keras
sys.modules[__name__ + ".keras.layers"] = layers


class _NNModule:  # pragma: no cover - 単純スタブ
    @staticmethod
    def tanh(x, /):
        return np.tanh(np.asarray(x, dtype=float))

    @staticmethod
    def softmax(x, /, axis=-1):
        return _softmax(x, axis=axis)


nn = _NNModule()


def _softmax(x, axis=-1):  # pragma: no cover - 単純スタブ
    array = np.asarray(x, dtype=float)
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=axis, keepdims=True)
    return exp / np.where(denom == 0, 1.0, denom)


def reduce_sum(x, axis=None):  # pragma: no cover - 単純スタブ
    return np.sum(np.asarray(x, dtype=float), axis=axis)


def reduce_mean(x, axis=None, keepdims=False):  # pragma: no cover - 単純スタブ
    return np.mean(np.asarray(x, dtype=float), axis=axis, keepdims=keepdims)


def concat(values, axis=0):  # pragma: no cover - 単純スタブ
    arrays = [np.asarray(v, dtype=float) for v in values]
    return np.concatenate(arrays, axis=axis)


def stack(values, axis=0):  # pragma: no cover - 単純スタブ
    arrays = [np.asarray(v, dtype=float) for v in values]
    return np.stack(arrays, axis=axis)


def shape(x):  # pragma: no cover - 単純スタブ
    return np.asarray(np.shape(x), dtype=int)


def squeeze(x, axis=None):  # pragma: no cover - 単純スタブ
    return np.squeeze(np.asarray(x, dtype=float), axis=axis)
