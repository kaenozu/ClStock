"""Utility factories and helpers for advanced precision prediction systems."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:  # pragma: no cover - fallback when torch is unavailable
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]


def create_dqn_agent(logger: logging.Logger) -> Any:
    """Create the stock trading DQN agent used by precision systems.

    Falls back to a rule-based approximation when PyTorch is unavailable.
    """

    if torch is None or nn is None or optim is None:
        logger.warning("PyTorch不可 - DQN簡易版使用")
        return SimpleDQN()

    class DQNNetwork(nn.Module):  # type: ignore[misc]
        def __init__(self, state_size: int = 50, action_size: int = 3, hidden_size: int = 256):
            super().__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, action_size)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):  # type: ignore[override]
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.relu(self.fc3(x))
            return self.fc4(x)

    class StockTradingDQN:
        def __init__(self):
            self.state_size = 50
            self.action_size = 3  # 買い/売り/ホールド
            self.memory: deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=10000)
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.learning_rate = 0.001
            self.q_network = DQNNetwork()
            self.target_network = DQNNetwork()
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

        def act(self, state):
            if np.random.random() <= self.epsilon:
                return int(np.random.randint(self.action_size))
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return int(q_values.argmax().item())

        def replay(self, batch_size: int = 32):
            if len(self.memory) < batch_size:
                return
            batch = list(np.random.choice(len(self.memory), batch_size, replace=False))
            samples = [self.memory[idx] for idx in batch]
            states = torch.FloatTensor([e[0] for e in samples])
            actions = torch.LongTensor([e[1] for e in samples])
            rewards = torch.FloatTensor([e[2] for e in samples])
            next_states = torch.FloatTensor([e[3] for e in samples])
            dones = torch.BoolTensor([e[4] for e in samples])
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (0.95 * next_q_values * ~dones)
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        def update_target_network(self):
            self.target_network.load_state_dict(self.q_network.state_dict())

        def predict_with_dqn(self, market_state):
            state_tensor = torch.FloatTensor(market_state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                confidence = torch.softmax(q_values, dim=1).max().item()
            return {
                "action": int(q_values.argmax().item()),
                "confidence": float(confidence),
                "q_values": q_values.numpy(),
            }

    return StockTradingDQN()


class SimpleDQN:
    """Rule-based fallback when the full DQN cannot be constructed."""

    def predict_with_dqn(self, market_state):
        momentum = np.mean(market_state[-5:]) - np.mean(market_state[-10:-5])
        if momentum > 0.01:
            action = 0  # 買い
        elif momentum < -0.01:
            action = 1  # 売り
        else:
            action = 2  # ホールド
        confidence = min(abs(momentum) * 10, 1.0)
        return {"action": action, "confidence": confidence}


def create_multimodal_analyzer(logger: logging.Logger) -> Any:
    """Create the multi-modal analyzer with graceful fallbacks."""

    try:
        import io

        import matplotlib.pyplot as plt  # type: ignore
        from PIL import Image  # type: ignore

        class MultiModalAnalyzer:
            def __init__(self):
                self.cnn_features_size = 128
                self.lstm_features_size = 64

            def create_chart_image(self, price_data):
                try:
                    plt.figure(figsize=(8, 6))
                    plt.plot(price_data, linewidth=2)
                    plt.title("Price Chart")
                    plt.grid(True, alpha=0.3)
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format="png", dpi=100, bbox_inches="tight")
                    img_buffer.seek(0)
                    img = Image.open(img_buffer)
                    img_array = np.array(img)
                    plt.close()
                    return img_array
                except Exception:
                    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

            def extract_chart_features(self, chart_image):
                try:
                    if len(chart_image.shape) == 3:
                        gray = np.mean(chart_image, axis=2)
                    else:
                        gray = chart_image
                    features = [
                        np.mean(gray),
                        np.std(gray),
                        np.max(gray),
                        np.min(gray),
                        len(np.where(np.diff(gray.flatten()) > 5)[0]),
                    ]
                    features.extend([0.0] * (self.cnn_features_size - len(features)))
                    return np.array(features[: self.cnn_features_size])
                except Exception:
                    return np.zeros(self.cnn_features_size)

            def extract_numerical_features(self, time_series_data):
                try:
                    if len(time_series_data) < 10:
                        return np.zeros(self.lstm_features_size)
                    features = [
                        np.mean(time_series_data),
                        np.std(time_series_data),
                        np.max(time_series_data),
                        np.min(time_series_data),
                        np.mean(np.diff(time_series_data)),
                        np.std(np.diff(time_series_data)),
                    ]
                    for window in [5, 10, 20]:
                        if len(time_series_data) >= window:
                            ma = np.mean(time_series_data[-window:])
                            features.append(ma)
                            features.append(time_series_data[-1] - ma)
                    features.extend([0.0] * (self.lstm_features_size - len(features)))
                    return np.array(features[: self.lstm_features_size])
                except Exception:
                    return np.zeros(self.lstm_features_size)

            def fuse_features(self, chart_features, numerical_features):
                try:
                    chart_weight = 0.4
                    numerical_weight = 0.6
                    chart_norm = chart_features / (np.linalg.norm(chart_features) + 1e-8)
                    numerical_norm = numerical_features / (
                        np.linalg.norm(numerical_features) + 1e-8
                    )
                    fused = np.concatenate([
                        chart_norm * chart_weight,
                        numerical_norm * numerical_weight,
                    ])
                    return fused
                except Exception:
                    return np.zeros(self.cnn_features_size + self.lstm_features_size)

            def predict_multimodal(self, price_data, volume_data=None):
                try:
                    chart_image = self.create_chart_image(price_data)
                    chart_features = self.extract_chart_features(chart_image)
                    numerical_features = self.extract_numerical_features(price_data)
                    fused_features = self.fuse_features(chart_features, numerical_features)
                    prediction_score = np.mean(fused_features) * 100
                    confidence = min(np.std(fused_features) * 2, 1.0)
                    return {
                        "prediction_score": float(prediction_score),
                        "confidence": float(confidence),
                        "chart_features": chart_features,
                        "numerical_features": numerical_features,
                        "fused_features": fused_features,
                    }
                except Exception as exc:
                    return {
                        "prediction_score": 50.0,
                        "confidence": 0.0,
                        "error": str(exc),
                    }

        return MultiModalAnalyzer()
    except Exception as exc:
        logger.warning(f"マルチモーダル依存関係不足: {exc}")
        return SimpleMultiModal()


class SimpleMultiModal:
    """Fallback multi-modal analyzer relying on statistical heuristics."""

    def predict_multimodal(self, price_data, volume_data=None):
        trend = np.mean(price_data[-5:]) - np.mean(price_data[-10:-5])
        volatility = np.std(price_data[-20:]) if len(price_data) >= 20 else np.std(price_data)
        score = 50 + trend * 1000 + (0.1 - volatility) * 100
        confidence = min(abs(trend) * 100, 1.0)
        return {"prediction_score": max(0, min(100, score)), "confidence": confidence}


class MetaLearningOptimizerAdapter:
    """Light-weight meta learning optimiser used by advanced precision models."""

    def __init__(self):
        self.symbol_adaptations: Dict[str, Dict[str, Any]] = {}
        self.sector_patterns: Dict[str, Dict[str, Any]] = {}

    def adapt_to_symbol(self, symbol: str, historical_performance: Iterable[float]):
        try:
            history = list(historical_performance)
            if len(history) >= 10:
                avg_performance = float(np.mean(history))
                volatility = float(np.std(history))
                trend = float(np.polyfit(range(len(history)), history, 1)[0])
                adaptation = {
                    "performance_bias": avg_performance - 50,
                    "volatility_factor": volatility,
                    "trend_factor": trend,
                    "adaptation_strength": min(len(history) / 50, 1.0),
                }
                self.symbol_adaptations[symbol] = adaptation
                return adaptation
            return {"adaptation_strength": 0.0}
        except Exception as exc:
            return {"adaptation_strength": 0.0, "error": str(exc)}

    def get_sector_adaptation(self, symbol: str):
        try:
            first_digit = symbol[0] if symbol else "0"
            sector_map = {
                "1": "construction",
                "2": "foods",
                "3": "textiles",
                "4": "chemicals",
                "5": "pharmacy",
                "6": "metals",
                "7": "machinery",
                "8": "electronics",
                "9": "transport",
            }
            sector = sector_map.get(first_digit, "others")
            sector_adjustments = {
                "electronics": {"volatility_multiplier": 1.2, "trend_sensitivity": 1.1},
                "machinery": {"volatility_multiplier": 0.9, "trend_sensitivity": 1.0},
                "transport": {"volatility_multiplier": 1.1, "trend_sensitivity": 0.9},
                "others": {"volatility_multiplier": 1.0, "trend_sensitivity": 1.0},
            }
            return sector_adjustments.get(sector, sector_adjustments["others"])
        except Exception:
            return {"volatility_multiplier": 1.0, "trend_sensitivity": 1.0}

    def meta_predict(self, symbol: str, base_prediction: float):
        try:
            symbol_adaptation = self.symbol_adaptations.get(symbol, {"adaptation_strength": 0.0})
            sector_adaptation = self.get_sector_adaptation(symbol)
            adaptation_strength = symbol_adaptation.get("adaptation_strength", 0.0)
            if adaptation_strength > 0.1:
                bias = symbol_adaptation.get("performance_bias", 0)
                trend_factor = symbol_adaptation.get("trend_factor", 0)
                adjusted_prediction = base_prediction + bias * adaptation_strength
                adjusted_prediction += trend_factor * 10 * adaptation_strength
                volatility_mult = sector_adaptation.get("volatility_multiplier", 1.0)
                adjusted_prediction = 50 + (adjusted_prediction - 50) * volatility_mult
                confidence_boost = adaptation_strength * 0.1
                return {
                    "adjusted_prediction": max(0, min(100, adjusted_prediction)),
                    "confidence_boost": confidence_boost,
                    "adaptation_applied": True,
                }
            return {
                "adjusted_prediction": base_prediction,
                "confidence_boost": 0.0,
                "adaptation_applied": False,
            }
        except Exception as exc:
            return {
                "adjusted_prediction": base_prediction,
                "confidence_boost": 0.0,
                "adaptation_applied": False,
                "error": str(exc),
            }


def create_meta_learning_optimizer() -> MetaLearningOptimizerAdapter:
    return MetaLearningOptimizerAdapter()


class AdvancedEnsembleAdapter:
    def __init__(self):
        self.base_weights = {
            "trend_following": 0.35,
            "dqn": 0.20,
            "multimodal": 0.20,
            "meta": 0.15,
            "transformer": 0.10,
        }
        self.performance_history: Dict[str, Any] = {}

    def update_weights_dynamically(self, recent_performances: Dict[str, float]):
        try:
            if not recent_performances:
                return self.base_weights
            total_performance = sum(recent_performances.values())
            if total_performance > 0:
                adjusted_weights: Dict[str, float] = {}
                for model, base_weight in self.base_weights.items():
                    performance = recent_performances.get(model, base_weight)
                    adjusted_weight = base_weight * (1 + (performance - 0.5) * 0.3)
                    adjusted_weights[model] = max(0.05, min(0.6, adjusted_weight))
                total_weight = sum(adjusted_weights.values())
                if total_weight > 0:
                    adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
                    return adjusted_weights
            return self.base_weights
        except Exception:
            return self.base_weights

    def ensemble_predict(self, predictions: Dict[str, float], confidences: Dict[str, float]):
        try:
            weights = self.update_weights_dynamically(confidences)
            weighted_sum = 0.0
            total_weight = 0.0
            for model, prediction in predictions.items():
                if model in weights and prediction is not None:
                    confidence = confidences.get(model, 0.5)
                    model_weight = weights[model]
                    effective_weight = model_weight * (0.5 + confidence * 0.5)
                    weighted_sum += prediction * effective_weight
                    total_weight += effective_weight
            if total_weight > 0:
                ensemble_prediction = weighted_sum / total_weight
                ensemble_confidence = min(total_weight / sum(weights.values()), 1.0)
            else:
                ensemble_prediction = 50.0
                ensemble_confidence = 0.0
            return {
                "ensemble_prediction": ensemble_prediction,
                "ensemble_confidence": ensemble_confidence,
                "used_weights": weights,
                "total_weight": total_weight,
            }
        except Exception as exc:
            return {
                "ensemble_prediction": 50.0,
                "ensemble_confidence": 0.0,
                "error": str(exc),
            }


def create_advanced_ensemble() -> AdvancedEnsembleAdapter:
    return AdvancedEnsembleAdapter()


class MarketTransformerAdapter:
    def __init__(self):
        self.sequence_length = 60
        self.feature_dim = 10

    def create_market_features(self, price_data, volume_data=None):
        try:
            if len(price_data) < self.sequence_length:
                padded_data = np.zeros(self.sequence_length)
                padded_data[-len(price_data) :] = price_data
                price_data = padded_data
            features = []
            for i in range(len(price_data) - self.sequence_length + 1):
                window = price_data[i : i + self.sequence_length]
                feature_vector = [
                    np.mean(window),
                    np.std(window),
                    np.max(window),
                    np.min(window),
                    window[-1] - window[0],
                    np.mean(np.diff(window)),
                    len(np.where(np.diff(window) > 0)[0]) / len(window),
                ]
                feature_vector.extend([0.0] * (self.feature_dim - len(feature_vector)))
                features.append(feature_vector[: self.feature_dim])
            return np.array(features) if features else np.zeros((1, self.feature_dim))
        except Exception:
            return np.zeros((1, self.feature_dim))

    def transformer_attention(self, features):
        try:
            if len(features.shape) != 2:
                return np.mean(features)
            attention_weights = np.exp(np.sum(features, axis=1))
            attention_weights = attention_weights / np.sum(attention_weights)
            attended_features = np.average(features, axis=0, weights=attention_weights)
            return attended_features
        except Exception:
            return np.mean(features, axis=0) if len(features.shape) == 2 else np.zeros(self.feature_dim)

    def transformer_predict(self, price_data, volume_data=None):
        try:
            features = self.create_market_features(price_data, volume_data)
            attended = self.transformer_attention(features)
            prediction_score = 50 + np.sum(attended) * 5
            prediction_score = max(0, min(100, prediction_score))
            confidence = min(np.std(attended) * 0.5, 1.0)
            return {
                "prediction_score": float(prediction_score),
                "confidence": float(confidence),
                "attention_weights": attended,
            }
        except Exception as exc:
            return {
                "prediction_score": 50.0,
                "confidence": 0.0,
                "error": str(exc),
            }


def create_market_transformer() -> MarketTransformerAdapter:
    return MarketTransformerAdapter()


def create_market_state(price_data: np.ndarray, volume_data: Optional[np.ndarray], state_size: int = 50) -> np.ndarray:
    try:
        if len(price_data) < state_size:
            state = np.zeros(state_size)
            state[-len(price_data) :] = price_data[-len(price_data) :]
        else:
            recent_prices = price_data[-state_size:]
            state = (recent_prices - np.mean(recent_prices)) / (np.std(recent_prices) + 1e-8)
        return state
    except Exception:
        return np.zeros(state_size)


def convert_action_to_score(action: int) -> float:
    action_map = {0: 75.0, 1: 25.0, 2: 50.0}
    return float(action_map.get(action, 50.0))


def analyze_model_contributions(
    predictions: Dict[str, float], confidences: Dict[str, float]
) -> Dict[str, Any]:
    try:
        contributions: Dict[str, Dict[str, float]] = {}
        total_confidence = sum(confidences.values())
        for model in predictions:
            confidence = confidences.get(model, 0)
            contribution_ratio = confidence / total_confidence if total_confidence > 0 else 0
            contributions[model] = {
                "prediction": predictions[model],
                "confidence": confidence,
                "contribution_ratio": contribution_ratio,
                "weighted_impact": predictions[model] * contribution_ratio,
            }
        return contributions
    except Exception:
        return {}


def return_fallback_prediction(symbol: str, error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "final_prediction": 50.0,
        "final_confidence": 0.3,
        "target_accuracy": 87.0,
        "fallback": True,
        "error": error,
    }
