from __future__ import annotations

import math

from .config import AppConfig
from .models import FeatureSnapshot, SignalSnapshot
from .util import bounded_tanh, clamp


class SignalEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def evaluate(self, features: FeatureSnapshot) -> SignalSnapshot:
        strategy = self.config.strategy
        universe = self.config.universe
        weights = strategy.weights

        momentum = (
            0.45 * bounded_tanh(features.momentum_15s, strategy.momentum_15s_scale)
            + 0.35 * bounded_tanh(features.momentum_60s, strategy.momentum_60s_scale)
            + 0.20 * bounded_tanh(features.momentum_300s, strategy.momentum_300s_scale)
        )
        book_imbalance = clamp((features.book_imbalance - 0.5) * 2.0, -1.0, 1.0)
        trade_flow = clamp(features.trade_flow, -1.0, 1.0)
        volume_acceleration = bounded_tanh(math.log(max(1e-6, features.volume_acceleration)), 1.0)
        microprice = bounded_tanh(features.microprice_edge, strategy.microprice_scale)
        spread_penalty = clamp(
            features.spread / max(1e-9, universe.max_spread_absolute), 0.0, 2.0
        )
        volatility_penalty = clamp(
            features.volatility_60s / max(1e-9, strategy.volatility_scale), 0.0, 2.0
        )

        components = {
            "momentum": momentum,
            "book_imbalance": book_imbalance,
            "trade_flow": trade_flow,
            "volume_acceleration": volume_acceleration,
            "microprice": microprice,
            "spread_penalty": spread_penalty,
            "volatility_penalty": volatility_penalty,
        }
        positive = (
            weights.get("momentum", 0.0) * momentum
            + weights.get("book_imbalance", 0.0) * book_imbalance
            + weights.get("trade_flow", 0.0) * trade_flow
            + weights.get("volume_acceleration", 0.0) * volume_acceleration
            + weights.get("microprice", 0.0) * microprice
        )
        penalties = (
            weights.get("spread_penalty", 0.0) * spread_penalty
            + weights.get("volatility_penalty", 0.0) * volatility_penalty
        )
        denominator = max(1e-9, sum(abs(value) for value in weights.values()))
        normalized = clamp((positive - penalties) / denominator, -1.0, 1.0)
        score = clamp(50.0 + 50.0 * normalized, 0.0, 100.0)

        eligible, reason = self._eligibility(features, score)
        return SignalSnapshot(
            token_id=features.token_id,
            timestamp=features.timestamp,
            score=score,
            eligible=eligible,
            reason=reason,
            features=features,
            components=components,
        )

    def _eligibility(self, features: FeatureSnapshot, score: float) -> tuple[bool, str]:
        universe = self.config.universe
        strategy = self.config.strategy
        if not features.warmed_up:
            return False, "warming up"
        if not (universe.min_price <= features.best_ask <= universe.max_price):
            return False, "entry price outside configured range"
        if features.spread > universe.max_spread_absolute:
            return False, "spread too wide"
        if features.bid_depth_usdc < universe.min_top_depth_usdc:
            return False, "insufficient bid depth"
        if features.ask_depth_usdc < universe.min_top_depth_usdc:
            return False, "insufficient ask depth"
        if score < strategy.entry_score:
            return False, "score below entry threshold"
        return True, "eligible"
