from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from .book import OrderBook
from .config import AppConfig
from .models import FeatureSnapshot, TradePrint


@dataclass(slots=True)
class AssetHistory:
    prices: Deque[tuple[float, float]] = field(default_factory=deque)
    trades: Deque[tuple[float, float, float]] = field(default_factory=deque)
    last_sample_time: float = 0.0
    last_mid: float | None = None


class FeatureEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.histories: dict[str, AssetHistory] = {}
        self.retention_seconds = max(900, config.strategy.warmup_seconds + 600)

    def observe_book(self, book: OrderBook, timestamp: float) -> None:
        midpoint = book.midpoint
        if midpoint is None:
            return
        mid = float(midpoint)
        history = self.histories.setdefault(book.token_id, AssetHistory())
        should_sample = (
            history.last_mid is None
            or abs(mid - history.last_mid) > 1e-12
            or timestamp - history.last_sample_time >= 0.5
        )
        if should_sample:
            history.prices.append((timestamp, mid))
            history.last_sample_time = timestamp
            history.last_mid = mid
        self._prune(history, timestamp)

    def observe_trade(self, trade: TradePrint) -> None:
        history = self.histories.setdefault(trade.token_id, AssetHistory())
        signed = float(trade.notional)
        if trade.side == "SELL":
            signed = -signed
        elif trade.side != "BUY":
            signed = 0.0
        history.trades.append((trade.timestamp, signed, float(trade.notional)))
        self._prune(history, trade.timestamp)

    def snapshot(self, book: OrderBook, timestamp: float) -> FeatureSnapshot | None:
        bid = book.best_bid
        ask = book.best_ask
        mid_decimal = book.midpoint
        spread_decimal = book.spread
        if bid is None or ask is None or mid_decimal is None or spread_decimal is None:
            return None
        if ask <= bid:
            return None

        history = self.histories.setdefault(book.token_id, AssetHistory())
        self.observe_book(book, timestamp)
        if not history.prices:
            return None

        mid = float(mid_decimal)
        spread = float(spread_decimal)
        momentum_15s = self._return(history, timestamp, 15, mid)
        momentum_60s = self._return(history, timestamp, 60, mid)
        momentum_300s = self._return(history, timestamp, 300, mid)
        volatility = self._volatility(history, timestamp, 60)
        trade_flow = self._trade_flow(history, timestamp, 60)
        volume_acceleration = self._volume_acceleration(history, timestamp)
        microprice = book.microprice
        microprice_edge = float((microprice - mid_decimal) / mid_decimal) if microprice else 0.0
        depth_levels = self.config.universe.depth_levels
        bid_depth = float(book.depth_notional("BUY", depth_levels))
        ask_depth = float(book.depth_notional("SELL", depth_levels))
        first_time = history.prices[0][0]
        warmed_up = timestamp - first_time >= self.config.strategy.warmup_seconds and len(history.prices) >= 8

        return FeatureSnapshot(
            token_id=book.token_id,
            timestamp=timestamp,
            warmed_up=warmed_up,
            mid=mid,
            best_bid=float(bid),
            best_ask=float(ask),
            spread=spread,
            spread_pct=spread / mid if mid > 0 else 0.0,
            momentum_15s=momentum_15s,
            momentum_60s=momentum_60s,
            momentum_300s=momentum_300s,
            book_imbalance=float(book.imbalance(depth_levels)),
            trade_flow=trade_flow,
            volume_acceleration=volume_acceleration,
            microprice_edge=microprice_edge,
            volatility_60s=volatility,
            bid_depth_usdc=bid_depth,
            ask_depth_usdc=ask_depth,
        )

    def _prune(self, history: AssetHistory, timestamp: float) -> None:
        cutoff = timestamp - self.retention_seconds
        while history.prices and history.prices[0][0] < cutoff:
            history.prices.popleft()
        while history.trades and history.trades[0][0] < cutoff:
            history.trades.popleft()

    @staticmethod
    def _return(history: AssetHistory, timestamp: float, window: int, current: float) -> float:
        target = timestamp - window
        reference: float | None = None
        for sample_time, price in reversed(history.prices):
            if sample_time <= target:
                reference = price
                break
        if reference is None or reference <= 0:
            return 0.0
        return current / reference - 1.0

    @staticmethod
    def _volatility(history: AssetHistory, timestamp: float, window: int) -> float:
        prices = [price for sample_time, price in history.prices if sample_time >= timestamp - window]
        if len(prices) < 3:
            return 0.0
        returns: list[float] = []
        previous = prices[0]
        for price in prices[1:]:
            if previous > 0 and price > 0:
                returns.append(math.log(price / previous))
            previous = price
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        return math.sqrt(max(0.0, variance)) * math.sqrt(len(returns))

    @staticmethod
    def _trade_flow(history: AssetHistory, timestamp: float, window: int) -> float:
        signed = 0.0
        total = 0.0
        cutoff = timestamp - window
        for sample_time, signed_notional, absolute_notional in history.trades:
            if sample_time >= cutoff:
                signed += signed_notional
                total += absolute_notional
        return signed / total if total > 0 else 0.0

    @staticmethod
    def _volume_acceleration(history: AssetHistory, timestamp: float) -> float:
        recent = 0.0
        baseline = 0.0
        for sample_time, _, absolute_notional in history.trades:
            age = timestamp - sample_time
            if 0 <= age <= 30:
                recent += absolute_notional
            elif 30 < age <= 300:
                baseline += absolute_notional
        expected_recent = baseline / 9.0
        if expected_recent <= 1e-9:
            return 1.0 if recent <= 1e-9 else 3.0
        return max(0.0, min(10.0, recent / expected_recent))
