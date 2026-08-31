from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from .util import ZERO


@dataclass(slots=True, frozen=True)
class FeeSchedule:
    enabled: bool = False
    rate: Decimal = ZERO
    exponent: Decimal = Decimal("1")
    taker_only: bool = True
    rebate_rate: Decimal = ZERO


@dataclass(slots=True)
class Market:
    id: str
    condition_id: str
    slug: str
    question: str
    outcomes: list[str]
    token_ids: list[str]
    active: bool
    closed: bool
    accepting_orders: bool
    enable_order_book: bool
    liquidity: Decimal
    volume_24h: Decimal
    best_bid: Decimal | None
    best_ask: Decimal | None
    spread: Decimal | None
    last_trade_price: Decimal | None
    one_day_price_change: Decimal | None
    end_date: datetime | None
    seconds_delay: int
    minimum_tick_size: Decimal | None
    minimum_order_size: Decimal | None
    fee_schedule: FeeSchedule
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(slots=True, frozen=True)
class Asset:
    token_id: str
    market_id: str
    condition_id: str
    slug: str
    question: str
    outcome: str
    market_liquidity: Decimal
    market_volume_24h: Decimal
    end_date: datetime | None
    fee_schedule: FeeSchedule

    @property
    def label(self) -> str:
        return f"{self.outcome}: {self.question}"


@dataclass(slots=True, frozen=True)
class TradePrint:
    token_id: str
    price: Decimal
    size: Decimal
    side: str
    timestamp: float
    transaction_hash: str | None = None

    @property
    def notional(self) -> Decimal:
        return self.price * self.size


@dataclass(slots=True)
class FeatureSnapshot:
    token_id: str
    timestamp: float
    warmed_up: bool
    mid: float
    best_bid: float
    best_ask: float
    spread: float
    spread_pct: float
    momentum_15s: float
    momentum_60s: float
    momentum_300s: float
    book_imbalance: float
    trade_flow: float
    volume_acceleration: float
    microprice_edge: float
    volatility_60s: float
    bid_depth_usdc: float
    ask_depth_usdc: float

    def as_dict(self) -> dict[str, float | bool | str]:
        return {
            "token_id": self.token_id,
            "timestamp": self.timestamp,
            "warmed_up": self.warmed_up,
            "mid": self.mid,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
            "momentum_15s": self.momentum_15s,
            "momentum_60s": self.momentum_60s,
            "momentum_300s": self.momentum_300s,
            "book_imbalance": self.book_imbalance,
            "trade_flow": self.trade_flow,
            "volume_acceleration": self.volume_acceleration,
            "microprice_edge": self.microprice_edge,
            "volatility_60s": self.volatility_60s,
            "bid_depth_usdc": self.bid_depth_usdc,
            "ask_depth_usdc": self.ask_depth_usdc,
        }


@dataclass(slots=True)
class SignalSnapshot:
    token_id: str
    timestamp: float
    score: float
    eligible: bool
    reason: str
    features: FeatureSnapshot
    components: dict[str, float]


@dataclass(slots=True)
class PaperOrder:
    id: str
    token_id: str
    market_id: str
    outcome: str
    side: str
    intent: str
    limit_price: Decimal
    quantity: Decimal
    filled_quantity: Decimal
    average_fill_price: Decimal
    queue_ahead: Decimal
    created_at: float
    expires_at: float | None
    status: str
    score_at_creation: float
    reason: str

    @property
    def remaining_quantity(self) -> Decimal:
        return max(ZERO, self.quantity - self.filled_quantity)

    @property
    def reserved_notional(self) -> Decimal:
        if self.side != "BUY" or self.status not in {"OPEN", "PARTIAL"}:
            return ZERO
        return self.remaining_quantity * self.limit_price


@dataclass(slots=True)
class Position:
    token_id: str
    market_id: str
    outcome: str
    question: str
    quantity: Decimal
    average_entry_price: Decimal
    entry_cost: Decimal
    entry_fees: Decimal
    opened_at: float
    entry_score: float
    target_price: Decimal
    target_order_id: str | None = None
    peak_bid: Decimal = ZERO
    last_score: float = 50.0


@dataclass(slots=True)
class Fill:
    id: str
    order_id: str
    token_id: str
    side: str
    intent: str
    quantity: Decimal
    price: Decimal
    fee: Decimal
    maker: bool
    timestamp: float


@dataclass(slots=True)
class ClosedTrade:
    id: str
    token_id: str
    market_id: str
    outcome: str
    question: str
    opened_at: float
    closed_at: float
    quantity: Decimal
    average_entry_price: Decimal
    average_exit_price: Decimal
    gross_pnl: Decimal
    fees: Decimal
    net_pnl: Decimal
    return_pct: Decimal
    exit_reason: str
    entry_score: float
    exit_score: float
