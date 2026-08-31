from __future__ import annotations

from decimal import Decimal

from pm_scalper.book import OrderBook
from pm_scalper.config import AppConfig
from pm_scalper.features import FeatureEngine
from pm_scalper.models import FeatureSnapshot, TradePrint
from pm_scalper.signal import SignalEngine


def make_book() -> OrderBook:
    book = OrderBook(token_id="token")
    book.apply_snapshot(
        {
            "asset_id": "token",
            "tick_size": "0.001",
            "bids": [
                {"price": "0.500", "size": "5000"},
                {"price": "0.499", "size": "5000"},
            ],
            "asks": [
                {"price": "0.502", "size": "1000"},
                {"price": "0.503", "size": "1000"},
            ],
        },
        1.0,
    )
    return book


def test_feature_warmup_and_trade_flow() -> None:
    config = AppConfig()
    config.strategy.warmup_seconds = 10
    engine = FeatureEngine(config)
    book = make_book()

    for second in range(12):
        # Nudge the midpoint upward so momentum is measurable.
        if second == 6:
            book.apply_change({"side": "BUY", "price": "0.501", "size": "5000"}, 7.0)
        engine.observe_book(book, float(second + 1))

    engine.observe_trade(
        TradePrint(
            token_id="token",
            price=Decimal("0.502"),
            size=Decimal("1000"),
            side="BUY",
            timestamp=12.0,
        )
    )
    snapshot = engine.snapshot(book, 12.0)
    assert snapshot is not None
    assert snapshot.warmed_up
    assert snapshot.trade_flow > 0
    assert snapshot.book_imbalance > 0.5


def test_strong_snapshot_is_ranked_above_weak_snapshot() -> None:
    config = AppConfig()
    signal_engine = SignalEngine(config)
    common = dict(
        token_id="token",
        timestamp=1.0,
        warmed_up=True,
        mid=0.5,
        best_bid=0.499,
        best_ask=0.501,
        spread=0.002,
        spread_pct=0.004,
        bid_depth_usdc=10000,
        ask_depth_usdc=10000,
    )
    strong = FeatureSnapshot(
        **common,
        momentum_15s=0.004,
        momentum_60s=0.01,
        momentum_300s=0.02,
        book_imbalance=0.85,
        trade_flow=0.8,
        volume_acceleration=4.0,
        microprice_edge=0.004,
        volatility_60s=0.002,
    )
    weak = FeatureSnapshot(
        **common,
        momentum_15s=-0.004,
        momentum_60s=-0.01,
        momentum_300s=-0.02,
        book_imbalance=0.2,
        trade_flow=-0.8,
        volume_acceleration=0.2,
        microprice_edge=-0.004,
        volatility_60s=0.02,
    )
    strong_signal = signal_engine.evaluate(strong)
    weak_signal = signal_engine.evaluate(weak)
    assert strong_signal.score > weak_signal.score
    assert strong_signal.eligible
    assert not weak_signal.eligible
