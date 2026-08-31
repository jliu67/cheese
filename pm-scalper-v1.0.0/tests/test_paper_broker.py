from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

from pm_scalper.book import BookStore
from pm_scalper.config import AppConfig
from pm_scalper.models import Asset, FeeSchedule, TradePrint
from pm_scalper.paper import PaperBroker


class FakeStore:
    def __init__(self) -> None:
        self.orders = []
        self.fills = []
        self.trades = []

    def record_order(self, run_id, order, force=False):  # noqa: ANN001
        self.orders.append((run_id, order.id, order.status, force))

    def record_fill(self, run_id, fill):  # noqa: ANN001
        self.fills.append(fill)

    def record_closed_trade(self, run_id, trade):  # noqa: ANN001
        self.trades.append(trade)


def make_asset(*, fees: bool = False) -> Asset:
    return Asset(
        token_id="yes-token",
        market_id="market-1",
        condition_id="0x1",
        slug="sample",
        question="Will it happen?",
        outcome="Yes",
        market_liquidity=Decimal("100000"),
        market_volume_24h=Decimal("50000"),
        end_date=None,
        fee_schedule=FeeSchedule(
            enabled=fees,
            rate=Decimal("0.07") if fees else Decimal("0"),
            exponent=Decimal("1"),
        ),
    )


def make_books(*, tick: str = "0.001") -> BookStore:
    books = BookStore()
    books.seed(
        {
            "asset_id": "yes-token",
            "timestamp": "1000000",
            "tick_size": tick,
            "min_order_size": "5",
            "bids": [
                {"price": "0.500", "size": "10000"},
                {"price": "0.499", "size": "10000"},
            ],
            "asks": [
                {"price": "0.510", "size": "10000"},
                {"price": "0.511", "size": "10000"},
            ],
        },
        1000.0,
    )
    return books


def make_broker(*, fees: bool = False) -> tuple[PaperBroker, Asset, BookStore, FakeStore]:
    config = AppConfig()
    config.risk.starting_cash_usdc = 1000
    config.risk.target_position_usdc = 100
    config.risk.max_position_pct_equity = 0.5
    config.risk.max_total_exposure_pct_equity = 0.8
    config.risk.depth_utilization = 1.0
    config.execution.queue_ahead_fraction = 0.0
    asset = make_asset(fees=fees)
    books = make_books()
    store = FakeStore()
    broker = PaperBroker(config, {asset.token_id: asset}, books, store, "run")
    return broker, asset, books, store


def test_maker_entry_and_target_close_profitably() -> None:
    broker, asset, books, store = make_broker()
    book = books.get(asset.token_id)
    assert book is not None

    order = broker.place_entry(asset, book, score=80.0, timestamp=1001.0)
    assert order is not None
    assert order.limit_price == Decimal("0.501")

    broker.on_trade(
        TradePrint(
            token_id=asset.token_id,
            price=order.limit_price,
            size=Decimal("10000"),
            side="SELL",
            timestamp=1002.0,
        )
    )
    assert asset.token_id in broker.positions
    target = next(
        item for item in broker.orders.values() if item.intent == "TARGET" and item.status == "OPEN"
    )
    assert target.limit_price == Decimal("0.507")

    broker.on_trade(
        TradePrint(
            token_id=asset.token_id,
            price=target.limit_price,
            size=Decimal("10000"),
            side="BUY",
            timestamp=1003.0,
        )
    )
    assert asset.token_id not in broker.positions
    assert broker.closed_trades == 1
    assert broker.realized_pnl > 0
    assert broker.cash > broker.starting_cash
    assert len(store.trades) == 1
    assert store.trades[0].exit_reason == "profit target"


def test_minimum_order_size_is_usdc_notional() -> None:
    broker, asset, books, _ = make_broker()
    book = books.get(asset.token_id)
    assert book is not None
    broker.config.risk.target_position_usdc = 4.99

    # Roughly ten shares is above a five-share interpretation, but the order is
    # below Polymarket's documented $5 minimum notional and must be rejected.
    assert broker.place_entry(asset, book, score=80.0, timestamp=1001.0) is None
    assert broker.entries_today == 0


def test_queue_ahead_must_be_consumed_before_fill() -> None:
    broker, asset, books, _ = make_broker()
    broker.config.execution.queue_ahead_fraction = 1.0
    book = books.get(asset.token_id)
    assert book is not None
    # Force the entry to rest at the displayed best bid, where 10,000 shares are ahead.
    broker.config.execution.improve_bid_ticks = 0
    order = broker.place_entry(asset, book, score=80.0, timestamp=1001.0)
    assert order is not None
    assert order.queue_ahead == Decimal("10000")

    broker.on_trade(
        TradePrint(asset.token_id, order.limit_price, Decimal("9999"), "SELL", 1002.0)
    )
    assert order.filled_quantity == 0
    broker.on_trade(
        TradePrint(asset.token_id, order.limit_price, Decimal("2"), "SELL", 1003.0)
    )
    assert order.filled_quantity == Decimal("1")


def test_taker_risk_exit_charges_market_fee() -> None:
    broker, asset, books, store = make_broker(fees=True)
    broker.config.execution.improve_bid_ticks = 0
    book = books.get(asset.token_id)
    assert book is not None
    order = broker.place_entry(asset, book, score=80.0, timestamp=1001.0)
    assert order is not None
    broker.on_trade(
        TradePrint(asset.token_id, order.limit_price, Decimal("10000"), "SELL", 1002.0)
    )
    assert asset.token_id in broker.positions

    # A newer book lets the risk engine execute against 0.49 bids.
    book.apply_snapshot(
        {
            "asset_id": asset.token_id,
            "timestamp": "1004000",
            "tick_size": "0.001",
            "min_order_size": "5",
            "bids": [{"price": "0.490", "size": "10000"}],
            "asks": [{"price": "0.500", "size": "10000"}],
        },
        1004.0,
    )
    assert broker.force_exit(asset.token_id, "test risk exit", 1004.0, 30.0)
    assert asset.token_id not in broker.positions
    assert broker.total_fees > 0
    assert store.trades[0].fees > 0
    assert store.trades[0].net_pnl < 0


def test_same_book_snapshot_cannot_be_reused_for_second_taker_exit() -> None:
    broker, asset, books, _ = make_broker(fees=True)
    broker.config.execution.improve_bid_ticks = 0
    book = books.get(asset.token_id)
    assert book is not None
    order = broker.place_entry(asset, book, score=80.0, timestamp=1001.0)
    assert order is not None
    broker.on_trade(
        TradePrint(asset.token_id, order.limit_price, Decimal("10000"), "SELL", 1002.0)
    )

    # Only one share is shown on the bid, so the first exit is partial.
    book.apply_snapshot(
        {
            "asset_id": asset.token_id,
            "timestamp": "1004000",
            "tick_size": "0.001",
            "min_order_size": "5",
            "bids": [{"price": "0.490", "size": "1"}],
            "asks": [{"price": "0.500", "size": "10000"}],
        },
        1004.0,
    )
    assert broker.force_exit(asset.token_id, "partial", 1004.0, 30.0)
    remaining = broker.positions[asset.token_id].quantity
    assert remaining > 0
    assert not broker.force_exit(asset.token_id, "duplicate", 1004.1, 30.0)
    assert broker.positions[asset.token_id].quantity == remaining


def test_trade_print_does_not_replenish_depth_for_second_taker_exit() -> None:
    broker, asset, books, _ = make_broker(fees=True)
    broker.config.execution.improve_bid_ticks = 0
    book = books.get(asset.token_id)
    assert book is not None
    order = broker.place_entry(asset, book, score=80.0, timestamp=1001.0)
    assert order is not None
    broker.on_trade(
        TradePrint(asset.token_id, order.limit_price, Decimal("10000"), "SELL", 1002.0)
    )

    book.apply_snapshot(
        {
            "asset_id": asset.token_id,
            "timestamp": "1004000",
            "hash": "partial-depth",
            "tick_size": "0.001",
            "min_order_size": "5",
            "bids": [{"price": "0.490", "size": "1"}],
            "asks": [{"price": "0.500", "size": "10000"}],
        },
        1004.0,
    )
    assert broker.force_exit(asset.token_id, "partial", 1004.0, 30.0)
    remaining = broker.positions[asset.token_id].quantity

    # A last-trade event updates market activity, not visible bid depth.
    books.apply_event(
        {
            "event_type": "last_trade_price",
            "asset_id": asset.token_id,
            "price": "0.490",
            "size": "5",
            "side": "SELL",
            "timestamp": "1005000",
        },
        1005.0,
    )
    assert not broker.force_exit(asset.token_id, "must wait for new depth", 1005.0, 30.0)
    assert broker.positions[asset.token_id].quantity == remaining


def test_ask_only_change_does_not_replenish_consumed_bid_depth() -> None:
    broker, asset, books, _ = make_broker(fees=True)
    broker.config.execution.improve_bid_ticks = 0
    book = books.get(asset.token_id)
    assert book is not None
    order = broker.place_entry(asset, book, score=80.0, timestamp=1001.0)
    assert order is not None
    broker.on_trade(
        TradePrint(asset.token_id, order.limit_price, Decimal("10000"), "SELL", 1002.0)
    )

    book.apply_snapshot(
        {
            "asset_id": asset.token_id,
            "timestamp": "1004000",
            "hash": "partial-bid",
            "tick_size": "0.001",
            "min_order_size": "5",
            "bids": [{"price": "0.490", "size": "1"}],
            "asks": [{"price": "0.500", "size": "10000"}],
        },
        1004.0,
    )
    assert broker.force_exit(asset.token_id, "partial", 1004.0, 30.0)
    remaining = broker.positions[asset.token_id].quantity
    consumed_bid_version = book.bid_version

    books.apply_event(
        {
            "event_type": "price_change",
            "timestamp": "1005000",
            "price_changes": [
                {
                    "asset_id": asset.token_id,
                    "side": "SELL",
                    "price": "0.500",
                    "size": "9999",
                    "hash": "ask-only-change",
                }
            ],
        },
        1005.0,
    )
    assert book.bid_version == consumed_bid_version
    assert not broker.force_exit(asset.token_id, "no phantom bid refill", 1005.0, 30.0)
    assert broker.positions[asset.token_id].quantity == remaining


def test_bid_only_change_does_not_reuse_crossed_ask_depth() -> None:
    broker, asset, books, _ = make_broker()
    book = books.get(asset.token_id)
    assert book is not None
    order = broker.place_entry(asset, book, score=80.0, timestamp=1001.0)
    assert order is not None

    book.apply_snapshot(
        {
            "asset_id": asset.token_id,
            "timestamp": "1002000",
            "hash": "crossed-ask",
            "tick_size": "0.001",
            "min_order_size": "5",
            "bids": [{"price": "0.500", "size": "10000"}],
            "asks": [{"price": order.limit_price, "size": "1"}],
        },
        1002.0,
    )
    broker.manage(1002.0, {})
    assert order.filled_quantity == Decimal("1")
    used_ask_version = book.ask_version

    books.apply_event(
        {
            "event_type": "price_change",
            "timestamp": "1003000",
            "price_changes": [
                {
                    "asset_id": asset.token_id,
                    "side": "BUY",
                    "price": "0.500",
                    "size": "9999",
                    "hash": "bid-only-change",
                }
            ],
        },
        1003.0,
    )
    assert book.ask_version == used_ask_version
    broker.manage(1003.0, {})
    assert order.filled_quantity == Decimal("1")


def test_daily_loss_halt_cancels_pending_entries() -> None:
    broker, asset, books, _ = make_broker()
    book = books.get(asset.token_id)
    assert book is not None
    order = broker.place_entry(asset, book, score=80.0, timestamp=1001.0)
    assert order is not None
    assert order.status == "OPEN"

    broker.cash = Decimal("970")
    broker.manage(1002.0, {})

    assert broker.halted
    assert broker.halt_reason == "daily loss limit reached"
    assert order.status == "CANCELLED"
    assert "risk halt" in order.reason


def test_daily_limits_reset_at_utc_midnight() -> None:
    broker, asset, books, _ = make_broker()
    book = books.get(asset.token_id)
    assert book is not None
    broker.config.risk.max_trades_per_day = 1

    order = broker.place_entry(asset, book, score=80.0, timestamp=1001.0)
    assert order is not None
    assert broker.entries_today == 1

    broker.cash = Decimal("970")
    broker.manage(1002.0, {})
    assert broker.halted
    assert broker.halt_reason == "daily loss limit reached"

    # 86,401 is the first second of the next UTC day relative to this fixture.
    broker.manage(86_401.0, {})
    assert broker.risk_day == "1970-01-02"
    assert broker.entries_today == 0
    assert broker.day_start_equity == Decimal("970")
    assert not broker.halted
    assert broker.halt_reason == ""


def test_entry_is_blocked_inside_resolution_buffer() -> None:
    broker, asset, books, _ = make_broker()
    book = books.get(asset.token_id)
    assert book is not None
    end_timestamp = 1001.0 + broker.config.universe.min_hours_to_end * 3600.0 - 1.0
    ending_asset = replace(
        asset,
        end_date=datetime.fromtimestamp(end_timestamp, tz=timezone.utc),
    )

    allowed, reason = broker.can_enter(ending_asset, book, timestamp=1001.0)

    assert not allowed
    assert reason == "market too close to resolution"


def test_open_position_is_forced_out_inside_resolution_buffer() -> None:
    broker, asset, books, store = make_broker()
    book = books.get(asset.token_id)
    assert book is not None
    end_timestamp = 1000.0 + 3.0 * 3600.0
    ending_asset = replace(
        asset,
        end_date=datetime.fromtimestamp(end_timestamp, tz=timezone.utc),
    )
    broker.assets[asset.token_id] = ending_asset

    order = broker.place_entry(ending_asset, book, score=80.0, timestamp=1001.0)
    assert order is not None
    broker.on_trade(
        TradePrint(asset.token_id, order.limit_price, Decimal("10000"), "SELL", 1002.0)
    )
    assert asset.token_id in broker.positions

    manage_at = end_timestamp - broker.config.universe.min_hours_to_end * 3600.0 + 1.0
    book.apply_snapshot(
        {
            "asset_id": asset.token_id,
            "timestamp": str(int(manage_at * 1000)),
            "hash": "resolution-exit",
            "tick_size": "0.001",
            "min_order_size": "5",
            "bids": [{"price": "0.500", "size": "10000"}],
            "asks": [{"price": "0.510", "size": "10000"}],
        },
        manage_at,
    )
    broker.manage(manage_at, {})

    assert asset.token_id not in broker.positions
    assert store.trades[-1].exit_reason == "market nearing resolution"



def test_market_resolution_settles_winner_and_blocks_reentry() -> None:
    broker, asset, books, store = make_broker()
    book = books.get(asset.token_id)
    assert book is not None
    order = broker.place_entry(asset, book, score=80.0, timestamp=1001.0)
    assert order is not None
    broker.on_trade(
        TradePrint(asset.token_id, order.limit_price, Decimal("10000"), "SELL", 1002.0)
    )
    assert asset.token_id in broker.positions

    broker.on_market_resolved([asset.token_id, "no-token"], asset.token_id, 1003.0)

    assert asset.token_id not in broker.positions
    assert asset.token_id in broker.resolved_tokens
    assert store.trades[-1].average_exit_price == Decimal("1")
    assert store.trades[-1].exit_reason == "market resolved"
    allowed, reason = broker.can_enter(asset, book, 1004.0)
    assert not allowed
    assert reason == "market resolved"


def test_market_resolution_settles_loser_at_zero() -> None:
    broker, asset, books, store = make_broker()
    book = books.get(asset.token_id)
    assert book is not None
    order = broker.place_entry(asset, book, score=80.0, timestamp=1001.0)
    assert order is not None
    broker.on_trade(
        TradePrint(asset.token_id, order.limit_price, Decimal("10000"), "SELL", 1002.0)
    )

    broker.on_market_resolved([asset.token_id, "no-token"], "no-token", 1003.0)

    assert asset.token_id not in broker.positions
    assert store.trades[-1].average_exit_price == Decimal("0")
    assert store.trades[-1].net_pnl < 0
    assert broker.cash < broker.starting_cash


def test_tick_size_change_cancels_invalid_resting_entry() -> None:
    broker, asset, books, _ = make_broker()
    book = books.get(asset.token_id)
    assert book is not None
    order = broker.place_entry(asset, book, score=80.0, timestamp=1001.0)
    assert order is not None
    assert order.limit_price == Decimal("0.501")

    book.apply_snapshot(
        {
            "asset_id": asset.token_id,
            "timestamp": "1002000",
            "hash": "coarser-tick",
            "tick_size": "0.01",
            "min_order_size": "5",
            "bids": [{"price": "0.50", "size": "10000"}],
            "asks": [{"price": "0.51", "size": "10000"}],
        },
        1002.0,
    )
    broker.manage(1002.0, {})

    assert order.status == "CANCELLED"
    assert order.reason == "entry price invalid after tick-size change"
    assert asset.token_id not in broker.positions


def test_updated_minimum_notional_cancels_untouched_entry() -> None:
    broker, asset, books, _ = make_broker()
    book = books.get(asset.token_id)
    assert book is not None
    order = broker.place_entry(asset, book, score=80.0, timestamp=1001.0)
    assert order is not None
    assert order.filled_quantity == 0

    book.apply_snapshot(
        {
            "asset_id": asset.token_id,
            "timestamp": "1002000",
            "hash": "higher-minimum",
            "tick_size": "0.001",
            "min_order_size": "101",
            "bids": [{"price": "0.500", "size": "10000"}],
            "asks": [{"price": "0.510", "size": "10000"}],
        },
        1002.0,
    )
    broker.manage(1002.0, {})

    assert order.status == "CANCELLED"
    assert order.reason == "entry below updated minimum order notional"
    assert asset.token_id not in broker.positions
