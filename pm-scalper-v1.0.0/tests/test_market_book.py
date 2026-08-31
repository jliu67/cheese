from __future__ import annotations

from decimal import Decimal

from pm_scalper.book import BookStore, OrderBook
from pm_scalper.polymarket import assets_from_markets, normalize_ws_event, parse_market


def test_parse_legacy_gamma_market() -> None:
    raw = {
        "id": "123",
        "conditionId": "0xabc",
        "slug": "sample-market",
        "question": "Will the sample happen?",
        "outcomes": '["Yes","No"]',
        "clobTokenIds": '["yes-token","no-token"]',
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "enableOrderBook": True,
        "liquidityNum": 25000,
        "volume24hr": 9000,
        "orderPriceMinTickSize": 0.001,
        "orderMinSize": 5,
        "feesEnabled": True,
        "feeSchedule": {
            "rate": 0.05,
            "exponent": 1,
            "takerOnly": True,
            "rebateRate": 0.25,
        },
    }
    market = parse_market(raw)
    assert market.token_ids == ["yes-token", "no-token"]
    assert market.outcomes == ["Yes", "No"]
    assert market.liquidity == Decimal("25000")
    assert market.fee_schedule.enabled
    assert market.fee_schedule.rate == Decimal("0.05")

    assets = assets_from_markets([market], ["Yes", "No"])
    assert [asset.outcome for asset in assets] == ["Yes", "No"]


def test_parse_normalized_outcome_object() -> None:
    market = parse_market(
        {
            "id": "5",
            "condition_id": "0x5",
            "outcomes": {
                "yes": {"token_id": "a"},
                "no": {"token_id": "b"},
            },
        }
    )
    assert market.outcomes == ["Yes", "No"]
    assert market.token_ids == ["a", "b"]


def test_order_book_math_and_updates() -> None:
    book = OrderBook(token_id="token")
    book.apply_snapshot(
        {
            "asset_id": "token",
            "timestamp": "1000000000000",
            "tick_size": "0.01",
            "min_order_size": "5",
            "neg_risk": "false",
            "bids": [
                {"price": "0.49", "size": "100"},
                {"price": "0.50", "size": "40"},
            ],
            "asks": [
                {"price": "0.52", "size": "20"},
                {"price": "0.51", "size": "60"},
            ],
        },
        1_000_000_000.0,
    )
    assert book.best_bid == Decimal("0.50")
    assert book.best_ask == Decimal("0.51")
    assert book.midpoint == Decimal("0.505")
    assert book.spread == Decimal("0.01")
    assert book.neg_risk is False
    assert book.microprice == Decimal("0.504")

    vwap, filled = book.taker_sell_vwap(Decimal("80"))
    assert filled == Decimal("80")
    assert vwap == Decimal("0.495")

    book.apply_change({"side": "BUY", "price": "0.50", "size": "0"}, 1_000_000_001.0)
    assert book.best_bid == Decimal("0.49")


def test_best_bid_ask_hint_does_not_invent_depth() -> None:
    store = BookStore()
    store.seed(
        {
            "asset_id": "token",
            "bids": [{"price": "0.40", "size": "10"}],
            "asks": [{"price": "0.60", "size": "10"}],
        },
        1.0,
    )
    store.apply_event(
        {
            "event_type": "best_bid_ask",
            "asset_id": "token",
            "best_bid": "0.55",
            "best_ask": "0.56",
            "timestamp": "2000",
        },
        2.0,
    )
    book = store.get("token")
    assert book is not None
    assert book.best_bid == Decimal("0.40")
    assert book.best_ask == Decimal("0.60")


def test_unified_websocket_event_normalization() -> None:
    event = normalize_ws_event(
        {
            "type": "last_trade_price",
            "payload": {
                "tokenId": "abc",
                "price": "0.42",
                "size": "10",
                "side": "BUY",
                "timestamp": "1000",
            },
        }
    )
    assert event is not None
    assert event["event_type"] == "last_trade_price"
    assert event["asset_id"] == "abc"


def test_depth_version_changes_only_for_new_depth_state() -> None:
    store = BookStore()
    book = store.seed(
        {
            "asset_id": "token",
            "timestamp": "1000",
            "hash": "hash-1",
            "bids": [{"price": "0.49", "size": "10"}],
            "asks": [{"price": "0.51", "size": "10"}],
        },
        1.0,
    )
    assert book is not None
    assert book.depth_version == 1
    assert book.bid_version == 1
    assert book.ask_version == 1

    # A periodic snapshot certifies freshness but the same hash cannot replenish
    # liquidity that a hypothetical paper order already consumed.
    book.apply_snapshot(
        {
            "asset_id": "token",
            "timestamp": "2000",
            "hash": "hash-1",
            "bids": [{"price": "0.49", "size": "10"}],
            "asks": [{"price": "0.51", "size": "10"}],
        },
        2.0,
    )
    assert book.depth_version == 1
    assert book.bid_version == 1
    assert book.ask_version == 1
    assert book.last_depth_update == 2000.0

    store.apply_event(
        {
            "event_type": "last_trade_price",
            "asset_id": "token",
            "price": "0.50",
            "size": "1",
            "side": "BUY",
            "timestamp": "3000",
        },
        3.0,
    )
    assert book.depth_version == 1
    assert book.last_update == 3000.0
    assert book.last_depth_update == 2000.0

    store.apply_event(
        {
            "event_type": "price_change",
            "timestamp": "4000",
            "price_changes": [
                {"asset_id": "token", "side": "BUY", "price": "0.49", "size": "11"}
            ],
        },
        4.0,
    )
    assert book.depth_version == 2
    assert book.bid_version == 2
    assert book.ask_version == 1
    assert book.last_depth_update == 4000.0


def test_stale_snapshot_cannot_roll_book_backward() -> None:
    store = BookStore()
    book = store.seed(
        {
            "asset_id": "token",
            "timestamp": "2000",
            "hash": "newer",
            "bids": [{"price": "0.50", "size": "20"}],
            "asks": [{"price": "0.52", "size": "20"}],
        },
        2.0,
    )
    assert book is not None
    version = book.depth_version

    changed = book.apply_snapshot(
        {
            "asset_id": "token",
            "timestamp": "1000",
            "hash": "older",
            "bids": [{"price": "0.40", "size": "5"}],
            "asks": [{"price": "0.60", "size": "5"}],
        },
        3.0,
    )

    assert not changed
    assert book.best_bid == Decimal("0.50")
    assert book.best_ask == Decimal("0.52")
    assert book.depth_version == version
    assert book.book_hash == "newer"


def test_new_depth_update_cannot_roll_activity_timestamp_backward() -> None:
    store = BookStore()
    book = store.seed(
        {
            "asset_id": "token",
            "timestamp": "1000",
            "hash": "depth-1",
            "bids": [{"price": "0.49", "size": "10"}],
            "asks": [{"price": "0.51", "size": "10"}],
        },
        1.0,
    )
    assert book is not None

    store.apply_event(
        {
            "event_type": "last_trade_price",
            "asset_id": "token",
            "price": "0.50",
            "size": "1",
            "side": "BUY",
            "timestamp": "3000",
        },
        3.0,
    )
    assert book.last_update == 3000.0

    # This is newer than the most recent depth state but older than the trade.
    changed = book.apply_snapshot(
        {
            "asset_id": "token",
            "timestamp": "2000",
            "hash": "depth-2",
            "bids": [{"price": "0.49", "size": "11"}],
            "asks": [{"price": "0.51", "size": "10"}],
        },
        4.0,
    )
    assert changed
    assert book.last_depth_update == 2000.0
    assert book.last_update == 3000.0

    store.apply_event(
        {
            "event_type": "price_change",
            "timestamp": "2500",
            "price_changes": [
                {"asset_id": "token", "side": "BUY", "price": "0.49", "size": "12"}
            ],
        },
        5.0,
    )
    assert book.last_depth_update == 2500.0
    assert book.last_update == 3000.0
