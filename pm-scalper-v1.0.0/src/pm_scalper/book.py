from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .models import TradePrint
from .util import ONE, ZERO, as_bool, dec, epoch_seconds, opt_dec


@dataclass(slots=True)
class OrderBook:
    token_id: str
    market: str = ""
    bids: dict[Decimal, Decimal] = field(default_factory=dict)
    asks: dict[Decimal, Decimal] = field(default_factory=dict)
    tick_size: Decimal = Decimal("0.01")
    # Polymarket documents this value as minimum USDC notional per order.
    min_order_size: Decimal = Decimal("5")
    neg_risk: bool = False
    last_trade_price: Decimal | None = None
    # last_update tracks any market-channel activity. last_depth_update tracks an
    # authoritative snapshot/change for freshness checks. Side-specific versions
    # advance only when that side's aggregate depth actually changes, preventing a
    # bid-only update from replenishing asks (or vice versa) in the paper simulator.
    last_update: float = 0.0
    last_depth_update: float = 0.0
    depth_version: int = 0
    bid_version: int = 0
    ask_version: int = 0
    book_hash: str | None = None

    def apply_snapshot(self, payload: dict[str, Any], received_at: float) -> bool:
        event_time = epoch_seconds(payload.get("timestamp"), received_at)
        if self.last_depth_update > 0 and event_time < self.last_depth_update:
            return False
        new_bids = levels_to_dict(payload.get("bids"))
        new_asks = levels_to_dict(payload.get("asks"))
        new_hash_raw = payload.get("hash")
        new_hash = str(new_hash_raw) if new_hash_raw else None

        is_first = self.last_depth_update <= 0
        bids_changed = is_first or new_bids != self.bids
        asks_changed = is_first or new_asks != self.asks
        aggregate_changed = bids_changed or asks_changed
        hash_changed = new_hash is not None and new_hash != self.book_hash
        changed = is_first or hash_changed or aggregate_changed

        self.market = str(payload.get("market") or self.market)
        self.bids = new_bids
        self.asks = new_asks
        tick = opt_dec(payload.get("tick_size") or payload.get("tickSize"))
        minimum = opt_dec(payload.get("min_order_size") or payload.get("minOrderSize"))
        if tick is not None and tick > ZERO:
            self.tick_size = tick
        if minimum is not None and minimum > ZERO:
            self.min_order_size = minimum
        if payload.get("neg_risk") is not None:
            self.neg_risk = as_bool(payload.get("neg_risk"), self.neg_risk)
        last = opt_dec(payload.get("last_trade_price") or payload.get("lastTradePrice"))
        if last is not None:
            self.last_trade_price = last
        if new_hash is not None:
            self.book_hash = new_hash
        self.last_update = max(self.last_update, event_time)
        self.last_depth_update = event_time
        if changed:
            self.depth_version += 1
        if bids_changed:
            self.bid_version += 1
        if asks_changed:
            self.ask_version += 1
        return changed

    def apply_change(self, change: dict[str, Any], received_at: float) -> bool:
        if self.last_depth_update > 0 and received_at < self.last_depth_update:
            return False
        price = dec(change.get("price"))
        size = dec(change.get("size"))
        side = str(change.get("side") or "").upper()
        if price <= ZERO or price >= ONE:
            return False
        levels = self.bids if side == "BUY" else self.asks if side == "SELL" else None
        if levels is None:
            return False

        old_size = levels.get(price, ZERO)
        if size <= ZERO:
            aggregate_changed = price in levels
            levels.pop(price, None)
        else:
            aggregate_changed = old_size != size
            levels[price] = size

        new_hash_raw = change.get("hash")
        new_hash = str(new_hash_raw) if new_hash_raw else None
        hash_changed = new_hash is not None and new_hash != self.book_hash
        changed = aggregate_changed or hash_changed
        if new_hash is not None:
            self.book_hash = new_hash
        self.last_update = max(self.last_update, received_at)
        self.last_depth_update = received_at
        if changed:
            self.depth_version += 1
        if aggregate_changed:
            if side == "BUY":
                self.bid_version += 1
            else:
                self.ask_version += 1
        return changed

    @property
    def best_bid(self) -> Decimal | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return min(self.asks) if self.asks else None

    @property
    def midpoint(self) -> Decimal | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None or ask <= bid:
            return None
        return (bid + ask) / Decimal("2")

    @property
    def spread(self) -> Decimal | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    @property
    def microprice(self) -> Decimal | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        bid_size = self.bids.get(bid, ZERO)
        ask_size = self.asks.get(ask, ZERO)
        total = bid_size + ask_size
        if total <= ZERO:
            return self.midpoint
        return (ask * bid_size + bid * ask_size) / total

    def depth_notional(self, side: str, levels: int = 5) -> Decimal:
        source = self.bids if side.upper() == "BUY" else self.asks
        prices = sorted(source, reverse=side.upper() == "BUY")[:levels]
        return sum((price * source[price] for price in prices), ZERO)

    def depth_shares(self, side: str, levels: int = 5) -> Decimal:
        source = self.bids if side.upper() == "BUY" else self.asks
        prices = sorted(source, reverse=side.upper() == "BUY")[:levels]
        return sum((source[price] for price in prices), ZERO)

    def size_at(self, side: str, price: Decimal) -> Decimal:
        source = self.bids if side.upper() == "BUY" else self.asks
        return source.get(price, ZERO)

    def imbalance(self, levels: int = 5) -> Decimal:
        bid = self.depth_notional("BUY", levels)
        ask = self.depth_notional("SELL", levels)
        total = bid + ask
        return bid / total if total > ZERO else Decimal("0.5")

    def taker_sell_vwap(self, quantity: Decimal) -> tuple[Decimal | None, Decimal]:
        return self._walk(self.bids, quantity, reverse=True)

    def taker_buy_vwap(self, quantity: Decimal) -> tuple[Decimal | None, Decimal]:
        return self._walk(self.asks, quantity, reverse=False)

    @staticmethod
    def _walk(
        levels: dict[Decimal, Decimal], quantity: Decimal, *, reverse: bool
    ) -> tuple[Decimal | None, Decimal]:
        remaining = quantity
        filled = ZERO
        notional = ZERO
        for price in sorted(levels, reverse=reverse):
            available = levels[price]
            take = min(remaining, available)
            if take <= ZERO:
                continue
            notional += take * price
            filled += take
            remaining -= take
            if remaining <= ZERO:
                break
        if filled <= ZERO:
            return None, ZERO
        return notional / filled, filled


class BookStore:
    def __init__(self) -> None:
        self.books: dict[str, OrderBook] = {}

    def get(self, token_id: str) -> OrderBook | None:
        return self.books.get(token_id)

    def ensure(self, token_id: str) -> OrderBook:
        book = self.books.get(token_id)
        if book is None:
            book = OrderBook(token_id=token_id)
            self.books[token_id] = book
        return book

    def seed(self, payload: dict[str, Any], received_at: float) -> OrderBook | None:
        token_id = str(payload.get("asset_id") or payload.get("token_id") or "")
        if not token_id:
            return None
        book = self.ensure(token_id)
        book.apply_snapshot(payload, received_at)
        return book

    def apply_event(
        self, payload: dict[str, Any], received_at: float
    ) -> tuple[list[OrderBook], list[TradePrint]]:
        event_type = str(payload.get("event_type") or "")
        updated: list[OrderBook] = []
        trades: list[TradePrint] = []

        if event_type == "book":
            token_id = str(payload.get("asset_id") or "")
            if token_id:
                book = self.ensure(token_id)
                book.apply_snapshot(payload, received_at)
                updated.append(book)
        elif event_type == "price_change":
            timestamp = epoch_seconds(payload.get("timestamp"), received_at)
            for change in payload.get("price_changes") or []:
                if not isinstance(change, dict):
                    continue
                token_id = str(change.get("asset_id") or "")
                if not token_id:
                    continue
                book = self.ensure(token_id)
                book.apply_change(change, timestamp)
                updated.append(book)
        elif event_type == "last_trade_price":
            token_id = str(payload.get("asset_id") or "")
            if token_id:
                price = dec(payload.get("price"))
                size = dec(payload.get("size"))
                timestamp = epoch_seconds(payload.get("timestamp"), received_at)
                book = self.ensure(token_id)
                if price > ZERO:
                    book.last_trade_price = price
                    book.last_update = max(book.last_update, timestamp)
                    updated.append(book)
                if price > ZERO and size > ZERO:
                    trades.append(
                        TradePrint(
                            token_id=token_id,
                            price=price,
                            size=size,
                            side=str(payload.get("side") or "").upper(),
                            timestamp=timestamp,
                            transaction_hash=payload.get("transaction_hash"),
                        )
                    )
        elif event_type == "tick_size_change":
            token_id = str(payload.get("asset_id") or "")
            tick = opt_dec(payload.get("new_tick_size"))
            if token_id and tick is not None and tick > ZERO:
                book = self.ensure(token_id)
                book.tick_size = tick
                book.last_update = max(
                    book.last_update,
                    epoch_seconds(payload.get("timestamp"), received_at),
                )
                updated.append(book)
        elif event_type == "best_bid_ask":
            # This event is a top-of-book hint, not a complete level update. Adding
            # zero-sized synthetic levels would leave stale prices behind and corrupt
            # later depth calculations, so full state remains driven by book snapshots
            # and price_change events.
            token_id = str(payload.get("asset_id") or "")
            if token_id:
                book = self.ensure(token_id)
                book.last_update = max(
                    book.last_update,
                    epoch_seconds(payload.get("timestamp"), received_at),
                )
                updated.append(book)
        return unique_books(updated), trades


def levels_to_dict(value: Any) -> dict[Decimal, Decimal]:
    result: dict[Decimal, Decimal] = {}
    if not isinstance(value, list):
        return result
    for level in value:
        if not isinstance(level, dict):
            continue
        price = dec(level.get("price"))
        size = dec(level.get("size"))
        if ZERO < price < ONE and size > ZERO:
            result[price] = size
    return result


def unique_books(books: list[OrderBook]) -> list[OrderBook]:
    return list({book.token_id: book for book in books}.values())
