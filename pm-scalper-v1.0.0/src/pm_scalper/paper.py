from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from typing import Mapping

from .book import BookStore, OrderBook
from .config import AppConfig
from .models import (
    Asset,
    ClosedTrade,
    Fill,
    PaperOrder,
    Position,
    SignalSnapshot,
    TradePrint,
)
from .storage import SQLiteStore
from .util import ONE, ZERO, ceil_to_tick, floor_to_tick

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ExitAccumulator:
    token_id: str
    market_id: str
    outcome: str
    question: str
    opened_at: float
    average_entry_price: Decimal
    entry_score: float
    quantity: Decimal = ZERO
    exit_notional: Decimal = ZERO
    entry_cost: Decimal = ZERO
    entry_fees: Decimal = ZERO
    exit_fees: Decimal = ZERO
    exit_reason: str = "unknown"
    exit_score: float = 50.0


class PaperBroker:
    """Conservative event-driven paper broker.

    Maker orders are filled only by qualifying trade prints after a configurable
    amount of displayed queue is consumed, or when the order becomes crossed by
    the visible book. Risk exits walk the visible bid book and pay the market's
    taker-fee curve.
    """

    def __init__(
        self,
        config: AppConfig,
        assets: Mapping[str, Asset],
        books: BookStore,
        store: SQLiteStore,
        run_id: str,
    ) -> None:
        self.config = config
        self.assets = dict(assets)
        self.books = books
        self.store = store
        self.run_id = run_id
        self.starting_cash = Decimal(str(config.risk.starting_cash_usdc))
        self.cash = self.starting_cash
        self.orders: dict[str, PaperOrder] = {}
        self.positions: dict[str, Position] = {}
        self.exit_accumulators: dict[str, ExitAccumulator] = {}
        self.cooldown_until: dict[str, float] = {}
        self.realized_pnl = ZERO
        self.total_fees = ZERO
        self.entries_today = 0
        self.risk_day: str | None = None
        self.day_start_equity = self.starting_cash
        self.closed_trades = 0
        self.wins = 0
        self.losses = 0
        self.consecutive_losses = 0
        self.halted = False
        self.halt_reason = ""
        self.resolved_tokens: set[str] = set()
        self._last_cross_version: dict[str, int] = {}
        self._last_taker_version: dict[str, int] = {}

    @property
    def reserved_cash(self) -> Decimal:
        return sum((order.reserved_notional for order in self.orders.values()), ZERO)

    @property
    def available_cash(self) -> Decimal:
        return max(ZERO, self.cash - self.reserved_cash)

    @property
    def exposure(self) -> Decimal:
        position_cost = sum((position.entry_cost for position in self.positions.values()), ZERO)
        return position_cost + self.reserved_cash

    @property
    def unrealized_pnl(self) -> Decimal:
        total = ZERO
        for token_id, position in self.positions.items():
            book = self.books.get(token_id)
            bid = book.best_bid if book else None
            if bid is None:
                continue
            total += position.quantity * bid - position.entry_cost - position.entry_fees
        return total

    @property
    def equity(self) -> Decimal:
        marked = self.cash
        for token_id, position in self.positions.items():
            book = self.books.get(token_id)
            bid = book.best_bid if book else None
            if bid is not None:
                marked += position.quantity * bid
        return marked

    @property
    def win_rate(self) -> float:
        return self.wins / self.closed_trades if self.closed_trades else 0.0

    def can_enter(self, asset: Asset, book: OrderBook, timestamp: float) -> tuple[bool, str]:
        self._roll_risk_day(timestamp)
        if self.halted:
            return False, self.halt_reason or "risk halt"
        if asset.token_id in self.resolved_tokens:
            return False, "market resolved"
        if asset.token_id in self.positions:
            return False, "position already open"
        if self._has_open_order(asset.token_id, intent="ENTRY"):
            return False, "entry already pending"
        if self.cooldown_until.get(asset.token_id, 0.0) > timestamp:
            return False, "cooldown"
        if asset.end_date is not None:
            end_buffer = self.config.universe.min_hours_to_end * 3600.0
            if asset.end_date.timestamp() - timestamp < end_buffer:
                return False, "market too close to resolution"
        if self.entries_today >= self.config.risk.max_trades_per_day:
            return False, "maximum entries reached"
        active_slots = len(self.positions) + sum(
            1
            for order in self.orders.values()
            if order.intent == "ENTRY" and order.status in {"OPEN", "PARTIAL"}
        )
        if active_slots >= self.config.risk.max_open_positions:
            return False, "maximum open positions reached"
        if any(
            position.market_id == asset.market_id for position in self.positions.values()
        ) or any(
            order.market_id == asset.market_id
            and order.status in {"OPEN", "PARTIAL"}
            for order in self.orders.values()
        ):
            return False, "another outcome from this market is active"
        if timestamp - book.last_depth_update > self.config.strategy.stale_book_seconds:
            return False, "stale order book"
        if book.best_bid is None or book.best_ask is None or book.best_ask <= book.best_bid:
            return False, "invalid order book"
        if self.available_cash <= ZERO:
            return False, "no available cash"
        max_exposure = self.equity * Decimal(str(self.config.risk.max_total_exposure_pct_equity))
        if self.exposure >= max_exposure:
            return False, "maximum total exposure reached"
        return True, "ok"

    def place_entry(
        self,
        asset: Asset,
        book: OrderBook,
        score: float,
        timestamp: float,
    ) -> PaperOrder | None:
        allowed, reason = self.can_enter(asset, book, timestamp)
        if not allowed:
            return None
        bid, ask = book.best_bid, book.best_ask
        if bid is None or ask is None:
            return None
        tick = book.tick_size
        improved = bid + tick * self.config.execution.improve_bid_ticks
        highest_post_only = ask - tick
        limit_price = min(improved, highest_post_only) if highest_post_only > bid else bid
        limit_price = floor_to_tick(limit_price, tick)
        if limit_price <= ZERO or limit_price >= ask:
            limit_price = bid
        if limit_price <= ZERO:
            return None

        equity = self.equity
        target = Decimal(str(self.config.risk.target_position_usdc))
        per_position_cap = equity * Decimal(str(self.config.risk.max_position_pct_equity))
        total_cap = equity * Decimal(str(self.config.risk.max_total_exposure_pct_equity))
        remaining_exposure = max(ZERO, total_cap - self.exposure)
        bid_depth_cap = book.depth_notional(
            "BUY", self.config.universe.depth_levels
        ) * Decimal(str(self.config.risk.depth_utilization))
        notional = min(
            target,
            per_position_cap,
            remaining_exposure,
            self.available_cash,
            bid_depth_cap,
        )
        quantum = Decimal("1").scaleb(-self.config.execution.quantity_decimals)
        quantity = (notional / limit_price).quantize(quantum, rounding=ROUND_FLOOR)
        order_notional = quantity * limit_price
        if quantity <= ZERO or order_notional < book.min_order_size:
            return None

        queue_ahead = book.size_at("BUY", limit_price) * Decimal(
            str(self.config.execution.queue_ahead_fraction)
        )
        order = PaperOrder(
            id=uuid.uuid4().hex,
            token_id=asset.token_id,
            market_id=asset.market_id,
            outcome=asset.outcome,
            side="BUY",
            intent="ENTRY",
            limit_price=limit_price,
            quantity=quantity,
            filled_quantity=ZERO,
            average_fill_price=ZERO,
            queue_ahead=queue_ahead,
            created_at=timestamp,
            expires_at=timestamp + self.config.strategy.entry_timeout_seconds,
            status="OPEN",
            score_at_creation=score,
            reason="maker signal entry",
        )
        self.orders[order.id] = order
        self.entries_today += 1
        self.store.record_order(self.run_id, order, force=True)
        LOGGER.info(
            "PAPER ENTRY %s %s @ %s for %s shares (score %.1f)",
            asset.outcome,
            asset.question[:60],
            limit_price,
            quantity,
            score,
        )
        return order

    def on_trade(self, trade: TradePrint) -> None:
        candidates = [
            order
            for order in self.orders.values()
            if order.token_id == trade.token_id and order.status in {"OPEN", "PARTIAL"}
        ]
        for order in sorted(candidates, key=lambda item: item.created_at):
            if order.side == "BUY":
                eligible = trade.side == "SELL" and trade.price <= order.limit_price
            else:
                eligible = trade.side == "BUY" and trade.price >= order.limit_price
            if not eligible:
                continue

            available = trade.size
            if trade.price == order.limit_price and order.queue_ahead > ZERO:
                queue_before = order.queue_ahead
                order.queue_ahead = max(ZERO, order.queue_ahead - available)
                available = max(ZERO, available - queue_before)
                self.store.record_order(self.run_id, order)
            elif trade.price != order.limit_price:
                order.queue_ahead = ZERO

            fill_quantity = min(order.remaining_quantity, available)
            if fill_quantity > ZERO:
                self._apply_fill(
                    order,
                    fill_quantity,
                    order.limit_price,
                    maker=True,
                    timestamp=trade.timestamp,
                    exit_reason="profit target" if order.intent == "TARGET" else "entry",
                )

    def manage(self, timestamp: float, signals: Mapping[str, SignalSnapshot]) -> None:
        self._roll_risk_day(timestamp)
        self._expire_entry_orders(timestamp)
        self._manage_entry_orders(timestamp, signals)

        for token_id in list(self.positions):
            position = self.positions.get(token_id)
            if position is None:
                continue
            book = self.books.get(token_id)
            if book is None or book.best_bid is None:
                continue
            signal = signals.get(token_id)
            if signal is not None:
                position.last_score = signal.score
            position.peak_bid = max(position.peak_bid, book.best_bid)

            if not self._has_open_order(token_id, intent="ENTRY"):
                self._ensure_target_order(position, book, timestamp)
                if token_id not in self.positions:
                    continue

            target_order = self._target_order(token_id)
            if (
                target_order
                and book.best_bid >= target_order.limit_price
                and book.bid_version > self._last_cross_version.get(target_order.id, -1)
            ):
                self._last_cross_version[target_order.id] = book.bid_version
                crossing_quantity = sum(
                    size for price, size in book.bids.items() if price >= target_order.limit_price
                )
                fill_quantity = min(target_order.remaining_quantity, crossing_quantity)
                if fill_quantity > ZERO:
                    self._apply_fill(
                        target_order,
                        fill_quantity,
                        target_order.limit_price,
                        maker=True,
                        timestamp=timestamp,
                        exit_reason="profit target",
                    )
                    if token_id not in self.positions:
                        continue
                    position = self.positions[token_id]

            asset = self.assets.get(token_id)
            if asset is not None and asset.end_date is not None:
                end_buffer = self.config.universe.min_hours_to_end * 3600.0
                if asset.end_date.timestamp() - timestamp < end_buffer:
                    self.force_exit(
                        token_id, "market nearing resolution", timestamp, position.last_score
                    )
                    continue

            age = timestamp - position.opened_at
            executable_return = book.best_bid / position.average_entry_price - ONE
            if executable_return <= -Decimal(str(self.config.strategy.stop_loss)):
                self.force_exit(token_id, "stop loss", timestamp, position.last_score)
                continue
            if age >= self.config.strategy.maximum_hold_seconds:
                self.force_exit(token_id, "maximum holding time", timestamp, position.last_score)
                continue
            if (
                signal is not None
                and age >= self.config.strategy.minimum_hold_seconds
                and signal.score < self.config.strategy.exit_score
            ):
                self.force_exit(token_id, "signal deterioration", timestamp, signal.score)

        self._check_kill_switch(timestamp)

    def force_exit(
        self,
        token_id: str,
        reason: str,
        timestamp: float,
        exit_score: float = 50.0,
    ) -> bool:
        position = self.positions.get(token_id)
        book = self.books.get(token_id)
        asset = self.assets.get(token_id)
        if position is None or book is None or asset is None:
            return False
        if book.bid_version <= self._last_taker_version.get(token_id, -1):
            return False
        if book.best_bid is None or position.quantity * book.best_bid < book.min_order_size:
            LOGGER.warning(
                "Cannot paper-exit %s: remaining position is below minimum order notional",
                asset.label,
            )
            return False
        self._cancel_orders_for_token(token_id, f"cancelled for {reason}")
        vwap, filled_quantity = book.taker_sell_vwap(position.quantity)
        if vwap is None or filled_quantity <= ZERO:
            LOGGER.warning("Cannot paper-exit %s: no visible bid depth", asset.label)
            return False
        self._last_taker_version[token_id] = book.bid_version
        if self.config.execution.taker_slippage_buffer_ticks > 0:
            vwap -= book.tick_size * self.config.execution.taker_slippage_buffer_ticks
            vwap = max(book.tick_size, floor_to_tick(vwap, book.tick_size))

        order = PaperOrder(
            id=uuid.uuid4().hex,
            token_id=token_id,
            market_id=asset.market_id,
            outcome=asset.outcome,
            side="SELL",
            intent="RISK_EXIT",
            limit_price=vwap,
            quantity=filled_quantity,
            filled_quantity=ZERO,
            average_fill_price=ZERO,
            queue_ahead=ZERO,
            created_at=timestamp,
            expires_at=None,
            status="OPEN",
            score_at_creation=exit_score,
            reason=reason,
        )
        self.orders[order.id] = order
        self.store.record_order(self.run_id, order, force=True)
        self._apply_fill(
            order,
            filled_quantity,
            vwap,
            maker=False,
            timestamp=timestamp,
            exit_reason=reason,
        )
        return True

    def flatten_all(self, reason: str, timestamp: float) -> None:
        for token_id in list(self.positions):
            position = self.positions.get(token_id)
            self.force_exit(
                token_id,
                reason,
                timestamp,
                position.last_score if position else 50.0,
            )

    def on_market_resolved(
        self,
        asset_ids: list[str] | tuple[str, ...] | set[str],
        winning_asset_id: str | None,
        timestamp: float,
    ) -> None:
        """Cancel trading and paper-settle any open outcome positions.

        The public market channel can announce a resolved market before a periodic
        discovery refresh would remove it. Settlement at 1/0 is deliberately
        separate from normal execution: it is the contract payoff, not a claim
        that visible CLOB liquidity existed at that price.
        """

        tokens = {str(token_id) for token_id in asset_ids if str(token_id)}
        winner = str(winning_asset_id) if winning_asset_id else None
        if not tokens:
            return

        self.resolved_tokens.update(tokens)
        for token_id in tokens:
            self._cancel_orders_for_token(token_id, "market resolved")

        for token_id in list(tokens):
            position = self.positions.get(token_id)
            asset = self.assets.get(token_id)
            if position is None or asset is None:
                continue
            if winner is None:
                LOGGER.error(
                    "Resolution event for %s has no winning asset; position left open",
                    asset.label,
                )
                continue

            settlement_price = ONE if token_id == winner else ZERO
            order = PaperOrder(
                id=uuid.uuid4().hex,
                token_id=token_id,
                market_id=asset.market_id,
                outcome=asset.outcome,
                side="SELL",
                intent="SETTLEMENT",
                limit_price=settlement_price,
                quantity=position.quantity,
                filled_quantity=ZERO,
                average_fill_price=ZERO,
                queue_ahead=ZERO,
                created_at=timestamp,
                expires_at=None,
                status="OPEN",
                score_at_creation=position.last_score,
                reason="market resolved",
            )
            self.orders[order.id] = order
            self.store.record_order(self.run_id, order, force=True)
            self._apply_fill(
                order,
                position.quantity,
                settlement_price,
                maker=True,
                timestamp=timestamp,
                exit_reason="market resolved",
            )

    def shutdown(self, timestamp: float) -> None:
        for order in list(self.orders.values()):
            if order.status in {"OPEN", "PARTIAL"} and order.intent == "ENTRY":
                self._cancel_order(order, "shutdown")
        if self.config.execution.flatten_on_shutdown:
            self.flatten_all("session shutdown", timestamp)
        else:
            for position in self.positions.values():
                book = self.books.get(position.token_id)
                if book:
                    self._ensure_target_order(position, book, timestamp)

    def _expire_entry_orders(self, timestamp: float) -> None:
        for order in list(self.orders.values()):
            if (
                order.intent == "ENTRY"
                and order.status in {"OPEN", "PARTIAL"}
                and order.expires_at is not None
                and timestamp >= order.expires_at
            ):
                self._cancel_order(order, "entry timeout")
                position = self.positions.get(order.token_id)
                book = self.books.get(order.token_id)
                if position and book:
                    self._ensure_target_order(position, book, timestamp)

    def _manage_entry_orders(
        self, timestamp: float, signals: Mapping[str, SignalSnapshot]
    ) -> None:
        """Cancel stale signals and reconcile standing bids that became marketable."""
        for order in list(self.orders.values()):
            if order.intent != "ENTRY" or order.status not in {"OPEN", "PARTIAL"}:
                continue
            signal = signals.get(order.token_id)
            if signal is not None and signal.score < self.config.strategy.entry_score:
                self._cancel_order(order, "entry signal faded")
                position = self.positions.get(order.token_id)
                book = self.books.get(order.token_id)
                if position and book:
                    self._ensure_target_order(position, book, timestamp)
                continue

            book = self.books.get(order.token_id)
            if book is None:
                continue

            # A market can change its permitted tick size while an order is
            # resting.  The real venue may cancel or reject that price; the
            # paper broker cancels it rather than granting a fill at an invalid
            # price.  A newly raised minimum notional is applied to untouched
            # entries, while partially filled remainders are left alone.
            if floor_to_tick(order.limit_price, book.tick_size) != order.limit_price:
                self._cancel_order(order, "entry price invalid after tick-size change")
                position = self.positions.get(order.token_id)
                if position:
                    self._ensure_target_order(position, book, timestamp)
                continue
            if (
                order.filled_quantity == ZERO
                and order.remaining_quantity * order.limit_price < book.min_order_size
            ):
                self._cancel_order(order, "entry below updated minimum order notional")
                continue

            if book.best_ask is None:
                continue
            if book.best_ask <= order.limit_price:
                if book.ask_version <= self._last_cross_version.get(order.id, -1):
                    continue
                self._last_cross_version[order.id] = book.ask_version
                crossed_size = sum(
                    size for price, size in book.asks.items() if price <= order.limit_price
                )
                fill_quantity = min(order.remaining_quantity, crossed_size)
                if fill_quantity > ZERO:
                    order.queue_ahead = ZERO
                    self._apply_fill(
                        order,
                        fill_quantity,
                        order.limit_price,
                        maker=True,
                        timestamp=timestamp,
                        exit_reason="entry",
                    )

    def _ensure_target_order(
        self, position: Position, book: OrderBook, timestamp: float
    ) -> PaperOrder | None:
        existing = self._target_order(position.token_id)
        target = ceil_to_tick(
            position.average_entry_price * (ONE + Decimal(str(self.config.strategy.target_return))),
            book.tick_size,
        )
        target = min(ONE - book.tick_size, target)
        position.target_price = target
        if existing is None and book.best_bid is not None and book.best_bid >= target:
            self.force_exit(
                position.token_id,
                "profit target immediately executable",
                timestamp,
                position.last_score,
            )
            return None
        if existing is not None:
            if existing.limit_price == target:
                position.target_order_id = existing.id
                return existing
            self._cancel_order(existing, "target repriced")

        if target <= ZERO or position.quantity <= ZERO:
            return None
        if position.quantity * target < book.min_order_size:
            LOGGER.warning(
                "Cannot place paper target for %s: remaining position is below minimum order notional",
                self.assets[position.token_id].label,
            )
            return None
        queue_ahead = book.size_at("SELL", target) * Decimal(
            str(self.config.execution.target_queue_ahead_fraction)
        )
        asset = self.assets[position.token_id]
        order = PaperOrder(
            id=uuid.uuid4().hex,
            token_id=position.token_id,
            market_id=position.market_id,
            outcome=position.outcome,
            side="SELL",
            intent="TARGET",
            limit_price=target,
            quantity=position.quantity,
            filled_quantity=ZERO,
            average_fill_price=ZERO,
            queue_ahead=queue_ahead,
            created_at=timestamp,
            expires_at=None,
            status="OPEN",
            score_at_creation=position.entry_score,
            reason="maker profit target",
        )
        self.orders[order.id] = order
        position.target_order_id = order.id
        self.store.record_order(self.run_id, order, force=True)
        LOGGER.info(
            "PAPER TARGET %s %s @ %s", asset.outcome, asset.question[:60], target
        )
        return order

    def _apply_fill(
        self,
        order: PaperOrder,
        quantity: Decimal,
        price: Decimal,
        *,
        maker: bool,
        timestamp: float,
        exit_reason: str,
    ) -> None:
        quantity = min(quantity, order.remaining_quantity)
        if quantity <= ZERO:
            return
        asset = self.assets[order.token_id]
        fee = self._fee(asset, quantity, price, maker)

        if order.side == "BUY":
            required = quantity * price + fee
            if required > self.cash:
                affordable = max(ZERO, (self.cash - fee) / price)
                quantum = Decimal("1").scaleb(-self.config.execution.quantity_decimals)
                quantity = affordable.quantize(quantum, rounding=ROUND_FLOOR)
                if quantity <= ZERO:
                    self._cancel_order(order, "insufficient paper cash")
                    return
                required = quantity * price + fee
            self.cash -= required
        else:
            position = self.positions.get(order.token_id)
            if position is None:
                self._cancel_order(order, "position no longer exists")
                return
            quantity = min(quantity, position.quantity)
            if quantity <= ZERO:
                return

        previous_filled = order.filled_quantity
        new_filled = previous_filled + quantity
        if new_filled > ZERO:
            order.average_fill_price = (
                order.average_fill_price * previous_filled + price * quantity
            ) / new_filled
        order.filled_quantity = new_filled
        order.status = "FILLED" if order.remaining_quantity <= ZERO else "PARTIAL"

        fill = Fill(
            id=uuid.uuid4().hex,
            order_id=order.id,
            token_id=order.token_id,
            side=order.side,
            intent=order.intent,
            quantity=quantity,
            price=price,
            fee=fee,
            maker=maker,
            timestamp=timestamp,
        )
        self.store.record_fill(self.run_id, fill)

        if order.side == "BUY":
            self._apply_buy_fill(order, asset, quantity, price, fee, timestamp)
        else:
            self._apply_sell_fill(
                order,
                asset,
                quantity,
                price,
                fee,
                timestamp,
                exit_reason,
            )
        self.total_fees += fee
        self.store.record_order(self.run_id, order, force=True)

    def _apply_buy_fill(
        self,
        order: PaperOrder,
        asset: Asset,
        quantity: Decimal,
        price: Decimal,
        fee: Decimal,
        timestamp: float,
    ) -> None:
        position = self.positions.get(order.token_id)
        if position is None:
            book = self.books.get(order.token_id)
            tick = book.tick_size if book else Decimal("0.01")
            target = ceil_to_tick(
                price * (ONE + Decimal(str(self.config.strategy.target_return))), tick
            )
            position = Position(
                token_id=order.token_id,
                market_id=asset.market_id,
                outcome=asset.outcome,
                question=asset.question,
                quantity=quantity,
                average_entry_price=price,
                entry_cost=quantity * price,
                entry_fees=fee,
                opened_at=timestamp,
                entry_score=order.score_at_creation,
                target_price=target,
                peak_bid=price,
                last_score=order.score_at_creation,
            )
            self.positions[order.token_id] = position
        else:
            old_quantity = position.quantity
            new_quantity = old_quantity + quantity
            position.average_entry_price = (
                position.average_entry_price * old_quantity + price * quantity
            ) / new_quantity
            position.quantity = new_quantity
            position.entry_cost += quantity * price
            position.entry_fees += fee

        if order.status == "FILLED":
            book = self.books.get(order.token_id)
            if book:
                self._ensure_target_order(position, book, timestamp)

    def _apply_sell_fill(
        self,
        order: PaperOrder,
        asset: Asset,
        quantity: Decimal,
        price: Decimal,
        fee: Decimal,
        timestamp: float,
        exit_reason: str,
    ) -> None:
        position = self.positions[order.token_id]
        pre_quantity = position.quantity
        ratio = quantity / pre_quantity if pre_quantity > ZERO else ONE
        allocated_cost = position.entry_cost * ratio
        allocated_entry_fees = position.entry_fees * ratio

        proceeds = quantity * price - fee
        self.cash += proceeds
        incremental_net = quantity * price - allocated_cost - allocated_entry_fees - fee
        self.realized_pnl += incremental_net

        accumulator = self.exit_accumulators.get(order.token_id)
        if accumulator is None:
            accumulator = ExitAccumulator(
                token_id=order.token_id,
                market_id=position.market_id,
                outcome=position.outcome,
                question=position.question,
                opened_at=position.opened_at,
                average_entry_price=position.average_entry_price,
                entry_score=position.entry_score,
            )
            self.exit_accumulators[order.token_id] = accumulator
        accumulator.quantity += quantity
        accumulator.exit_notional += quantity * price
        accumulator.entry_cost += allocated_cost
        accumulator.entry_fees += allocated_entry_fees
        accumulator.exit_fees += fee
        accumulator.exit_reason = exit_reason
        accumulator.exit_score = order.score_at_creation

        position.quantity -= quantity
        position.entry_cost -= allocated_cost
        position.entry_fees -= allocated_entry_fees
        if position.quantity <= Decimal("0.00000001"):
            position.quantity = ZERO
            self._finalize_trade(order.token_id, timestamp)
        elif order.intent == "TARGET" and order.status == "FILLED":
            # A fully filled target order should normally close the position. This
            # branch protects against rounding and creates a new order for leftovers.
            position.target_order_id = None
            book = self.books.get(order.token_id)
            if book:
                self._ensure_target_order(position, book, timestamp)

    def _finalize_trade(self, token_id: str, timestamp: float) -> None:
        position = self.positions.pop(token_id)
        accumulator = self.exit_accumulators.pop(token_id)
        average_exit = (
            accumulator.exit_notional / accumulator.quantity
            if accumulator.quantity > ZERO
            else ZERO
        )
        gross = accumulator.exit_notional - accumulator.entry_cost
        fees = accumulator.entry_fees + accumulator.exit_fees
        net = gross - fees
        denominator = accumulator.entry_cost + accumulator.entry_fees
        return_pct = net / denominator if denominator > ZERO else ZERO
        trade = ClosedTrade(
            id=uuid.uuid4().hex,
            token_id=token_id,
            market_id=accumulator.market_id,
            outcome=accumulator.outcome,
            question=accumulator.question,
            opened_at=accumulator.opened_at,
            closed_at=timestamp,
            quantity=accumulator.quantity,
            average_entry_price=accumulator.average_entry_price,
            average_exit_price=average_exit,
            gross_pnl=gross,
            fees=fees,
            net_pnl=net,
            return_pct=return_pct,
            exit_reason=accumulator.exit_reason,
            entry_score=accumulator.entry_score,
            exit_score=accumulator.exit_score,
        )
        self.store.record_closed_trade(self.run_id, trade)
        self.closed_trades += 1
        if net > ZERO:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1
        self.cooldown_until[token_id] = timestamp + self.config.strategy.cooldown_after_exit_seconds
        position.target_order_id = None
        LOGGER.info(
            "PAPER CLOSE %s %s | P&L %s (%s%%) | %s",
            trade.outcome,
            trade.question[:55],
            f"{trade.net_pnl:.2f}",
            f"{trade.return_pct * 100:.2f}",
            trade.exit_reason,
        )
        if self.consecutive_losses >= self.config.risk.max_consecutive_losses:
            self._halt("maximum consecutive losses reached", timestamp, flatten=False)

    def _fee(
        self, asset: Asset, quantity: Decimal, price: Decimal, maker: bool
    ) -> Decimal:
        schedule = asset.fee_schedule
        if maker or not schedule.enabled:
            return ZERO
        rate = schedule.rate
        if rate <= ZERO:
            rate = Decimal(str(self.config.execution.default_taker_fee_rate))
        exponent = schedule.exponent
        if exponent <= ZERO:
            exponent = Decimal(str(self.config.execution.default_fee_exponent))
        curve = price * (ONE - price)
        fee = quantity * rate * (curve**exponent)
        return fee.quantize(Decimal("0.00001"))

    def _check_kill_switch(self, timestamp: float) -> None:
        self._roll_risk_day(timestamp)
        floor = self.day_start_equity * (
            ONE - Decimal(str(self.config.risk.max_daily_loss_pct))
        )
        if self.equity > floor:
            return
        if not self.halted:
            LOGGER.error("PAPER KILL SWITCH: equity fell to %s", self.equity)
        self._halt(
            "daily loss limit reached",
            timestamp,
            flatten=self.config.risk.flatten_on_kill_switch,
        )

    def _roll_risk_day(self, timestamp: float) -> None:
        try:
            day = datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return
        if self.risk_day is None:
            self.risk_day = day
            self.day_start_equity = self.equity
            return
        if day == self.risk_day:
            return

        previous_day = self.risk_day
        self.risk_day = day
        self.entries_today = 0
        self.day_start_equity = self.equity
        if self.halted and self.halt_reason == "daily loss limit reached":
            self.halted = False
            self.halt_reason = ""
        LOGGER.info(
            "Rolled paper risk controls from UTC day %s to %s; equity anchor %s",
            previous_day,
            day,
            self.day_start_equity,
        )

    def _halt(self, reason: str, timestamp: float, *, flatten: bool) -> None:
        self.halted = True
        self.halt_reason = reason
        for order in list(self.orders.values()):
            if order.intent == "ENTRY" and order.status in {"OPEN", "PARTIAL"}:
                self._cancel_order(order, f"risk halt: {reason}")
        if flatten and self.positions:
            self.flatten_all(reason, timestamp)

    def _cancel_order(self, order: PaperOrder, reason: str) -> None:
        if order.status not in {"OPEN", "PARTIAL"}:
            return
        order.status = "CANCELLED"
        order.reason = reason
        self._last_cross_version.pop(order.id, None)
        self.store.record_order(self.run_id, order, force=True)
        position = self.positions.get(order.token_id)
        if position and position.target_order_id == order.id:
            position.target_order_id = None

    def _cancel_orders_for_token(self, token_id: str, reason: str) -> None:
        for order in self.orders.values():
            if order.token_id == token_id and order.status in {"OPEN", "PARTIAL"}:
                self._cancel_order(order, reason)

    def _has_open_order(self, token_id: str, *, intent: str | None = None) -> bool:
        return any(
            order.token_id == token_id
            and order.status in {"OPEN", "PARTIAL"}
            and (intent is None or order.intent == intent)
            for order in self.orders.values()
        )

    def _target_order(self, token_id: str) -> PaperOrder | None:
        for order in self.orders.values():
            if (
                order.token_id == token_id
                and order.intent == "TARGET"
                and order.status in {"OPEN", "PARTIAL"}
            ):
                return order
        return None
