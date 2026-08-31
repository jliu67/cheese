from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, AsyncIterator

import httpx
import websockets

from .config import AppConfig
from .models import Asset, FeeSchedule, Market
from .util import as_bool, dec, opt_dec, parse_jsonish, parse_timestamp

LOGGER = logging.getLogger(__name__)


class GammaClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.api.gamma_url.rstrip("/"),
            timeout=config.api.request_timeout_seconds,
            headers={"User-Agent": "pm-scalper/1.0 paper-research"},
        )

    async def __aenter__(self) -> "GammaClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    async def discover_markets(self) -> list[Market]:
        # The full /markets endpoint exposes liquidity, volume, token IDs, order
        # settings and fee metadata. The keyset endpoint is a useful fallback, but
        # its compact records may omit the trading fields required by the scanner.
        try:
            raw_markets = await self._discover_legacy()
        except (httpx.HTTPError, ValueError) as exc:
            LOGGER.warning("Gamma full-market discovery failed (%s); trying keyset", exc)
            raw_markets = []
        if not raw_markets:
            raw_markets = await self._discover_keyset()

        parsed = [parse_market(item) for item in raw_markets]
        parsed = [market for market in parsed if market.id and market.condition_id]
        parsed.sort(key=market_discovery_score, reverse=True)

        if self.config.universe.hydrate_market_details:
            # Hydrate compact keyset records before applying liquidity/token filters.
            # Full-list records are hydrated only when essential details are absent.
            pool_limit = max(self.config.universe.max_markets * 4, 40)
            hydration_pool = parsed[:pool_limit]
            needs = [market for market in hydration_pool if needs_hydration(market)]
            if needs:
                hydrated = await self._hydrate(needs)
                by_id = {market.id: market for market in hydrated}
                parsed = [by_id.get(market.id, market) for market in parsed]

        candidates = [market for market in parsed if self._market_level_eligible(market)]
        candidates.sort(key=market_discovery_score, reverse=True)
        return candidates[: self.config.universe.max_markets]

    async def _discover_keyset(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(self.config.api.discovery_pages):
            params: dict[str, Any] = {
                "closed": "false",
                "limit": self.config.api.discovery_page_size,
            }
            if cursor:
                params["after_cursor"] = cursor
            response = await self._client.get("/markets/keyset", params=params)
            response.raise_for_status()
            body = response.json()
            if isinstance(body, list):
                page = body
                cursor = None
            elif isinstance(body, dict):
                page = body.get("markets") or body.get("items") or []
                cursor = body.get("next_cursor") or body.get("nextCursor")
            else:
                page = []
                cursor = None
            results.extend(item for item in page if isinstance(item, dict))
            if not cursor or not page:
                break
        return deduplicate_raw_markets(results)

    async def _discover_legacy(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for page_index in range(self.config.api.discovery_pages):
            params = {
                "closed": "false",
                "limit": self.config.api.discovery_page_size,
                "offset": page_index * self.config.api.discovery_page_size,
                "order": "volume24hr",
                "ascending": "false",
                "liquidity_num_min": self.config.universe.min_liquidity_usdc,
                "volume_num_min": self.config.universe.min_volume_24h_usdc,
            }
            response = await self._client.get("/markets", params=params)
            response.raise_for_status()
            body = response.json()
            page = body if isinstance(body, list) else body.get("markets", [])
            page = [item for item in page if isinstance(item, dict)]
            results.extend(page)
            if len(page) < self.config.api.discovery_page_size:
                break
        return deduplicate_raw_markets(results)

    async def _hydrate(self, markets: list[Market]) -> list[Market]:
        semaphore = asyncio.Semaphore(8)

        async def fetch_one(market: Market) -> Market:
            async with semaphore:
                try:
                    response = await self._client.get(f"/markets/{market.id}")
                    response.raise_for_status()
                    body = response.json()
                    return parse_market(body) if isinstance(body, dict) else market
                except (httpx.HTTPError, ValueError) as exc:
                    LOGGER.debug("Could not hydrate market %s: %s", market.id, exc)
                    return market

        return list(await asyncio.gather(*(fetch_one(market) for market in markets)))

    def _market_level_eligible(self, market: Market) -> bool:
        universe = self.config.universe
        if market.closed or not market.active or not market.accepting_orders or not market.enable_order_book:
            return False
        if len(market.token_ids) < 2 or len(market.outcomes) < 2:
            return False
        if float(market.liquidity) < universe.min_liquidity_usdc:
            return False
        if float(market.volume_24h) < universe.min_volume_24h_usdc:
            return False
        if market.seconds_delay > universe.max_seconds_delay:
            return False
        if market.end_date is not None:
            hours_left = (market.end_date - datetime.now(timezone.utc)).total_seconds() / 3600.0
            if hours_left < universe.min_hours_to_end:
                return False
        return True


class ClobDataClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.api.clob_url.rstrip("/"),
            timeout=config.api.request_timeout_seconds,
            headers={"User-Agent": "pm-scalper/1.0 paper-research"},
        )

    async def __aenter__(self) -> "ClobDataClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    async def fetch_books(self, token_ids: list[str]) -> list[dict[str, Any]]:
        if not token_ids:
            return []
        payload = [{"token_id": token_id} for token_id in token_ids]
        response = await self._client.post("/books", json=payload)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, list):
            raise ValueError("Unexpected CLOB /books response")
        return [item for item in body if isinstance(item, dict)]


class MarketWebSocket:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    async def stream(
        self,
        token_ids: list[str],
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        if not token_ids:
            raise ValueError("At least one token ID is required")
        backoff = self.config.api.reconnect_min_seconds
        while stop_event is None or not stop_event.is_set():
            try:
                async with websockets.connect(
                    self.config.api.websocket_url,
                    ping_interval=None,
                    close_timeout=5,
                    open_timeout=self.config.api.request_timeout_seconds,
                    max_size=None,
                ) as websocket:
                    subscription = {
                        "assets_ids": token_ids,
                        "type": "market",
                        "custom_feature_enabled": True,
                    }
                    await websocket.send(json.dumps(subscription))
                    LOGGER.info("Subscribed to %d Polymarket outcome tokens", len(token_ids))
                    backoff = self.config.api.reconnect_min_seconds
                    heartbeat = asyncio.create_task(self._heartbeat(websocket))
                    try:
                        async for message in websocket:
                            if stop_event is not None and stop_event.is_set():
                                break
                            if message in {"PONG", "PING", ""}:
                                continue
                            try:
                                decoded = json.loads(message)
                            except json.JSONDecodeError:
                                LOGGER.debug("Ignoring non-JSON WebSocket frame: %r", message[:200])
                                continue
                            values = decoded if isinstance(decoded, list) else [decoded]
                            for value in values:
                                if isinstance(value, dict):
                                    normalized = normalize_ws_event(value)
                                    if normalized is not None:
                                        yield normalized
                    finally:
                        heartbeat.cancel()
                        await asyncio.gather(heartbeat, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # WebSocket libraries expose several version-specific exceptions.
                if stop_event is not None and stop_event.is_set():
                    break
                LOGGER.warning("Market WebSocket disconnected: %s; reconnecting", exc)
                await asyncio.sleep(backoff)
                backoff = min(self.config.api.reconnect_max_seconds, max(backoff * 2, 1.0))

    async def _heartbeat(self, websocket: Any) -> None:
        while True:
            await asyncio.sleep(self.config.api.heartbeat_seconds)
            await websocket.send("PING")


def deduplicate_raw_markets(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in items:
        market_id = str(item.get("id") or item.get("marketId") or "")
        if market_id:
            output[market_id] = item
    return list(output.values())


def parse_market(raw: dict[str, Any]) -> Market:
    outcomes_value = parse_jsonish(raw.get("outcomes"), [])
    token_value = parse_jsonish(raw.get("clobTokenIds") or raw.get("clob_token_ids"), [])

    # Be permissive toward the normalized SDK-shaped outcome object.
    if isinstance(outcomes_value, dict):
        ordered_outcomes: list[str] = []
        ordered_tokens: list[str] = []
        for key, label in (("yes", "Yes"), ("no", "No")):
            entry = outcomes_value.get(key) or {}
            token = entry.get("tokenId") or entry.get("token_id") if isinstance(entry, dict) else None
            if token:
                ordered_outcomes.append(label)
                ordered_tokens.append(str(token))
        outcomes = ordered_outcomes
        token_ids = ordered_tokens
    else:
        outcomes = [str(value) for value in outcomes_value] if isinstance(outcomes_value, list) else []
        token_ids = [str(value) for value in token_value] if isinstance(token_value, list) else []

    fee_raw = raw.get("feeSchedule") or raw.get("fee_schedule") or {}
    fee_raw = parse_jsonish(fee_raw, {})
    if not isinstance(fee_raw, dict):
        fee_raw = {}
    fee_rate = dec(fee_raw.get("rate"), Decimal("0"))
    fees_enabled = as_bool(raw.get("feesEnabled"), fee_rate > 0)
    fee_schedule = FeeSchedule(
        enabled=fees_enabled,
        rate=fee_rate,
        exponent=dec(fee_raw.get("exponent"), Decimal("1")),
        taker_only=as_bool(fee_raw.get("takerOnly"), True),
        rebate_rate=dec(fee_raw.get("rebateRate"), Decimal("0")),
    )

    active = as_bool(raw.get("active"), True)
    closed = as_bool(raw.get("closed"), False)
    accepting_orders = as_bool(raw.get("acceptingOrders"), active and not closed)
    enable_order_book = as_bool(raw.get("enableOrderBook"), True)
    condition_id = str(raw.get("conditionId") or raw.get("condition_id") or "")
    end_date = parse_timestamp(raw.get("endDateIso") or raw.get("endDate") or raw.get("end_date"))

    return Market(
        id=str(raw.get("id") or raw.get("marketId") or ""),
        condition_id=condition_id,
        slug=str(raw.get("slug") or ""),
        question=str(raw.get("question") or raw.get("title") or "Untitled market"),
        outcomes=outcomes,
        token_ids=token_ids,
        active=active,
        closed=closed,
        accepting_orders=accepting_orders,
        enable_order_book=enable_order_book,
        liquidity=dec(raw.get("liquidity") or raw.get("liquidityNum")),
        volume_24h=dec(raw.get("volume24hr") or raw.get("volume24h") or raw.get("volume_24h")),
        best_bid=opt_dec(raw.get("bestBid") or raw.get("best_bid")),
        best_ask=opt_dec(raw.get("bestAsk") or raw.get("best_ask")),
        spread=opt_dec(raw.get("spread")),
        last_trade_price=opt_dec(raw.get("lastTradePrice") or raw.get("last_trade_price")),
        one_day_price_change=opt_dec(raw.get("oneDayPriceChange") or raw.get("one_day_price_change")),
        end_date=end_date,
        seconds_delay=int(dec(raw.get("secondsDelay") or raw.get("seconds_delay"))),
        minimum_tick_size=opt_dec(raw.get("orderPriceMinTickSize") or raw.get("minimumTickSize")),
        minimum_order_size=opt_dec(raw.get("orderMinSize") or raw.get("minimumOrderSize")),
        fee_schedule=fee_schedule,
        raw=raw,
    )


def needs_hydration(market: Market) -> bool:
    if len(market.token_ids) < 2:
        return True
    raw = market.raw
    has_order_settings = (
        "orderPriceMinTickSize" in raw
        and "orderMinSize" in raw
    )
    has_fee_metadata = "feesEnabled" in raw or "feeSchedule" in raw
    return not has_order_settings or not has_fee_metadata


def market_discovery_score(market: Market) -> float:
    liquidity = max(0.0, float(market.liquidity))
    volume = max(0.0, float(market.volume_24h))
    spread = max(0.0, float(market.spread or Decimal("0.02")))
    activity = math.log1p(liquidity) + 1.25 * math.log1p(volume)
    return activity - 12.0 * spread


def assets_from_markets(markets: list[Market], include_outcomes: list[str]) -> list[Asset]:
    include = {value.casefold() for value in include_outcomes}
    assets: list[Asset] = []
    for market in markets:
        for outcome, token_id in zip(market.outcomes, market.token_ids, strict=False):
            if outcome.casefold() not in include:
                continue
            assets.append(
                Asset(
                    token_id=token_id,
                    market_id=market.id,
                    condition_id=market.condition_id,
                    slug=market.slug,
                    question=market.question,
                    outcome=outcome,
                    market_liquidity=market.liquidity,
                    market_volume_24h=market.volume_24h,
                    end_date=market.end_date,
                    fee_schedule=market.fee_schedule,
                )
            )
    return assets


def normalize_ws_event(value: dict[str, Any]) -> dict[str, Any] | None:
    # Raw CLOB market-channel shape.
    if "event_type" in value:
        return value

    # Unified SDK stream shape, retained here so recorded/replayed data remains
    # usable if Polymarket moves the public socket to this envelope.
    event_type = value.get("type")
    payload = value.get("payload")
    if not event_type or not isinstance(payload, dict):
        return None

    normalized: dict[str, Any] = {"event_type": event_type}
    if event_type == "book":
        normalized.update(
            {
                "market": payload.get("market"),
                "asset_id": payload.get("token_id") or payload.get("tokenId"),
                "timestamp": payload.get("timestamp"),
                "hash": payload.get("hash"),
                "bids": payload.get("bids", []),
                "asks": payload.get("asks", []),
                "min_order_size": payload.get("min_order_size") or payload.get("minOrderSize"),
                "tick_size": payload.get("tick_size") or payload.get("tickSize"),
                "neg_risk": payload.get("neg_risk") or payload.get("negRisk"),
                "last_trade_price": payload.get("last_trade_price") or payload.get("lastTradePrice"),
            }
        )
    elif event_type == "price_change":
        changes = []
        for change in payload.get("price_changes") or payload.get("priceChanges") or []:
            if not isinstance(change, dict):
                continue
            changes.append(
                {
                    "asset_id": change.get("asset_id") or change.get("token_id") or change.get("tokenId"),
                    "price": change.get("price"),
                    "size": change.get("size"),
                    "side": change.get("side"),
                    "hash": change.get("hash"),
                    "best_bid": change.get("best_bid") or change.get("bestBid"),
                    "best_ask": change.get("best_ask") or change.get("bestAsk"),
                }
            )
        normalized.update(
            {
                "market": payload.get("market"),
                "price_changes": changes,
                "timestamp": payload.get("timestamp"),
            }
        )
    elif event_type == "last_trade_price":
        normalized.update(
            {
                "market": payload.get("market"),
                "asset_id": payload.get("asset_id") or payload.get("token_id") or payload.get("tokenId"),
                "price": payload.get("price"),
                "size": payload.get("size"),
                "side": payload.get("side"),
                "fee_rate_bps": payload.get("fee_rate_bps") or payload.get("feeRateBps"),
                "timestamp": payload.get("timestamp"),
                "transaction_hash": payload.get("transaction_hash") or payload.get("transactionHash"),
            }
        )
    elif event_type == "tick_size_change":
        normalized.update(
            {
                "asset_id": payload.get("asset_id") or payload.get("token_id") or payload.get("tokenId"),
                "old_tick_size": payload.get("old_tick_size") or payload.get("oldTickSize"),
                "new_tick_size": payload.get("new_tick_size") or payload.get("newTickSize"),
                "timestamp": payload.get("timestamp"),
            }
        )
    elif event_type == "best_bid_ask":
        normalized.update(
            {
                "asset_id": payload.get("asset_id") or payload.get("token_id") or payload.get("tokenId"),
                "best_bid": payload.get("best_bid") or payload.get("bestBid"),
                "best_ask": payload.get("best_ask") or payload.get("bestAsk"),
                "timestamp": payload.get("timestamp"),
            }
        )
    else:
        normalized.update(payload)
    return normalized
