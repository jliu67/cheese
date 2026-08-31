from __future__ import annotations

import json

import httpx
import pytest

from pm_scalper.config import AppConfig
from pm_scalper.polymarket import ClobDataClient, GammaClient


def full_market(market_id: str = "123") -> dict[str, object]:
    return {
        "id": market_id,
        "conditionId": f"0x{market_id}",
        "slug": f"market-{market_id}",
        "question": "Will the test pass?",
        "outcomes": json.dumps(["Yes", "No"]),
        "clobTokenIds": json.dumps([f"yes-{market_id}", f"no-{market_id}"]),
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "enableOrderBook": True,
        "liquidityNum": 100_000,
        "volume24hr": 50_000,
        "endDateIso": "2099-01-01T00:00:00Z",
        "secondsDelay": 0,
        "orderPriceMinTickSize": 0.001,
        "orderMinSize": 5,
        "feesEnabled": False,
        "feeSchedule": {
            "rate": 0,
            "exponent": 1,
            "takerOnly": True,
            "rebateRate": 0,
        },
    }


@pytest.mark.asyncio
async def test_gamma_discovery_uses_full_filtered_market_list() -> None:
    config = AppConfig()
    config.api.discovery_pages = 1
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.path == "/markets"
        return httpx.Response(200, json=[full_market()])

    client = GammaClient(config)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=config.api.gamma_url,
        transport=httpx.MockTransport(handler),
    )
    try:
        markets = await client.discover_markets()
    finally:
        await client._client.aclose()

    assert len(markets) == 1
    assert markets[0].token_ids == ["yes-123", "no-123"]
    params = seen[0].url.params
    assert params["closed"] == "false"
    assert params["order"] == "volume24hr"
    assert params["ascending"] == "false"
    assert params["liquidity_num_min"] == str(config.universe.min_liquidity_usdc)
    assert params["volume_num_min"] == str(config.universe.min_volume_24h_usdc)


@pytest.mark.asyncio
async def test_gamma_keyset_fallback_hydrates_compact_records_before_filtering() -> None:
    config = AppConfig()
    config.api.discovery_pages = 1
    paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/markets":
            return httpx.Response(503, json={"error": "temporary"})
        if request.url.path == "/markets/keyset":
            return httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "id": "123",
                            "conditionId": "0x123",
                            "slug": "market-123",
                            "question": "Will the test pass?",
                        }
                    ],
                    "next_cursor": None,
                },
            )
        if request.url.path == "/markets/123":
            return httpx.Response(200, json=full_market())
        raise AssertionError(f"Unexpected request: {request.url}")

    client = GammaClient(config)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=config.api.gamma_url,
        transport=httpx.MockTransport(handler),
    )
    try:
        markets = await client.discover_markets()
    finally:
        await client._client.aclose()

    assert [market.id for market in markets] == ["123"]
    assert paths == ["/markets", "/markets/keyset", "/markets/123"]


@pytest.mark.asyncio
async def test_clob_batch_books_request_uses_token_id_objects() -> None:
    config = AppConfig()
    captured: list[object] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=[
                {
                    "asset_id": "token-a",
                    "bids": [{"price": "0.49", "size": "10"}],
                    "asks": [{"price": "0.51", "size": "10"}],
                }
            ],
        )

    client = ClobDataClient(config)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=config.api.clob_url,
        transport=httpx.MockTransport(handler),
    )
    try:
        books = await client.fetch_books(["token-a"])
    finally:
        await client._client.aclose()

    assert captured == [[{"token_id": "token-a"}]]
    assert books[0]["asset_id"] == "token-a"
