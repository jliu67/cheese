from __future__ import annotations

import asyncio
import logging
import math
import signal as os_signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.table import Table

from .book import BookStore
from .config import AppConfig
from .dashboard import Dashboard
from .features import FeatureEngine
from .models import Asset, SignalSnapshot
from .paper import PaperBroker
from .polymarket import ClobDataClient, GammaClient, MarketWebSocket, assets_from_markets
from .signal import SignalEngine
from .storage import RawEventRecorder, SQLiteStore
from .util import chunks, epoch_seconds, fmt_money, json_dumps

LOGGER = logging.getLogger(__name__)
CONSOLE = Console()


@dataclass(slots=True)
class PreparedUniverse:
    assets: list[Asset]
    books: BookStore
    initial_events: list[dict[str, Any]]
    discovered_market_count: int
    rejected_asset_count: int


EventQueue = asyncio.Queue[tuple[float, dict[str, Any]]]


async def websocket_event_producer(
    config: AppConfig,
    token_ids: list[str],
    stop_event: asyncio.Event,
    queue: EventQueue,
) -> None:
    websocket = MarketWebSocket(config)
    async for event in websocket.stream(token_ids, stop_event):
        await queue.put((time.time(), event))


async def rest_book_refresh_producer(
    config: AppConfig,
    token_ids: list[str],
    stop_event: asyncio.Event,
    queue: EventQueue,
) -> None:
    interval = config.api.book_refresh_seconds
    if interval <= 0:
        return
    async with ClobDataClient(config) as clob:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                pass

            try:
                for token_chunk in chunks(token_ids, 250):
                    payloads = await clob.fetch_books(token_chunk)
                    received_at = time.time()
                    for payload in payloads:
                        event = {
                            **payload,
                            "event_type": "book",
                            "_source": "clob_periodic_refresh",
                        }
                        await queue.put((received_at, event))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.warning("Periodic CLOB book refresh failed: %s", exc)


def start_event_producers(
    config: AppConfig,
    token_ids: list[str],
    stop_event: asyncio.Event,
    queue: EventQueue,
) -> list[asyncio.Task[None]]:
    tasks = [
        asyncio.create_task(
            websocket_event_producer(config, token_ids, stop_event, queue),
            name="polymarket-market-stream",
        )
    ]
    if config.api.book_refresh_seconds > 0:
        tasks.append(
            asyncio.create_task(
                rest_book_refresh_producer(config, token_ids, stop_event, queue),
                name="polymarket-book-refresh",
            )
        )
    return tasks


def raise_if_producer_failed(tasks: list[asyncio.Task[None]], stop_event: asyncio.Event) -> None:
    if stop_event.is_set():
        return
    for task in tasks:
        if not task.done() or task.cancelled():
            continue
        exc = task.exception()
        if exc is not None:
            raise RuntimeError(f"Public-data producer {task.get_name()} failed") from exc
        raise RuntimeError(f"Public-data producer {task.get_name()} stopped unexpectedly")


async def prepare_universe(config: AppConfig) -> PreparedUniverse:
    async with GammaClient(config) as gamma:
        markets = await gamma.discover_markets()
    if not markets:
        raise RuntimeError("No eligible active markets were returned by Gamma")

    candidate_assets = assets_from_markets(markets, config.universe.include_outcomes)
    if not candidate_assets:
        raise RuntimeError("No outcome tokens matched universe.include_outcomes")

    token_ids = [asset.token_id for asset in candidate_assets]
    book_payloads: list[dict[str, Any]] = []
    async with ClobDataClient(config) as clob:
        for token_chunk in chunks(token_ids, 250):
            book_payloads.extend(await clob.fetch_books(token_chunk))

    now = time.time()
    books = BookStore()
    for payload in book_payloads:
        books.seed(payload, now)

    valid: list[tuple[float, Asset]] = []
    rejected = 0
    for asset in candidate_assets:
        book = books.get(asset.token_id)
        if book is None or book.best_bid is None or book.best_ask is None or book.spread is None:
            rejected += 1
            continue
        if book.best_ask <= book.best_bid:
            rejected += 1
            continue
        if not (
            config.universe.min_price
            <= float(book.best_ask)
            <= config.universe.max_price
        ):
            rejected += 1
            continue
        if float(book.spread) > config.universe.max_spread_absolute:
            rejected += 1
            continue
        bid_depth = float(book.depth_notional("BUY", config.universe.depth_levels))
        ask_depth = float(book.depth_notional("SELL", config.universe.depth_levels))
        if min(bid_depth, ask_depth) < config.universe.min_top_depth_usdc:
            rejected += 1
            continue
        activity = math.log1p(float(asset.market_liquidity)) + math.log1p(
            float(asset.market_volume_24h)
        )
        quality = activity + 0.0002 * min(bid_depth, ask_depth) - 20.0 * float(book.spread)
        valid.append((quality, asset))

    valid.sort(key=lambda item: item[0], reverse=True)
    assets = [asset for _, asset in valid]
    if not assets:
        raise RuntimeError(
            "Markets were found, but every outcome failed the live price/spread/depth filters. "
            "Lower the universe thresholds or retry when markets are more active."
        )

    # Keep only books that belong to selected assets.
    allowed = {asset.token_id for asset in assets}
    books.books = {token_id: book for token_id, book in books.books.items() if token_id in allowed}
    initial_events = [
        {**payload, "event_type": "book", "_source": "clob_initial"}
        for payload in book_payloads
        if str(payload.get("asset_id") or payload.get("token_id") or "") in allowed
    ]
    return PreparedUniverse(
        assets=assets,
        books=books,
        initial_events=initial_events,
        discovered_market_count=len(markets),
        rejected_asset_count=rejected,
    )


async def show_discovery(config: AppConfig) -> None:
    prepared = await prepare_universe(config)
    table = Table(title="PM-Scalper live universe", expand=True)
    table.add_column("Outcome", width=8)
    table.add_column("Bid / Ask", justify="right", width=15)
    table.add_column("Spread", justify="right", width=8)
    table.add_column("Bid depth", justify="right", width=12)
    table.add_column("24h volume", justify="right", width=13)
    table.add_column("Liquidity", justify="right", width=13)
    table.add_column("Market")
    for asset in prepared.assets:
        book = prepared.books.get(asset.token_id)
        if book is None or book.best_bid is None or book.best_ask is None or book.spread is None:
            continue
        table.add_row(
            asset.outcome,
            f"{book.best_bid:.4f} / {book.best_ask:.4f}",
            f"{book.spread:.4f}",
            fmt_money(book.depth_notional("BUY", config.universe.depth_levels)),
            fmt_money(asset.market_volume_24h),
            fmt_money(asset.market_liquidity),
            asset.question,
        )
    CONSOLE.print(table)
    CONSOLE.print(
        f"Selected [bold]{len(prepared.assets)}[/bold] outcome tokens from "
        f"{prepared.discovered_market_count} markets; "
        f"{prepared.rejected_asset_count} outcomes failed live book filters."
    )


async def run_recorder(config: AppConfig, duration_seconds: float = 0.0) -> Path:
    store = SQLiteStore(config)
    run_id = store.start_run(config, note="record-only session")
    recorder = RawEventRecorder(config, run_id)
    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)
    producer_tasks: list[asyncio.Task[None]] = []
    count = 0
    status = "COMPLETED"
    try:
        prepared = await prepare_universe(config)
        store.record_assets(run_id, prepared.assets)
        now = time.time()
        for event in prepared.initial_events:
            recorder.write(event, now)
        token_ids = [asset.token_id for asset in prepared.assets]
        queue: EventQueue = asyncio.Queue(maxsize=100_000)
        producer_tasks = start_event_producers(config, token_ids, stop_event, queue)
        started = time.monotonic()
        CONSOLE.print(
            f"Recording [bold]{len(token_ids)}[/bold] tokens to [bold]{recorder.path}[/bold]. "
            "Press Ctrl+C to stop."
        )
        while not stop_event.is_set():
            if duration_seconds > 0 and time.monotonic() - started >= duration_seconds:
                stop_event.set()
                break
            try:
                received_at, event = await asyncio.wait_for(queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                raise_if_producer_failed(producer_tasks, stop_event)
                continue
            recorder.write(event, received_at)
            count += 1
            queue.task_done()
            if count % 500 == 0:
                CONSOLE.print(f"Recorded {count:,} public-data events")
    except asyncio.CancelledError:
        status = "CANCELLED"
        raise
    except Exception as exc:
        status = "FAILED"
        store.finish_run(run_id, status, note=str(exc))
        raise
    finally:
        stop_event.set()
        for task in producer_tasks:
            task.cancel()
        if producer_tasks:
            await asyncio.gather(*producer_tasks, return_exceptions=True)
        recorder.close()
        if status != "FAILED":
            store.finish_run(run_id, status, note=f"recorded_events={count}")
        store.close()
    return recorder.path


async def run_paper(
    config: AppConfig,
    duration_seconds: float = 0.0,
    dashboard_enabled: bool = True,
) -> str:
    store = SQLiteStore(config)
    run_id = store.start_run(config, note="live-data paper-trading session")
    recorder = RawEventRecorder(config, run_id)
    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)
    producer_tasks: list[asyncio.Task[None]] = []
    live: Live | None = None
    broker: PaperBroker | None = None
    status = "COMPLETED"
    failure_note: str | None = None

    try:
        prepared = await prepare_universe(config)
        assets = {asset.token_id: asset for asset in prepared.assets}
        store.record_assets(run_id, prepared.assets)
        now = time.time()
        for event in prepared.initial_events:
            recorder.write(event, now)

        feature_engine = FeatureEngine(config)
        signal_engine = SignalEngine(config)
        broker = PaperBroker(config, assets, prepared.books, store, run_id)
        for book in prepared.books.books.values():
            feature_engine.observe_book(book, now)

        queue: EventQueue = asyncio.Queue(maxsize=100_000)
        token_ids = list(assets)
        producer_tasks = start_event_producers(config, token_ids, stop_event, queue)
        dashboard = Dashboard(assets)
        use_live = dashboard_enabled and sys.stdout.isatty()
        if use_live:
            live = Live(
                dashboard.render(broker, {}, run_id, str(recorder.path)),
                console=CONSOLE,
                refresh_per_second=4,
                screen=False,
            )
            live.start()
        else:
            CONSOLE.print(
                f"PM-Scalper paper run [bold]{run_id}[/bold] watching "
                f"[bold]{len(token_ids)}[/bold] outcome tokens. Raw data: {recorder.path}"
            )

        started_monotonic = time.monotonic()
        next_decision = time.monotonic()
        next_dashboard = time.monotonic()
        next_equity = time.monotonic()
        next_signal_log = time.monotonic()
        next_plain_status = time.monotonic() + 30
        signals: dict[str, SignalSnapshot] = {}

        while not stop_event.is_set():
            if duration_seconds > 0 and time.monotonic() - started_monotonic >= duration_seconds:
                stop_event.set()
                break

            try:
                received_at, event = await asyncio.wait_for(queue.get(), timeout=0.25)
                recorder.write(event, received_at)
                if str(event.get("event_type") or "") == "market_resolved":
                    raw_asset_ids = (
                        event.get("assets_ids")
                        or event.get("asset_ids")
                        or event.get("clob_token_ids")
                        or event.get("token_ids")
                        or []
                    )
                    if isinstance(raw_asset_ids, str):
                        raw_asset_ids = [raw_asset_ids]
                    winning_asset_id = (
                        event.get("winning_asset_id")
                        or event.get("winningAssetId")
                        or event.get("winning_token_id")
                    )
                    broker.on_market_resolved(
                        [str(token_id) for token_id in raw_asset_ids],
                        str(winning_asset_id) if winning_asset_id else None,
                        epoch_seconds(event.get("timestamp"), received_at),
                    )
                updated_books, trades = prepared.books.apply_event(event, received_at)
                for book in updated_books:
                    feature_engine.observe_book(book, book.last_update or received_at)
                for trade in trades:
                    feature_engine.observe_trade(trade)
                    broker.on_trade(trade)
                queue.task_done()
            except asyncio.TimeoutError:
                raise_if_producer_failed(producer_tasks, stop_event)

            monotonic_now = time.monotonic()
            wall_now = time.time()
            if monotonic_now >= next_decision:
                fresh_signals: dict[str, SignalSnapshot] = {}
                for token_id in assets:
                    book = prepared.books.get(token_id)
                    if book is None:
                        continue
                    features = feature_engine.snapshot(book, wall_now)
                    if features is None:
                        continue
                    fresh_signals[token_id] = signal_engine.evaluate(features)
                signals = fresh_signals
                broker.manage(wall_now, signals)

                for candidate in sorted(
                    (item for item in signals.values() if item.eligible),
                    key=lambda item: item.score,
                    reverse=True,
                ):
                    asset = assets[candidate.token_id]
                    book = prepared.books.get(candidate.token_id)
                    if book is not None and broker.place_entry(
                        asset, book, candidate.score, wall_now
                    ):
                        # One new order per decision interval prevents a burst of
                        # highly correlated entries from the same data update.
                        break
                next_decision = monotonic_now + config.strategy.decision_interval_seconds

            if monotonic_now >= next_signal_log:
                for signal_snapshot in signals.values():
                    store.record_signal(run_id, signal_snapshot)
                next_signal_log = monotonic_now + config.strategy.signal_log_seconds

            if monotonic_now >= next_equity:
                store.record_equity(
                    run_id,
                    wall_now,
                    broker.cash,
                    broker.exposure,
                    broker.unrealized_pnl,
                    broker.realized_pnl,
                    broker.equity,
                    len(broker.positions),
                )
                next_equity = monotonic_now + config.strategy.equity_sample_seconds

            if live is not None and monotonic_now >= next_dashboard:
                live.update(dashboard.render(broker, signals, run_id, str(recorder.path)))
                next_dashboard = monotonic_now + config.strategy.dashboard_interval_seconds
            elif live is None and monotonic_now >= next_plain_status:
                LOGGER.info(
                    "run=%s equity=%s cash=%s positions=%d closed=%d",
                    run_id,
                    broker.equity,
                    broker.cash,
                    len(broker.positions),
                    broker.closed_trades,
                )
                next_plain_status = monotonic_now + 30

    except asyncio.CancelledError:
        status = "CANCELLED"
        raise
    except Exception as exc:
        status = "FAILED"
        failure_note = str(exc)
        LOGGER.exception("Paper engine failed")
        raise
    finally:
        stop_event.set()
        for task in producer_tasks:
            task.cancel()
        if producer_tasks:
            await asyncio.gather(*producer_tasks, return_exceptions=True)
        if broker is not None:
            broker.shutdown(time.time())
            store.record_equity(
                run_id,
                time.time(),
                broker.cash,
                broker.exposure,
                broker.unrealized_pnl,
                broker.realized_pnl,
                broker.equity,
                len(broker.positions),
            )
        if live is not None:
            live.stop()
        recorder.close()
        store.finish_run(run_id, status, note=failure_note)
        store.close()
    return run_id


def install_signal_handlers(stop_event: asyncio.Event) -> None:
    try:
        loop = asyncio.get_running_loop()
        for signal_name in (os_signal.SIGINT, os_signal.SIGTERM):
            loop.add_signal_handler(signal_name, stop_event.set)
    except (NotImplementedError, RuntimeError):
        # Windows or a non-main thread; KeyboardInterrupt still works.
        return


def configure_logging(config: AppConfig, verbose: bool = False) -> Path:
    log_dir = Path(config.storage.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pm-scalper.log"
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)
    if not sys.stdout.isatty():
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        root.addHandler(stream_handler)
    return log_path
