from __future__ import annotations

import argparse
import asyncio
import sys
import time
from decimal import Decimal
from pathlib import Path

from rich.console import Console

from . import __version__
from .book import OrderBook
from .config import AppConfig, load_config
from .engine import configure_logging, run_paper, run_recorder, show_discovery
from .features import FeatureEngine
from .report import generate_report
from .signal import SignalEngine

CONSOLE = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pm-scalper",
        description="Polymarket public-data recorder and paper-trading research engine",
    )
    parser.add_argument("--version", action="version", version=f"pm-scalper {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--config", default="config/default.yaml", help="YAML configuration path"
        )
        command.add_argument("--verbose", action="store_true", help="Enable debug logs")

    validate = subparsers.add_parser("validate", help="Validate config and local mechanics")
    common(validate)
    validate.add_argument(
        "--online", action="store_true", help="Also test Gamma/CLOB market-data access"
    )

    discover = subparsers.add_parser("discover", help="Show the current eligible universe")
    common(discover)

    record = subparsers.add_parser("record", help="Record public market WebSocket data")
    common(record)
    record.add_argument(
        "--duration",
        default="0",
        help="Optional run duration such as 30m or 2h; 0 runs until interrupted",
    )

    paper = subparsers.add_parser("paper", help="Run the live-data paper trader")
    common(paper)
    paper.add_argument(
        "--duration",
        default="0",
        help="Optional run duration such as 30m or 2h; 0 runs until interrupted",
    )
    paper.add_argument("--no-dashboard", action="store_true", help="Disable the live terminal UI")

    report = subparsers.add_parser("report", help="Export and summarize a paper run")
    common(report)
    report.add_argument("--run-id", default=None, help="Defaults to the most recent run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        configure_logging(config, args.verbose)
        if args.command == "validate":
            validate_local(config)
            if args.online:
                asyncio.run(show_discovery(config))
            return 0
        if args.command == "discover":
            asyncio.run(show_discovery(config))
            return 0
        if args.command == "record":
            path = asyncio.run(run_recorder(config, parse_duration(args.duration)))
            CONSOLE.print(f"Recorder stopped. Data: [bold]{path}[/bold]")
            return 0
        if args.command == "paper":
            run_id = asyncio.run(
                run_paper(
                    config,
                    duration_seconds=parse_duration(args.duration),
                    dashboard_enabled=not args.no_dashboard,
                )
            )
            CONSOLE.print(
                f"Paper session finished: [bold]{run_id}[/bold]\n"
                f"Run [bold]pm-scalper report --config {args.config} --run-id {run_id}[/bold]"
            )
            return 0
        if args.command == "report":
            generate_report(config, args.run_id)
            return 0
    except KeyboardInterrupt:
        CONSOLE.print("\nStopped.")
        return 130
    except Exception as exc:
        CONSOLE.print(f"[bold red]PM-Scalper error:[/bold red] {exc}")
        return 1
    return 0


def parse_duration(value: str) -> float:
    text = str(value).strip().lower()
    if text in {"", "0", "none", "off"}:
        return 0.0
    multiplier = 1.0
    if text.endswith("s"):
        text = text[:-1]
    elif text.endswith("m"):
        text = text[:-1]
        multiplier = 60.0
    elif text.endswith("h"):
        text = text[:-1]
        multiplier = 3600.0
    elif text.endswith("d"):
        text = text[:-1]
        multiplier = 86_400.0
    result = float(text) * multiplier
    if result < 0:
        raise ValueError("Duration cannot be negative")
    return result


def validate_local(config: AppConfig) -> None:
    config.validate()
    book = OrderBook(token_id="self-test")
    now = time.time()
    book.apply_snapshot(
        {
            "asset_id": "self-test",
            "timestamp": str(int(now * 1000)),
            "tick_size": "0.001",
            "min_order_size": "5",
            "bids": [
                {"price": "0.498", "size": "1000"},
                {"price": "0.499", "size": "1200"},
            ],
            "asks": [
                {"price": "0.502", "size": "900"},
                {"price": "0.501", "size": "800"},
            ],
        },
        now,
    )
    assert book.best_bid == Decimal("0.499")
    assert book.best_ask == Decimal("0.501")
    features = FeatureEngine(config)
    for offset in range(config.strategy.warmup_seconds + 2):
        features.observe_book(book, now + offset)
    snapshot = features.snapshot(book, now + config.strategy.warmup_seconds + 2)
    assert snapshot is not None and snapshot.warmed_up
    signal = SignalEngine(config).evaluate(snapshot)
    assert 0.0 <= signal.score <= 100.0
    assert config.execution.mode == "paper"
    CONSOLE.print(
        "[bold green]Validation passed.[/bold green] "
        "Configuration, tick rounding, book parsing, feature warm-up, and signal scoring are operational."
    )
    CONSOLE.print(
        f"Execution mode: [bold]{config.execution.mode}[/bold] | "
        f"Starting paper cash: [bold]${config.risk.starting_cash_usdc:,.2f}[/bold] | "
        f"Target: [bold]{config.strategy.target_return * 100:.2f}%[/bold]"
    )


if __name__ == "__main__":
    raise SystemExit(main())
