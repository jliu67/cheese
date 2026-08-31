from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from pm_scalper.cli import parse_duration
from pm_scalper.config import AppConfig
from pm_scalper.models import Asset, FeeSchedule
from pm_scalper.report import generate_report
from pm_scalper.storage import SQLiteStore


def test_duration_parser() -> None:
    assert parse_duration("30s") == 30
    assert parse_duration("2m") == 120
    assert parse_duration("1.5h") == 5400
    assert parse_duration("0") == 0
    with pytest.raises(ValueError):
        parse_duration("-1m")


def test_storage_accepts_slotted_fee_schedule_and_report(tmp_path: Path) -> None:
    config = AppConfig()
    config.storage.sqlite_path = str(tmp_path / "data" / "test.sqlite3")
    config.storage.export_dir = str(tmp_path / "exports")
    config.storage.raw_event_dir = str(tmp_path / "raw")
    config.storage.log_dir = str(tmp_path / "logs")

    store = SQLiteStore(config)
    run_id = store.start_run(config, note="test")
    asset = Asset(
        token_id="token",
        market_id="market",
        condition_id="condition",
        slug="slug",
        question="Question?",
        outcome="Yes",
        market_liquidity=Decimal("1000"),
        market_volume_24h=Decimal("100"),
        end_date=None,
        fee_schedule=FeeSchedule(enabled=True, rate=Decimal("0.05")),
    )
    store.record_assets(run_id, [asset])
    store.record_equity(
        run_id,
        1.0,
        Decimal("10000"),
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        Decimal("10000"),
        0,
    )
    store.finish_run(run_id, "COMPLETED")
    store.close()

    # Reports must remain tied to the run snapshot even if the YAML changes later.
    config.risk.starting_cash_usdc = 5000
    summary_path = generate_report(config, run_id)
    assert summary_path.exists()
    text = summary_path.read_text(encoding="utf-8")
    assert '"closed_trades": 0' in text
    summary = json.loads(text)
    assert summary["metrics"]["starting_equity"] == "10000.0"
    assert summary["metrics"]["net_pnl"] == "0.0"
    assert (summary_path.parent / "equity.csv").exists()
