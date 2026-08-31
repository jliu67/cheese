from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from pm_scalper.config import AppConfig, ExecutionConfig, load_config
from pm_scalper.models import FeeSchedule
from pm_scalper.util import ceil_to_tick, floor_to_tick, json_dumps, parse_timestamp


ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_paper_only() -> None:
    config = load_config(ROOT / "config" / "default.yaml")
    assert config.execution.mode == "paper"
    assert config.execution.entry_style == "maker"
    assert config.strategy.target_return == pytest.approx(0.01)


def test_live_mode_is_rejected() -> None:
    config = AppConfig(execution=ExecutionConfig(mode="live"))
    with pytest.raises(ValueError, match="no live-order path"):
        config.validate()


def test_tick_rounding_and_timestamps() -> None:
    tick = Decimal("0.005")
    assert floor_to_tick(Decimal("0.5079"), tick) == Decimal("0.505")
    assert ceil_to_tick(Decimal("0.5079"), tick) == Decimal("0.51")
    parsed = parse_timestamp("1782753357257")
    assert parsed is not None
    assert parsed.tzinfo is not None


def test_slotted_dataclass_serializes() -> None:
    payload = json_dumps(FeeSchedule(enabled=True, rate=Decimal("0.05")))
    assert '"enabled":true' in payload
    assert '"rate":"0.05"' in payload
