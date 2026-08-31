from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any, Iterable

ZERO = Decimal("0")
ONE = Decimal("1")


def dec(value: Any, default: Decimal = ZERO) -> Decimal:
    """Convert loose API values to Decimal without leaking float artifacts."""
    if value is None or value == "":
        return default
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return Decimal(int(value))
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def opt_dec(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float, Decimal)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict, tuple)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    return default


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float, Decimal)) or str(value).isdigit():
        number = float(value)
        if number > 10_000_000_000:
            number /= 1000.0
        try:
            return datetime.fromtimestamp(number, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def epoch_seconds(value: Any, fallback: float | None = None) -> float:
    parsed = parse_timestamp(value)
    if parsed is not None:
        return parsed.timestamp()
    return fallback if fallback is not None else utc_now().timestamp()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def bounded_tanh(value: float, scale: float) -> float:
    if not math.isfinite(value) or scale <= 0:
        return 0.0
    return math.tanh(value / scale)


def floor_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    if tick <= ZERO:
        return price
    units = (price / tick).to_integral_value(rounding=ROUND_FLOOR)
    return (units * tick).normalize()


def ceil_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    if tick <= ZERO:
        return price
    units = (price / tick).to_integral_value(rounding=ROUND_CEILING)
    return (units * tick).normalize()


def decimal_places(value: Decimal) -> int:
    exponent = value.as_tuple().exponent
    return max(0, -exponent)


def fmt_decimal(value: Decimal | None, places: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{places}f}"


def fmt_money(value: Decimal | float | int) -> str:
    return f"${float(value):,.2f}"


def json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_dumps(value: Any, *, compact: bool = True) -> str:
    kwargs = {"default": json_default, "ensure_ascii": False}
    if compact:
        kwargs.update({"separators": (",", ":")})
    else:
        kwargs.update({"indent": 2, "sort_keys": True})
    return json.dumps(value, **kwargs)


def ensure_parent(path: str | Path) -> Path:
    result = Path(path)
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]
