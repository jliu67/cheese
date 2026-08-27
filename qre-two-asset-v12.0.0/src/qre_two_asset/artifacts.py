"""Atomic, hashed research artifacts and a multiple-testing ledger."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_frame(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    if isinstance(normalized.index, pd.DatetimeIndex):
        normalized.index = normalized.index.strftime("%Y-%m-%d")
    payload = normalized.to_csv(index=True, float_format="%.16g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite numbers and normalize artifact types."""

    if value is pd.NA or value is pd.NaT:
        return None
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return None if pd.isna(value) else value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(
            json_safe(payload),
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
    )


def atomic_write_csv(path: str | Path, frame: pd.DataFrame, *, index: bool = True) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=index)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def atomic_joblib_dump(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    try:
        joblib.dump(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def append_ledger(path: str | Path, record: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"recorded_at": utc_now(), **record}
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                json_safe(payload), sort_keys=True, default=_json_default, allow_nan=False
            )
            + "\n"
        )


def write_audit_database(path: str | Path, tables: dict[str, pd.DataFrame]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    connection = sqlite3.connect(temporary)
    try:
        with connection:
            for name, frame in tables.items():
                output = frame.copy()
                if isinstance(output.index, pd.DatetimeIndex):
                    output.index = output.index.strftime("%Y-%m-%d")
                    output.index.name = "date"
                output.to_sql(name, connection, if_exists="replace", index=True)
            connection.execute(
                "CREATE TABLE schema_info(schema_version TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO schema_info(schema_version, created_at) VALUES (?, ?)",
                ("12.0", utc_now()),
            )
    finally:
        connection.close()
    os.replace(temporary, destination)
    return destination
