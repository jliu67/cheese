from __future__ import annotations

import csv
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable

from .config import AppConfig
from .models import Asset, ClosedTrade, Fill, PaperOrder, SignalSnapshot
from .util import iso_now, json_dumps


class RawEventRecorder:
    def __init__(self, config: AppConfig, run_id: str) -> None:
        root = Path(config.storage.raw_event_dir)
        date_dir = root / iso_now()[:10]
        date_dir.mkdir(parents=True, exist_ok=True)
        self.path = date_dir / f"events-{run_id}.jsonl"
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self._flush_every = max(1, config.storage.raw_flush_every)
        self._count = 0

    def write(self, payload: dict[str, Any], received_at: float) -> None:
        record = {"received_at": received_at, "payload": payload}
        self._handle.write(json_dumps(record) + "\n")
        self._count += 1
        if self._count % self._flush_every == 0:
            self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()


class SQLiteStore:
    def __init__(self, config: AppConfig) -> None:
        path = Path(config.storage.sqlite_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self._dirty = 0
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                config_json TEXT NOT NULL,
                note TEXT
            );

            CREATE TABLE IF NOT EXISTS assets (
                run_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                condition_id TEXT NOT NULL,
                slug TEXT,
                question TEXT NOT NULL,
                outcome TEXT NOT NULL,
                market_liquidity TEXT NOT NULL,
                market_volume_24h TEXT NOT NULL,
                end_date TEXT,
                fee_json TEXT NOT NULL,
                PRIMARY KEY (run_id, token_id)
            );

            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                token_id TEXT NOT NULL,
                score REAL NOT NULL,
                eligible INTEGER NOT NULL,
                reason TEXT NOT NULL,
                features_json TEXT NOT NULL,
                components_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_signals_run_time ON signals(run_id, timestamp);

            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                side TEXT NOT NULL,
                intent TEXT NOT NULL,
                limit_price TEXT NOT NULL,
                quantity TEXT NOT NULL,
                filled_quantity TEXT NOT NULL,
                average_fill_price TEXT NOT NULL,
                queue_ahead TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL,
                status TEXT NOT NULL,
                score_at_creation REAL NOT NULL,
                reason TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                order_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                side TEXT NOT NULL,
                intent TEXT NOT NULL,
                quantity TEXT NOT NULL,
                price TEXT NOT NULL,
                fee TEXT NOT NULL,
                maker INTEGER NOT NULL,
                timestamp REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS closed_trades (
                trade_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                question TEXT NOT NULL,
                opened_at REAL NOT NULL,
                closed_at REAL NOT NULL,
                quantity TEXT NOT NULL,
                average_entry_price TEXT NOT NULL,
                average_exit_price TEXT NOT NULL,
                gross_pnl TEXT NOT NULL,
                fees TEXT NOT NULL,
                net_pnl TEXT NOT NULL,
                return_pct TEXT NOT NULL,
                exit_reason TEXT NOT NULL,
                entry_score REAL NOT NULL,
                exit_score REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                cash TEXT NOT NULL,
                exposure TEXT NOT NULL,
                unrealized_pnl TEXT NOT NULL,
                realized_pnl TEXT NOT NULL,
                equity TEXT NOT NULL,
                open_positions INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_equity_run_time ON equity(run_id, timestamp);
            """
        )
        self.connection.commit()

    def start_run(self, config: AppConfig, note: str | None = None) -> str:
        run_id = uuid.uuid4().hex[:12]
        self.connection.execute(
            "INSERT INTO runs(run_id, started_at, mode, status, config_json, note) VALUES(?,?,?,?,?,?)",
            (run_id, iso_now(), config.execution.mode, "RUNNING", json_dumps(config.as_dict()), note),
        )
        self.connection.commit()
        return run_id

    def finish_run(self, run_id: str, status: str, note: str | None = None) -> None:
        self.connection.execute(
            "UPDATE runs SET finished_at=?, status=?, note=COALESCE(?, note) WHERE run_id=?",
            (iso_now(), status, note, run_id),
        )
        self.connection.commit()

    def record_assets(self, run_id: str, assets: Iterable[Asset]) -> None:
        rows = []
        for asset in assets:
            rows.append(
                (
                    run_id,
                    asset.token_id,
                    asset.market_id,
                    asset.condition_id,
                    asset.slug,
                    asset.question,
                    asset.outcome,
                    str(asset.market_liquidity),
                    str(asset.market_volume_24h),
                    asset.end_date.isoformat() if asset.end_date else None,
                    json_dumps(asset.fee_schedule),
                )
            )
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO assets(
                run_id, token_id, market_id, condition_id, slug, question, outcome,
                market_liquidity, market_volume_24h, end_date, fee_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        self.connection.commit()

    def record_signal(self, run_id: str, signal: SignalSnapshot) -> None:
        self.connection.execute(
            """
            INSERT INTO signals(run_id, timestamp, token_id, score, eligible, reason, features_json, components_json)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                signal.timestamp,
                signal.token_id,
                signal.score,
                int(signal.eligible),
                signal.reason,
                json_dumps(signal.features.as_dict()),
                json_dumps(signal.components),
            ),
        )
        self._mark_dirty()

    def record_order(self, run_id: str, order: PaperOrder, *, force: bool = False) -> None:
        self.connection.execute(
            """
            INSERT INTO orders(
                order_id, run_id, token_id, market_id, outcome, side, intent,
                limit_price, quantity, filled_quantity, average_fill_price, queue_ahead,
                created_at, expires_at, status, score_at_creation, reason, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(order_id) DO UPDATE SET
                filled_quantity=excluded.filled_quantity,
                average_fill_price=excluded.average_fill_price,
                queue_ahead=excluded.queue_ahead,
                expires_at=excluded.expires_at,
                status=excluded.status,
                reason=excluded.reason,
                updated_at=excluded.updated_at
            """,
            (
                order.id,
                run_id,
                order.token_id,
                order.market_id,
                order.outcome,
                order.side,
                order.intent,
                str(order.limit_price),
                str(order.quantity),
                str(order.filled_quantity),
                str(order.average_fill_price),
                str(order.queue_ahead),
                order.created_at,
                order.expires_at,
                order.status,
                order.score_at_creation,
                order.reason,
                iso_now(),
            ),
        )
        self._mark_dirty(force=force)

    def record_fill(self, run_id: str, fill: Fill) -> None:
        self.connection.execute(
            """
            INSERT INTO fills(fill_id, run_id, order_id, token_id, side, intent, quantity, price, fee, maker, timestamp)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                fill.id,
                run_id,
                fill.order_id,
                fill.token_id,
                fill.side,
                fill.intent,
                str(fill.quantity),
                str(fill.price),
                str(fill.fee),
                int(fill.maker),
                fill.timestamp,
            ),
        )
        self._mark_dirty(force=True)

    def record_closed_trade(self, run_id: str, trade: ClosedTrade) -> None:
        self.connection.execute(
            """
            INSERT INTO closed_trades(
                trade_id, run_id, token_id, market_id, outcome, question, opened_at,
                closed_at, quantity, average_entry_price, average_exit_price,
                gross_pnl, fees, net_pnl, return_pct, exit_reason, entry_score, exit_score
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                trade.id,
                run_id,
                trade.token_id,
                trade.market_id,
                trade.outcome,
                trade.question,
                trade.opened_at,
                trade.closed_at,
                str(trade.quantity),
                str(trade.average_entry_price),
                str(trade.average_exit_price),
                str(trade.gross_pnl),
                str(trade.fees),
                str(trade.net_pnl),
                str(trade.return_pct),
                trade.exit_reason,
                trade.entry_score,
                trade.exit_score,
            ),
        )
        self._mark_dirty(force=True)

    def record_equity(
        self,
        run_id: str,
        timestamp: float,
        cash: Any,
        exposure: Any,
        unrealized_pnl: Any,
        realized_pnl: Any,
        equity: Any,
        open_positions: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO equity(run_id, timestamp, cash, exposure, unrealized_pnl, realized_pnl, equity, open_positions)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                timestamp,
                str(cash),
                str(exposure),
                str(unrealized_pnl),
                str(realized_pnl),
                str(equity),
                open_positions,
            ),
        )
        self._mark_dirty()

    def latest_run_id(self) -> str | None:
        row = self.connection.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return str(row[0]) if row else None

    def fetch_all(self, query: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return list(self.connection.execute(query, parameters).fetchall())

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def _mark_dirty(self, *, force: bool = False) -> None:
        self._dirty += 1
        if force or self._dirty >= 50:
            self.connection.commit()
            self._dirty = 0


def write_csv(path: Path, rows: list[sqlite3.Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
