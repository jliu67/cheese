from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .config import AppConfig
from .storage import SQLiteStore, write_csv
from .util import fmt_money, json_dumps

CONSOLE = Console()
ZERO = Decimal("0")


def generate_report(config: AppConfig, run_id: str | None = None) -> Path:
    store = SQLiteStore(config)
    try:
        selected_run = run_id or store.latest_run_id()
        if not selected_run:
            raise RuntimeError("No PM-Scalper runs exist in the configured database")
        run_rows = store.fetch_all("SELECT * FROM runs WHERE run_id=?", (selected_run,))
        if not run_rows:
            raise RuntimeError(f"Run not found: {selected_run}")
        run = dict(run_rows[0])
        trades = store.fetch_all(
            "SELECT * FROM closed_trades WHERE run_id=? ORDER BY closed_at", (selected_run,)
        )
        equity_rows = store.fetch_all(
            "SELECT * FROM equity WHERE run_id=? ORDER BY timestamp", (selected_run,)
        )
        signals = store.fetch_all(
            "SELECT * FROM signals WHERE run_id=? ORDER BY timestamp", (selected_run,)
        )
        orders = store.fetch_all(
            "SELECT * FROM orders WHERE run_id=? ORDER BY created_at", (selected_run,)
        )
        fills = store.fetch_all(
            "SELECT * FROM fills WHERE run_id=? ORDER BY timestamp", (selected_run,)
        )

        export_root = Path(config.storage.export_dir) / selected_run
        export_root.mkdir(parents=True, exist_ok=True)
        write_csv(export_root / "trades.csv", trades)
        write_csv(export_root / "equity.csv", equity_rows)
        write_csv(export_root / "signals.csv", signals)
        write_csv(export_root / "orders.csv", orders)
        write_csv(export_root / "fills.csv", fills)

        net_values = [Decimal(row["net_pnl"]) for row in trades]
        return_values = [Decimal(row["return_pct"]) for row in trades]
        fee_values = [Decimal(row["fee"]) for row in fills]
        gross_profit = sum((value for value in net_values if value > ZERO), ZERO)
        gross_loss = -sum((value for value in net_values if value < ZERO), ZERO)
        wins = sum(1 for value in net_values if value > ZERO)
        losses = sum(1 for value in net_values if value <= ZERO)
        realized_net_pnl = sum(net_values, ZERO)
        total_fees = sum(fee_values, ZERO)
        average_return = (
            sum(return_values, ZERO) / Decimal(len(return_values)) if return_values else ZERO
        )
        profit_factor = (
            float(gross_profit / gross_loss) if gross_loss > ZERO else None
        )

        equity_values = [Decimal(row["equity"]) for row in equity_rows]
        try:
            stored_config = json.loads(run.get("config_json") or "{}")
            stored_starting_cash = stored_config["risk"]["starting_cash_usdc"]
        except (json.JSONDecodeError, KeyError, TypeError):
            stored_starting_cash = config.risk.starting_cash_usdc
        starting_equity = Decimal(str(stored_starting_cash))
        ending_equity = equity_values[-1] if equity_values else starting_equity
        session_net_pnl = ending_equity - starting_equity
        max_drawdown = calculate_max_drawdown(equity_values)
        entry_orders = sum(1 for row in orders if row["intent"] == "ENTRY")
        filled_entries = sum(
            1
            for row in orders
            if row["intent"] == "ENTRY" and Decimal(row["filled_quantity"]) > ZERO
        )
        fill_rate = filled_entries / entry_orders if entry_orders else 0.0

        summary: dict[str, Any] = {
            "run": run,
            "metrics": {
                "starting_equity": str(starting_equity),
                "ending_equity": str(ending_equity),
                "net_pnl": str(session_net_pnl),
                "realized_net_pnl": str(realized_net_pnl),
                "total_fees": str(total_fees),
                "closed_trades": len(trades),
                "wins": wins,
                "losses": losses,
                "win_rate": wins / len(trades) if trades else 0.0,
                "average_trade_return": str(average_return),
                "profit_factor": profit_factor,
                "max_drawdown": max_drawdown,
                "entry_orders": entry_orders,
                "entry_fill_rate": fill_rate,
                "fills": len(fills),
                "signal_samples": len(signals),
            },
            "artifacts": {
                "trades_csv": str(export_root / "trades.csv"),
                "equity_csv": str(export_root / "equity.csv"),
                "signals_csv": str(export_root / "signals.csv"),
                "orders_csv": str(export_root / "orders.csv"),
                "fills_csv": str(export_root / "fills.csv"),
            },
        }
        summary_path = export_root / "summary.json"
        summary_path.write_text(json_dumps(summary, compact=False) + "\n", encoding="utf-8")
        print_summary(summary, selected_run, summary_path)
        return summary_path
    finally:
        store.close()


def calculate_max_drawdown(values: list[Decimal]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    worst = Decimal("0")
    for value in values:
        peak = max(peak, value)
        if peak > ZERO:
            drawdown = value / peak - Decimal("1")
            worst = min(worst, drawdown)
    return float(worst)


def print_summary(summary: dict[str, Any], run_id: str, path: Path) -> None:
    metrics = summary["metrics"]
    table = Table(title=f"PM-Scalper report — {run_id}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Starting equity", fmt_money(Decimal(metrics["starting_equity"])))
    table.add_row("Ending equity", fmt_money(Decimal(metrics["ending_equity"])))
    table.add_row("Session net P&L", fmt_money(Decimal(metrics["net_pnl"])))
    table.add_row("Realized net P&L", fmt_money(Decimal(metrics["realized_net_pnl"])))
    table.add_row("Closed trades", str(metrics["closed_trades"]))
    table.add_row("Win rate", f"{metrics['win_rate'] * 100:.1f}%")
    table.add_row("Average trade return", f"{Decimal(metrics['average_trade_return']) * 100:.3f}%")
    table.add_row(
        "Profit factor",
        "—" if metrics["profit_factor"] is None else f"{metrics['profit_factor']:.2f}",
    )
    table.add_row("Maximum drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
    table.add_row("Entry fill rate", f"{metrics['entry_fill_rate'] * 100:.1f}%")
    table.add_row("Fees", fmt_money(Decimal(metrics["total_fees"])))
    CONSOLE.print(table)
    CONSOLE.print(f"Report files: [bold]{path.parent}[/bold]")
