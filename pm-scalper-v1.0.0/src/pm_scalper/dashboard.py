from __future__ import annotations

from decimal import Decimal
from typing import Mapping

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import Asset, SignalSnapshot
from .paper import PaperBroker
from .util import fmt_money


class Dashboard:
    def __init__(self, assets: Mapping[str, Asset]) -> None:
        self.assets = dict(assets)

    def render(
        self,
        broker: PaperBroker,
        signals: Mapping[str, SignalSnapshot],
        run_id: str,
        raw_path: str,
    ) -> Group:
        header = Table.grid(expand=True)
        header.add_column()
        header.add_column(justify="right")
        status = "HALTED: " + broker.halt_reason if broker.halted else "PAPER MODE"
        header.add_row(
            Text(f"PM-SCALPER 1.0  |  run {run_id}", style="bold"),
            Text(status, style="bold red" if broker.halted else "bold green"),
        )
        header.add_row(
            f"Equity {fmt_money(broker.equity)}   Cash {fmt_money(broker.cash)}   "
            f"Exposure {fmt_money(broker.exposure)}",
            f"Realized {fmt_money(broker.realized_pnl)}   Unrealized {fmt_money(broker.unrealized_pnl)}",
        )
        header.add_row(
            f"Closed {broker.closed_trades}   Win rate {broker.win_rate * 100:.1f}%   "
            f"Fees {fmt_money(broker.total_fees)}",
            f"Raw feed: {raw_path}",
        )

        opportunities = Table(title="Top short-horizon signals", expand=True)
        opportunities.add_column("Score", justify="right", width=7)
        opportunities.add_column("Outcome", width=8)
        opportunities.add_column("Bid / Ask", justify="right", width=14)
        opportunities.add_column("Spread", justify="right", width=8)
        opportunities.add_column("Flow", justify="right", width=8)
        opportunities.add_column("Market")
        ranked = sorted(signals.values(), key=lambda item: item.score, reverse=True)[:10]
        for signal in ranked:
            asset = self.assets.get(signal.token_id)
            if asset is None:
                continue
            features = signal.features
            score_style = "green" if signal.eligible else "yellow" if signal.score >= 60 else "white"
            opportunities.add_row(
                Text(f"{signal.score:5.1f}", style=score_style),
                asset.outcome,
                f"{features.best_bid:.4f} / {features.best_ask:.4f}",
                f"{features.spread:.4f}",
                f"{features.trade_flow:+.2f}",
                asset.question[:78],
            )
        if not ranked:
            opportunities.add_row("—", "—", "—", "—", "—", "Waiting for books")

        positions = Table(title="Positions and working orders", expand=True)
        positions.add_column("Type", width=9)
        positions.add_column("Outcome", width=8)
        positions.add_column("Entry / Limit", justify="right", width=14)
        positions.add_column("Bid / Target", justify="right", width=14)
        positions.add_column("Qty", justify="right", width=11)
        positions.add_column("P&L", justify="right", width=10)
        positions.add_column("Market")

        for token_id, position in broker.positions.items():
            asset = self.assets[token_id]
            book = broker.books.get(token_id)
            bid = book.best_bid if book and book.best_bid is not None else Decimal("0")
            pnl = position.quantity * bid - position.entry_cost - position.entry_fees
            positions.add_row(
                "POSITION",
                position.outcome,
                f"{position.average_entry_price:.4f}",
                f"{bid:.4f} / {position.target_price:.4f}",
                f"{position.quantity:.2f}",
                f"{pnl:+.2f}",
                asset.question[:72],
            )

        for order in broker.orders.values():
            if order.status not in {"OPEN", "PARTIAL"}:
                continue
            asset = self.assets[order.token_id]
            positions.add_row(
                order.intent,
                order.outcome,
                f"{order.limit_price:.4f}",
                f"queue {order.queue_ahead:.1f}",
                f"{order.remaining_quantity:.2f}",
                "—",
                asset.question[:72],
            )
        if not broker.positions and not any(
            order.status in {"OPEN", "PARTIAL"} for order in broker.orders.values()
        ):
            positions.add_row("—", "—", "—", "—", "—", "—", "No trades yet")

        return Group(Panel(header, title="Session"), opportunities, positions)
