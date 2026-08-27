"""Human-readable reports derived only from calculated artifacts."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value * 100:.2f}%"


def _table(frame: pd.DataFrame) -> str:
    """Render a dependency-free fixed-width table inside Markdown."""

    if frame.empty:
        return "_No rows available._"
    return "```text\n" + frame.to_string(index=False) + "\n```"


def development_report(
    manifest: dict[str, Any], ablations: pd.DataFrame, benchmarks: pd.DataFrame
) -> str:
    selected = manifest["selected"]
    lines = [
        "# QRE VOO/KMLM development report",
        "",
        "This report uses pre-holdout walk-forward predictions only. The final holdout has not been evaluated.",
        "",
        "## Frozen candidate",
        "",
        f"- Feature variant: {selected['variant']}",
        f"- Allocator: {selected['allocator']}",
        f"- Pre-holdout CAGR: {_pct(selected['metrics']['cagr'])}",
        f"- VOO CAGR: {_pct(selected['metrics']['benchmark_cagr'])}",
        f"- Annualized excess: {_pct(selected['metrics']['annualized_excess_return'])}",
        f"- Reality Check p-value: {manifest['statistics']['reality_check']['pvalue']:.4f}",
        "",
        "## Ablations",
        "",
        _table(ablations),
        "",
        "## Static and single-asset benchmarks",
        "",
        _table(benchmarks),
        "",
        "## Status",
        "",
        "No alpha claim is permitted until the untouched holdout is evaluated once and passes every frozen gate.",
        "",
    ]
    return "\n".join(lines)


def holdout_report(result: dict[str, Any], benchmarks: pd.DataFrame, rolling: pd.DataFrame) -> str:
    metrics = result["metrics"]
    gate = result["deployment_gate"]
    status = "PASSED" if gate["passed"] else "FAILED"
    lines = [
        "# QRE VOO/KMLM final holdout report",
        "",
        f"## Primary objective: {status}",
        "",
        f"- QRE CAGR: {_pct(metrics['cagr'])}",
        f"- VOO CAGR: {_pct(metrics['benchmark_cagr'])}",
        f"- Annualized excess return: {_pct(metrics['annualized_excess_return'])}",
        f"- QRE terminal wealth: {metrics['terminal_wealth']:.4f}",
        f"- VOO terminal wealth: {metrics['benchmark_terminal_wealth']:.4f}",
        f"- Maximum drawdown (secondary): {_pct(metrics['maximum_drawdown'])}",
        "",
        "## Frozen gate checks",
        "",
    ]
    for name, passed in gate["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    lines.extend(
        [
            "",
            "## Rolling excess-return windows",
            "",
            _table(rolling),
            "",
            "## Static and single-asset benchmarks",
            "",
            _table(benchmarks),
            "",
        ]
    )
    if not gate["passed"]:
        lines.extend(
            [
                "The strategy failed the stated objective. Better drawdown or Sharpe statistics do not override that failure.",
                "",
            ]
        )
    return "\n".join(lines)


def live_report(payload: dict[str, Any]) -> str:
    allocation = payload["allocation"]
    interval = payload["prediction_interval_80"]
    lines = [
        "# QRE VOO/KMLM allocation",
        "",
        f"Signal date: {payload['signal_date']}",
        f"Deployment status: {payload['deployment_status']}",
        "",
        f"- VOO: {allocation['VOO'] * 100:.1f}%",
        f"- KMLM: {allocation['KMLM'] * 100:.1f}%",
        f"- Probability VOO outperforms: {payload['probability_voo_outperforms'] * 100:.1f}%",
        f"- Directional confidence: {payload['directional_confidence'] * 100:.1f}%",
        f"- 63-session-equivalent expected relative return: {_pct(payload['expected_relative_return_63_equivalent'])}",
        (
            "- 80% relative-return prediction interval: n/a"
            if interval[0] is None
            else f"- 80% relative-return prediction interval: {_pct(interval[0])} to {_pct(interval[1])}"
        ),
        "",
        "Horizon forecasts (positive favors VOO; negative favors KMLM):",
        "",
    ]
    for horizon, forecast in payload.get("horizon_forecasts", {}).items():
        horizon_interval = forecast.get("prediction_interval_80", [None, None])
        interval_text = (
            "n/a"
            if horizon_interval[0] is None
            else f"{_pct(horizon_interval[0])} to {_pct(horizon_interval[1])}"
        )
        lines.append(
            f"- {horizon} sessions: {_pct(forecast['expected_relative_return'])}; "
            f"P(VOO wins)={_pct(forecast['probability_voo_outperforms'])}; "
            f"80% interval={interval_text}"
        )
    lines.extend(
        [
        "",
        "Major mechanically derived signals:",
        "",
        ]
    )
    for item in payload.get("major_signals", []):
        direction = "VOO-positive" if item["contribution"] >= 0 else "KMLM-positive"
        lines.append(f"- {direction} — {item['feature']}: {item['contribution']:+.5f}")
    if payload.get("failure_reason"):
        lines.extend(["", f"Fail-closed reason: {payload['failure_reason']}"])
    lines.extend(
        [
            "",
            "The signal uses data through the stated close and assumes execution no earlier than the next session close.",
            "",
        ]
    )
    return "\n".join(lines)
