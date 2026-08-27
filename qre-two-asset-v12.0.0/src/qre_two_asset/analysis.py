"""Diagnostic analyses that explain where excess wealth came from."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import AllocationConfig, ModelConfig
from .portfolio import AllocatorSpec, BacktestResult, allocate, backtest, performance_metrics


HISTORICAL_EPISODES = (
    ("dot_com_bear", "2000-03-24", "2002-10-09"),
    ("2003_2007_bull", "2003-03-12", "2007-10-09"),
    ("global_financial_crisis", "2007-10-09", "2009-03-09"),
    ("2009_2019_expansion", "2009-03-09", "2019-12-31"),
    ("2011_volatility", "2011-07-22", "2011-10-03"),
    ("2015_2016_selloff", "2015-05-21", "2016-02-11"),
    ("q4_2018", "2018-09-20", "2018-12-24"),
    ("covid_crash", "2020-02-19", "2020-03-23"),
    ("2020_2021_recovery", "2020-03-23", "2021-12-31"),
    ("2022_inflation_rate_shock", "2022-01-03", "2022-10-12"),
    ("post_2022_equity_bull", "2022-10-12", "2099-12-31"),
)

BEAR_EPISODES = (
    ("dot_com", "2000-03-24", "2002-10-09", "2007-05-30"),
    ("global_financial_crisis", "2007-10-09", "2009-03-09", "2013-03-28"),
    ("q4_2018", "2018-09-20", "2018-12-24", "2019-04-23"),
    ("covid", "2020-02-19", "2020-03-23", "2020-08-18"),
    ("2022_bear", "2022-01-03", "2022-10-12", "2024-01-19"),
)


def _compound(series: pd.Series) -> float:
    return float((1.0 + series.dropna()).prod() - 1.0) if len(series.dropna()) else float("nan")


def historical_episode_analysis(
    result: BacktestResult, decision_returns: pd.DataFrame
) -> pd.DataFrame:
    """Fixed, hindsight-labeled episodes for diagnosis only, never optimization."""

    rows: list[dict[str, float | int | str]] = []
    for name, start, end in HISTORICAL_EPISODES:
        dates = result.returns.index[
            (result.returns.index >= pd.Timestamp(start))
            & (result.returns.index <= pd.Timestamp(end))
        ]
        if len(dates) < 20:
            continue
        strategy = result.returns.reindex(dates)
        voo = decision_returns["VOO"].reindex(dates)
        kmlm = decision_returns["KMLM"].reindex(dates)
        rows.append(
            {
                "episode": name,
                "start": dates[0].date().isoformat(),
                "end": dates[-1].date().isoformat(),
                "sessions": len(dates),
                "strategy_return": _compound(strategy),
                "voo_return": _compound(voo),
                "kmlm_return": _compound(kmlm),
                "strategy_minus_voo_wealth": _compound(strategy) - _compound(voo),
                "average_voo_weight": float(result.weights.reindex(dates)["VOO"].mean()),
            }
        )
    return pd.DataFrame(rows)


def bear_market_timing_analysis(
    result: BacktestResult, decision_returns: pd.DataFrame
) -> pd.DataFrame:
    """Quantify reduction, re-entry, missed recovery, and net wealth by fixed bear episode."""

    rows: list[dict[str, float | str | None]] = []
    weights = result.weights["VOO"]
    changes = weights.diff()
    for name, peak, trough, recovery_end in BEAR_EPISODES:
        peak_date = pd.Timestamp(peak)
        trough_date = pd.Timestamp(trough)
        recovery_date = pd.Timestamp(recovery_end)
        decline_dates = weights.index[
            (weights.index >= peak_date) & (weights.index <= trough_date)
        ]
        recovery_dates = weights.index[
            (weights.index > trough_date) & (weights.index <= recovery_date)
        ]
        if len(decline_dates) < 10 or len(recovery_dates) < 10:
            continue
        reductions = changes.reindex(decline_dates).loc[lambda value: value < -1e-12]
        increases = changes.reindex(recovery_dates).loc[lambda value: value > 1e-12]
        first_reduction = reductions.index[0] if len(reductions) else None
        first_increase = increases.index[0] if len(increases) else None
        before_reduction_dates = decline_dates[
            decline_dates <= (first_reduction if first_reduction is not None else decline_dates[-1])
        ]
        after_reduction_dates = decline_dates[
            decline_dates >= (first_reduction if first_reduction is not None else decline_dates[0])
        ]
        before_increase_dates = recovery_dates[
            recovery_dates <= (first_increase if first_increase is not None else recovery_dates[-1])
        ]
        whole_dates = weights.index[
            (weights.index >= peak_date) & (weights.index <= recovery_date)
        ]
        strategy_return = _compound(result.returns.reindex(whole_dates))
        voo_return = _compound(decision_returns["VOO"].reindex(whole_dates))
        rows.append(
            {
                "episode": name,
                "first_voo_reduction": (
                    first_reduction.date().isoformat() if first_reduction is not None else None
                ),
                "voo_return_before_reduction": _compound(
                    decision_returns["VOO"].reindex(before_reduction_dates)
                ),
                "kmlm_return_reduction_to_trough": _compound(
                    decision_returns["KMLM"].reindex(after_reduction_dates)
                ),
                "first_voo_increase_after_trough": (
                    first_increase.date().isoformat() if first_increase is not None else None
                ),
                "voo_recovery_before_increase": _compound(
                    decision_returns["VOO"].reindex(before_increase_dates)
                ),
                "average_voo_weight_decline": float(weights.reindex(decline_dates).mean()),
                "average_voo_weight_recovery": float(weights.reindex(recovery_dates).mean()),
                "strategy_peak_to_recovery_return": strategy_return,
                "voo_peak_to_recovery_return": voo_return,
                "net_wealth_advantage_vs_voo": strategy_return - voo_return,
            }
        )
    return pd.DataFrame(rows)


def regime_contribution(
    result: BacktestResult,
    decision_returns: pd.DataFrame,
    voo_levels: pd.Series,
) -> pd.DataFrame:
    dates = result.returns.index
    voo = decision_returns["VOO"].reindex(dates)
    drawdown = (voo_levels / voo_levels.cummax() - 1.0).reindex(dates).ffill()
    realized_vol = (
        np.log(voo_levels).diff().rolling(21, min_periods=10).std(ddof=0) * np.sqrt(252.0)
    ).reindex(dates)
    high_vol_threshold = float(realized_vol.quantile(0.75))
    labels = pd.Series("ordinary", index=dates, dtype=object)
    labels.loc[drawdown <= -0.20] = "major_bear"
    labels.loc[(drawdown <= -0.10) & (drawdown > -0.20)] = "correction"
    labels.loc[(drawdown > -0.10) & (realized_vol >= high_vol_threshold)] = "high_vol_nonbear"
    labels.loc[(drawdown > -0.05) & (realized_vol < high_vol_threshold)] = "bull_or_calm"
    rows: list[dict[str, float | int | str]] = []
    for label, selected_dates in labels.groupby(labels).groups.items():
        strategy = result.returns.reindex(selected_dates).dropna()
        benchmark = voo.reindex(strategy.index)
        rows.append(
            {
                "regime": str(label),
                "sessions": len(strategy),
                "strategy_log_wealth_contribution": float(np.log1p(strategy).sum()),
                "voo_log_wealth_contribution": float(np.log1p(benchmark).sum()),
                "excess_log_wealth_contribution": float(
                    np.log1p(strategy).sum() - np.log1p(benchmark).sum()
                ),
                "average_voo_weight": float(result.weights.loc[strategy.index, "VOO"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def false_signal_analysis(
    result: BacktestResult, decision_returns: pd.DataFrame, defensive_threshold: float = 0.50
) -> pd.DataFrame:
    dates = result.returns.index
    voo = decision_returns["VOO"].reindex(dates)
    kmlm = decision_returns["KMLM"].reindex(dates)
    defensive = result.weights["VOO"].reindex(dates) < defensive_threshold
    false = defensive & voo.gt(kmlm)
    opportunity = (voo - (result.weights["VOO"] * voo + result.weights["KMLM"] * kmlm)).where(
        false, 0.0
    )
    groups = (~false).cumsum()
    durations = false.groupby(groups).sum()
    return pd.DataFrame(
        [
            {
                "defensive_sessions": int(defensive.sum()),
                "false_defensive_sessions": int(false.sum()),
                "false_signal_rate": float(false.sum() / max(defensive.sum(), 1)),
                "longest_false_signal_sessions": int(durations.max()) if len(durations) else 0,
                "approximate_voo_opportunity_cost": float(opportunity.sum()),
            }
        ]
    )


def cost_and_threshold_robustness(
    signals: pd.DataFrame,
    decision_returns: pd.DataFrame,
    base_config: AllocationConfig,
    selected: AllocatorSpec,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    def evaluate(
        scenario: str,
        scenario_signals: pd.DataFrame,
        scenario_returns: pd.DataFrame,
        spec: AllocatorSpec,
        cost: float,
    ) -> None:
        weights = allocate(scenario_signals, base_config, spec)
        result = backtest(weights, scenario_returns, transaction_cost_bps=cost)
        metrics = performance_metrics(result, scenario_returns["VOO"])
        rows.append(
            {
                "scenario": scenario,
                "transaction_cost_bps": cost,
                "minimum_change": spec.minimum_change,
                "cadence": spec.cadence,
                "cagr": metrics["cagr"],
                "annualized_excess_return": metrics["annualized_excess_return"],
                "terminal_wealth": metrics["terminal_wealth"],
                "annualized_turnover": metrics["annualized_turnover"],
            }
        )

    for cost in (2.0, 5.0, 10.0, 20.0):
        for change in sorted({0.03, selected.minimum_change, 0.10, 0.15}):
            spec = AllocatorSpec(selected.method, selected.cadence, float(change))
            evaluate("implementation_grid", signals, decision_returns, spec, cost)

    alternate_cadence = "daily" if selected.cadence == "weekly" else "weekly"
    evaluate(
        "alternate_cadence",
        signals,
        decision_returns,
        AllocatorSpec(selected.method, alternate_cadence, selected.minimum_change),
        base_config.transaction_cost_bps,
    )
    for scale in (0.90, 1.10):
        perturbed = signals.copy()
        perturbed["allocation_score"] = (perturbed["allocation_score"] * scale).clip(-1, 1)
        perturbed["probability_voo_outperforms"] = (
            0.5 + (perturbed["probability_voo_outperforms"] - 0.5) * scale
        ).clip(0, 1)
        evaluate(
            f"forecast_strength_{scale:.2f}",
            perturbed,
            decision_returns,
            selected,
            base_config.transaction_cost_bps,
        )

    # One extra session of delay is a direct stress of the execution assumption.
    evaluate(
        "additional_one_session_delay",
        signals,
        decision_returns.shift(-1),
        selected,
        base_config.transaction_cost_bps,
    )

    first = signals.index.min()
    last = signals.index.max()
    for years in (1, 3):
        later = signals.index[signals.index >= first + pd.DateOffset(years=years)]
        if len(later) >= 252:
            evaluate(
                f"start_plus_{years}y",
                signals.loc[later],
                decision_returns.loc[later],
                selected,
                base_config.transaction_cost_bps,
            )
        earlier = signals.index[signals.index <= last - pd.DateOffset(years=years)]
        if len(earlier) >= 252:
            evaluate(
                f"end_minus_{years}y",
                signals.loc[earlier],
                decision_returns.loc[earlier],
                selected,
                base_config.transaction_cost_bps,
            )
    return pd.DataFrame(rows)


def horizon_weight_robustness(
    raw_predictions: pd.DataFrame,
    decision_returns: pd.DataFrame,
    model_config: ModelConfig,
    allocation_config: AllocationConfig,
    selected: AllocatorSpec,
) -> pd.DataFrame:
    """Stress frozen horizon forecasts without refitting or candidate selection."""

    from .models import aggregate_horizons

    weight_sets = {
        "base": model_config.horizon_weights,
        "equal": (1 / 3, 1 / 3, 1 / 3),
        "21_only": (1.0, 0.0, 0.0),
        "63_only": (0.0, 1.0, 0.0),
        "126_only": (0.0, 0.0, 1.0),
    }
    rows: list[dict[str, float | str]] = []
    for scenario, weights in weight_sets.items():
        signals = aggregate_horizons(
            raw_predictions, replace(model_config, horizon_weights=weights)
        )
        result = backtest(
            allocate(signals, allocation_config, selected),
            decision_returns,
            transaction_cost_bps=allocation_config.transaction_cost_bps,
        )
        metrics = performance_metrics(result, decision_returns["VOO"])
        rows.append(
            {
                "scenario": scenario,
                "horizon_weights": "/".join(f"{value:.4f}" for value in weights),
                "cagr": metrics["cagr"],
                "annualized_excess_return": metrics["annualized_excess_return"],
                "terminal_wealth": metrics["terminal_wealth"],
                "annualized_turnover": metrics["annualized_turnover"],
            }
        )
    return pd.DataFrame(rows)


def deployment_gate(
    holdout_metrics: dict[str, float],
    preholdout_rolling_5y_win_rate: float,
    preholdout_reality_check_pvalue: float,
    provenance_allows_claim: bool,
    *,
    minimum_rolling_win_rate: float,
    maximum_pvalue: float,
) -> dict[str, object]:
    checks = {
        "provenance_allows_primary_claim": bool(provenance_allows_claim),
        "holdout_cagr_excess_positive": holdout_metrics["annualized_excess_return"] > 0.0,
        "holdout_terminal_excess_positive": holdout_metrics["cumulative_excess_wealth"] > 0.0,
        "preholdout_rolling_5y_win_rate": bool(
            np.isfinite(preholdout_rolling_5y_win_rate)
            and preholdout_rolling_5y_win_rate >= minimum_rolling_win_rate
        ),
        "preholdout_reality_check": bool(
            np.isfinite(preholdout_reality_check_pvalue)
            and preholdout_reality_check_pvalue <= maximum_pvalue
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "failure_message": "" if all(checks.values()) else "QRE failed the frozen VOO alpha gate",
    }
