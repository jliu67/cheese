"""Transparent allocator, transaction-cost backtest, and benchmark metrics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from .config import AllocationConfig
from .errors import DataIntegrityError
from .splitters import weekly_decision_dates


@dataclass(frozen=True)
class AllocatorSpec:
    method: str
    cadence: str
    minimum_change: float

    @property
    def name(self) -> str:
        return f"{self.method}__{self.cadence}__change_{self.minimum_change:.2f}"


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series
    gross_returns: pd.Series
    equity: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series


def _nearest_bucket(value: float, buckets: tuple[float, ...]) -> float:
    array = np.asarray(buckets, dtype=float)
    return float(array[np.argmin(np.abs(array - value))])


def _raw_voo_target(signals: pd.DataFrame, config: AllocationConfig, method: str) -> pd.Series:
    score = signals["allocation_score"].clip(-1.0, 1.0)
    probability = signals["probability_voo_outperforms"].clip(0.0, 1.0)
    continuous = (config.neutral_voo_weight + config.maximum_tilt * score).clip(0.0, 1.0)
    if method == "continuous":
        return continuous
    if method == "bucketed":
        return continuous.map(lambda value: _nearest_bucket(float(value), config.buckets))
    if method == "probability":
        return ((probability - 0.35) / 0.30).clip(0.0, 1.0)
    raise DataIntegrityError(f"unknown allocation method: {method}")


def allocate(
    signals: pd.DataFrame,
    config: AllocationConfig,
    spec: AllocatorSpec,
    *,
    initial_voo_weight: float = 1.0,
) -> pd.DataFrame:
    if spec.cadence not in {"daily", "weekly"}:
        raise DataIntegrityError(f"unknown allocation cadence: {spec.cadence}")
    raw = _raw_voo_target(signals, config, spec.method)
    update_dates = set(raw.index)
    if spec.cadence == "weekly":
        update_dates = set(weekly_decision_dates(raw.index))
    accepted: list[float] = []
    if not 0 <= initial_voo_weight <= 1:
        raise DataIntegrityError("initial VOO weight must be between zero and one")
    previous = float(initial_voo_weight)  # benchmark is the fail-closed default
    for date, value in raw.items():
        if date not in update_dates or not np.isfinite(value):
            accepted.append(previous)
            continue
        if abs(float(value) - previous) + 1e-12 >= spec.minimum_change:
            previous = float(value)
        accepted.append(previous)
    weights = pd.DataFrame({"VOO": accepted}, index=raw.index)
    weights["KMLM"] = 1.0 - weights["VOO"]
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("allocator weights do not sum to one")
    if ((weights < -1e-12) | (weights > 1 + 1e-12)).any().any():
        raise AssertionError("allocator produced an out-of-bounds weight")
    return weights


def backtest(
    weights: pd.DataFrame,
    decision_returns: pd.DataFrame,
    *,
    transaction_cost_bps: float,
) -> BacktestResult:
    dates = weights.index.intersection(decision_returns.dropna(how="any").index)
    if len(dates) < 2:
        raise DataIntegrityError("not enough aligned observations for a backtest")
    aligned_weights = weights.loc[dates, ["VOO", "KMLM"]].astype(float)
    aligned_returns = decision_returns.loc[dates, ["VOO", "KMLM"]].astype(float)
    changes = aligned_weights.diff()
    changes.iloc[0] = aligned_weights.iloc[0] - pd.Series({"VOO": 1.0, "KMLM": 0.0})
    turnover = 0.5 * changes.abs().sum(axis=1)
    costs = turnover * float(transaction_cost_bps) / 10_000.0
    gross = (aligned_weights * aligned_returns).sum(axis=1)
    net = gross - costs
    equity = (1.0 + net).cumprod()
    return BacktestResult(net.rename("net_return"), gross.rename("gross_return"), equity, aligned_weights, turnover, costs)


def static_mix_backtest(
    decision_returns: pd.DataFrame,
    voo_weight: float,
    *,
    transaction_cost_bps: float,
    rebalance: str = "monthly",
) -> BacktestResult:
    if not 0 <= voo_weight <= 1:
        raise DataIntegrityError("static VOO weight must be between zero and one")
    returns = decision_returns.dropna(how="any").loc[:, ["VOO", "KMLM"]]
    target = np.array([voo_weight, 1.0 - voo_weight], dtype=float)
    current = target.copy()
    weight_rows: list[np.ndarray] = []
    turnover_rows: list[float] = []
    net_rows: list[float] = []
    gross_rows: list[float] = []
    cost_rows: list[float] = []
    previous_period: tuple[int, int] | None = None
    for date, row in returns.iterrows():
        period = (date.year, date.month) if rebalance == "monthly" else (date.year, 1)
        turnover = (
            0.5 * float(np.abs(target - np.array([1.0, 0.0])).sum())
            if previous_period is None
            else 0.0
        )
        if previous_period is not None and period != previous_period:
            turnover = 0.5 * float(np.abs(target - current).sum())
            current = target.copy()
        previous_period = period
        weight_rows.append(current.copy())
        gross = float(np.dot(current, row.to_numpy(dtype=float)))
        cost = turnover * transaction_cost_bps / 10_000.0
        gross_rows.append(gross)
        cost_rows.append(cost)
        turnover_rows.append(turnover)
        net_rows.append(gross - cost)
        post = current * (1.0 + row.to_numpy(dtype=float))
        current = post / post.sum()
    weights = pd.DataFrame(weight_rows, index=returns.index, columns=["VOO", "KMLM"])
    net = pd.Series(net_rows, index=returns.index, name="net_return")
    gross = pd.Series(gross_rows, index=returns.index, name="gross_return")
    turnover_series = pd.Series(turnover_rows, index=returns.index, name="turnover")
    costs = pd.Series(cost_rows, index=returns.index, name="cost")
    return BacktestResult(net, gross, (1.0 + net).cumprod(), weights, turnover_series, costs)


def _maximum_underwater_sessions(equity: pd.Series) -> int:
    underwater = equity < equity.cummax() - 1e-12
    groups = (~underwater).cumsum()
    return int(underwater.groupby(groups).sum().max()) if len(underwater) else 0


def _maximum_recovery_sessions(equity: pd.Series) -> int:
    """Longest trough-to-recovery duration; open episodes end at the sample end."""

    values = equity.to_numpy(dtype=float)
    if len(values) == 0:
        return 0
    running_peak = np.maximum.accumulate(values)
    underwater = values < running_peak - 1e-12
    durations: list[int] = []
    start = 0
    while start < len(values):
        if not underwater[start]:
            start += 1
            continue
        end = start
        while end + 1 < len(values) and underwater[end + 1]:
            end += 1
        trough = start + int(np.argmin(values[start : end + 1] / running_peak[start : end + 1]))
        recovery = min(end + 1, len(values) - 1)
        durations.append(recovery - trough)
        start = end + 1
    return max(durations, default=0)


def performance_metrics(result: BacktestResult, benchmark_returns: pd.Series) -> dict[str, float]:
    returns = result.returns.dropna()
    benchmark = benchmark_returns.reindex(returns.index).dropna()
    returns = returns.reindex(benchmark.index)
    if len(returns) < 2:
        raise DataIntegrityError("not enough returns for metrics")
    years = len(returns) / 252.0
    terminal = float((1.0 + returns).prod())
    benchmark_terminal = float((1.0 + benchmark).prod())
    cagr = terminal ** (1.0 / years) - 1.0
    benchmark_cagr = benchmark_terminal ** (1.0 / years) - 1.0
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    volatility = float(returns.std(ddof=0) * np.sqrt(252.0))
    downside = returns.where(returns < 0.0, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(252.0))
    arithmetic_return = float(returns.mean() * 252.0)
    max_drawdown = float(drawdown.min())

    def worst_period(window: int) -> float:
        if len(returns) < window:
            return float("nan")
        return float((1.0 + returns).rolling(window).apply(np.prod, raw=True).min() - 1.0)

    return {
        "cagr": float(cagr),
        "terminal_wealth": terminal,
        "benchmark_cagr": float(benchmark_cagr),
        "benchmark_terminal_wealth": benchmark_terminal,
        "annualized_excess_return": float(cagr - benchmark_cagr),
        "cumulative_excess_wealth": float(terminal - benchmark_terminal),
        "maximum_drawdown": max_drawdown,
        "annualized_volatility": volatility,
        "sharpe": arithmetic_return / volatility if volatility > 0 else float("nan"),
        "sortino": arithmetic_return / downside_deviation if downside_deviation > 0 else float("nan"),
        "calmar": cagr / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        "downside_deviation": downside_deviation,
        "ulcer_index": float(np.sqrt(np.mean(np.square(drawdown)))),
        "worst_month_21": worst_period(21),
        "worst_3_month_63": worst_period(63),
        "worst_12_month_252": worst_period(252),
        "time_underwater_sessions": float(_maximum_underwater_sessions(equity)),
        "maximum_recovery_sessions": float(_maximum_recovery_sessions(equity)),
        "annualized_turnover": float(result.turnover.sum() / years),
        "total_transaction_cost": float(result.costs.sum()),
        "observations": float(len(returns)),
    }


def rolling_excess(
    strategy_returns: pd.Series, benchmark_returns: pd.Series, years: int
) -> pd.Series:
    window = int(252 * years)
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1
    ).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float, name=f"rolling_{years}y_excess")
    strategy_log = np.log1p(aligned["strategy"]).rolling(window).sum()
    benchmark_log = np.log1p(aligned["benchmark"]).rolling(window).sum()
    strategy_cagr = np.expm1(strategy_log * 252.0 / window)
    benchmark_cagr = np.expm1(benchmark_log * 252.0 / window)
    return (strategy_cagr - benchmark_cagr).dropna().rename(f"rolling_{years}y_excess")


def rolling_summary(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for years in (3, 5, 10):
        values = rolling_excess(strategy_returns, benchmark_returns, years)
        rows.append(
            {
                "years": years,
                "windows": len(values),
                "median_excess": float(values.median()) if len(values) else float("nan"),
                "worst_excess": float(values.min()) if len(values) else float("nan"),
                "best_excess": float(values.max()) if len(values) else float("nan"),
                "win_rate": float(values.gt(0).mean()) if len(values) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def allocator_grid(config: AllocationConfig) -> tuple[AllocatorSpec, ...]:
    return tuple(
        AllocatorSpec(method, cadence, float(change))
        for method, cadence, change in product(
            config.methods, config.cadences, config.minimum_changes
        )
    )


def select_allocator(
    signals: pd.DataFrame,
    decision_returns: pd.DataFrame,
    config: AllocationConfig,
) -> tuple[AllocatorSpec, BacktestResult, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    results: dict[str, tuple[AllocatorSpec, BacktestResult]] = {}
    benchmark = decision_returns["VOO"]
    for spec in allocator_grid(config):
        weights = allocate(signals, config, spec)
        result = backtest(weights, decision_returns, transaction_cost_bps=config.transaction_cost_bps)
        metrics = performance_metrics(result, benchmark)
        rows.append({"name": spec.name, "method": spec.method, "cadence": spec.cadence, "minimum_change": spec.minimum_change, **metrics})
        results[spec.name] = (spec, result)
    table = pd.DataFrame(rows)
    best_cagr = float(table["cagr"].max())
    tolerance = config.selection_tolerance_annual_bps / 10_000.0
    finalists = table[table["cagr"] >= best_cagr - tolerance].copy()
    finalists["cadence_rank"] = finalists["cadence"].map({"weekly": 0, "daily": 1})
    finalists["method_rank"] = finalists["method"].map(
        {"continuous": 0, "probability": 1, "bucketed": 2}
    )
    winner = finalists.sort_values(
        ["annualized_turnover", "cadence_rank", "method_rank", "name"]
    ).iloc[0]
    spec, result = results[str(winner["name"])]
    table["selected"] = table["name"].eq(spec.name)
    return spec, result, table.sort_values("cagr", ascending=False).reset_index(drop=True)


def benchmark_table(
    decision_returns: pd.DataFrame, transaction_cost_bps: float
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    rows: list[dict[str, float | str]] = []
    results: dict[str, BacktestResult] = {}
    voo = decision_returns["VOO"]
    for weight in (1.0, 0.0, 0.5, 0.6, 0.7, 0.8):
        name = "VOO" if weight == 1.0 else "KMLM" if weight == 0.0 else f"static_{int(weight * 100)}_{int((1-weight)*100)}"
        result = static_mix_backtest(
            decision_returns,
            weight,
            transaction_cost_bps=transaction_cost_bps,
            rebalance="monthly",
        )
        results[name] = result
        rows.append({"strategy": name, **performance_metrics(result, voo)})
    return pd.DataFrame(rows).sort_values("cagr", ascending=False).reset_index(drop=True), results


def execution_ledger(
    result: BacktestResult,
    decision_returns: pd.DataFrame,
    execution_dates: pd.Series | None = None,
) -> pd.DataFrame:
    """One auditable row per assumed execution/earned return period."""

    dates = result.returns.index
    ledger = pd.DataFrame(index=dates)
    ledger.index.name = "signal_date"
    if execution_dates is not None:
        ledger["execution_date"] = pd.to_datetime(
            execution_dates.reindex(dates), errors="coerce"
        )
    ledger["executed_voo_weight"] = result.weights.reindex(dates)["VOO"]
    ledger["executed_kmlm_weight"] = result.weights.reindex(dates)["KMLM"]
    ledger["turnover"] = result.turnover.reindex(dates)
    ledger["transaction_cost"] = result.costs.reindex(dates)
    ledger["gross_return"] = result.gross_returns.reindex(dates)
    ledger["net_return"] = result.returns.reindex(dates)
    ledger["voo_benchmark_return"] = decision_returns["VOO"].reindex(dates)
    ledger["kmlm_return"] = decision_returns["KMLM"].reindex(dates)
    return ledger
