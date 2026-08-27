"""Block-bootstrap safeguards for overlapping financial observations and data snooping."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _circular_block_indices(
    length: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    blocks = int(np.ceil(length / block_length))
    starts = rng.integers(0, length, size=blocks)
    pieces = [(start + np.arange(block_length)) % length for start in starts]
    return np.concatenate(pieces)[:length]


def block_bootstrap_cagr_excess(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    block_length: int = 21,
    simulations: int = 1000,
    seed: int = 1729,
) -> dict[str, float]:
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1
    ).dropna()
    values = aligned.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(simulations, dtype=float)
    for simulation in range(simulations):
        sample = values[_circular_block_indices(len(values), block_length, rng)]
        strategy_growth = np.log1p(sample[:, 0]).mean() * 252.0
        benchmark_growth = np.log1p(sample[:, 1]).mean() * 252.0
        estimates[simulation] = np.expm1(strategy_growth) - np.expm1(benchmark_growth)
    return {
        "estimate": float(
            np.expm1(np.log1p(values[:, 0]).mean() * 252.0)
            - np.expm1(np.log1p(values[:, 1]).mean() * 252.0)
        ),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "probability_positive": float(np.mean(estimates > 0.0)),
        "block_length": float(block_length),
        "simulations": float(simulations),
    }


def reality_check(
    candidate_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    block_length: int = 21,
    simulations: int = 1000,
    seed: int = 1729,
) -> dict[str, float | str]:
    aligned = candidate_returns.join(benchmark_returns.rename("benchmark"), how="inner").dropna()
    candidates = aligned.drop(columns="benchmark")
    if candidates.empty:
        return {"pvalue": float("nan"), "winner": "", "observed_statistic": float("nan")}
    excess = candidates.sub(aligned["benchmark"], axis=0).to_numpy(dtype=float)
    means = excess.mean(axis=0)
    winner_index = int(np.argmax(means))
    observed = float(np.sqrt(len(excess)) * means[winner_index])
    centered = excess - means
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(simulations, dtype=float)
    for simulation in range(simulations):
        sample = centered[_circular_block_indices(len(centered), block_length, rng)]
        bootstrap[simulation] = float(np.sqrt(len(sample)) * sample.mean(axis=0).max())
    return {
        "pvalue": float((1 + np.sum(bootstrap >= observed)) / (simulations + 1)),
        "winner": str(candidates.columns[winner_index]),
        "observed_statistic": observed,
        "candidate_count": float(candidates.shape[1]),
        "block_length": float(block_length),
        "simulations": float(simulations),
    }


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    valid = pvalues.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype=float)
    running = 0.0
    count = len(valid)
    for rank, (name, value) in enumerate(valid.items()):
        candidate = min(1.0, float(value) * (count - rank))
        running = max(running, candidate)
        adjusted.loc[name] = running
    return adjusted
