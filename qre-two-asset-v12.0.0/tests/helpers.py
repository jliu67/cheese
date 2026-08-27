from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from qre_two_asset.data import MarketData, ProvenanceSegment


def synthetic_market(periods: int = 3600) -> MarketData:
    dates = pd.bdate_range("1990-01-02", periods=periods)
    rng = np.random.default_rng(42)
    cycle = np.sin(np.arange(len(dates)) / 180.0)
    voo_returns = 0.00035 + 0.00045 * cycle + rng.normal(0.0, 0.009, len(dates))
    kmlm_returns = 0.00015 - 0.00030 * cycle + rng.normal(0.0, 0.007, len(dates))
    levels = pd.DataFrame(
        {
            "VOO": 100.0 * np.exp(np.cumsum(voo_returns)),
            "KMLM": 100.0 * np.exp(np.cumsum(kmlm_returns)),
        },
        index=dates,
    )
    returns = levels.pct_change(fill_method=None)
    segments = (
        ProvenanceSegment(
            "VOO", dates[0], None, "test", "VOO_TEST", "synthetic_test",
            "net_total_return", 0.0, True, False,
        ),
        ProvenanceSegment(
            "KMLM", dates[0], None, "test", "KMLM_TEST", "synthetic_test",
            "net_total_return", 0.0, True, False,
        ),
    )
    return MarketData(levels.copy(), levels, returns, segments, False, ("synthetic",))


def mutate_market_future(market: MarketData, cutoff: pd.Timestamp) -> MarketData:
    levels = market.levels.copy()
    levels.loc[levels.index > cutoff, "VOO"] *= np.linspace(
        1.0, 2.0, (levels.index > cutoff).sum()
    )
    return replace(
        market,
        raw_levels=levels.copy(),
        levels=levels,
        returns=levels.pct_change(fill_method=None),
    )
