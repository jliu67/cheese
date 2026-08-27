from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qre_two_asset.config import AllocationConfig
from qre_two_asset.portfolio import (
    AllocatorSpec,
    allocate,
    backtest,
    performance_metrics,
    rolling_summary,
)


def _signals(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "allocation_score": np.linspace(-1, 1, len(dates)),
            "probability_voo_outperforms": np.linspace(0.1, 0.9, len(dates)),
        },
        index=dates,
    )


class PortfolioTests(unittest.TestCase):
    def test_allocator_is_fully_invested_and_bounded(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=100)
        weights = allocate(
            _signals(dates), AllocationConfig(),
            AllocatorSpec("continuous", "weekly", 0.05),
        )
        self.assertTrue(np.allclose(weights.sum(axis=1), 1.0))
        self.assertTrue(((weights >= 0) & (weights <= 1)).all().all())

    def test_transaction_cost_reduces_wealth(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=100)
        weights = allocate(
            _signals(dates), AllocationConfig(),
            AllocatorSpec("probability", "daily", 0.0),
        )
        returns = pd.DataFrame({"VOO": 0.001, "KMLM": 0.0005}, index=dates)
        free = backtest(weights, returns, transaction_cost_bps=0)
        costly = backtest(weights, returns, transaction_cost_bps=20)
        self.assertLess(costly.equity.iloc[-1], free.equity.iloc[-1])

    def test_metrics_keep_cagr_primary(self) -> None:
        dates = pd.bdate_range("2000-01-03", periods=3000)
        returns = pd.DataFrame({"VOO": 0.00035, "KMLM": 0.0001}, index=dates)
        weights = pd.DataFrame({"VOO": 0.8, "KMLM": 0.2}, index=dates)
        result = backtest(weights, returns, transaction_cost_bps=0)
        metrics = performance_metrics(result, returns["VOO"])
        self.assertLess(metrics["annualized_excess_return"], 0)
        rolling = rolling_summary(result.returns, returns["VOO"])
        self.assertEqual(rolling.loc[rolling["years"] == 3, "win_rate"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
