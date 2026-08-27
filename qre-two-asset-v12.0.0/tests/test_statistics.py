from __future__ import annotations

import unittest

import pandas as pd

from qre_two_asset.statistics import block_bootstrap_cagr_excess, reality_check


class StatisticsTests(unittest.TestCase):
    def test_bootstrap_and_reality_check_are_deterministic(self) -> None:
        dates = pd.bdate_range("2000-01-03", periods=1000)
        benchmark = pd.Series(0.0002, index=dates)
        candidates = pd.DataFrame({"good": 0.0004, "bad": 0.0001}, index=dates)
        first = reality_check(candidates, benchmark, simulations=100, seed=1)
        second = reality_check(candidates, benchmark, simulations=100, seed=1)
        self.assertEqual(first, second)
        interval = block_bootstrap_cagr_excess(
            candidates["good"], benchmark, simulations=100, seed=1
        )
        self.assertEqual(interval["probability_positive"], 1.0)


if __name__ == "__main__":
    unittest.main()
