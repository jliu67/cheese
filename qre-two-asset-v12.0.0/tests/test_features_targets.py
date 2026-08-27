from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qre_two_asset.features import audit_feature_causality, build_features
from qre_two_asset.targets import build_targets, decision_period_returns
from tests.helpers import mutate_market_future, synthetic_market


class FeatureTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.market = synthetic_market()

    def test_feature_variants_have_frozen_sizes(self) -> None:
        features = build_features(self.market)
        self.assertEqual(
            {name: len(columns) for name, columns in features.variants.items()},
            {"A": 12, "B": 23, "C": 35, "D": 40},
        )
        self.assertFalse(features.external_available)

    def test_future_mutation_does_not_change_past_features(self) -> None:
        cutoff = self.market.dates[2000]
        audit_feature_causality(
            self.market, mutate_market_future(self.market, cutoff), cutoff
        )

    def test_target_starts_after_execution_lag(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=200)
        levels = pd.DataFrame(
            {
                "VOO": np.exp(np.arange(200) * 0.01),
                "KMLM": np.exp(np.arange(200) * 0.005),
            },
            index=dates,
        )
        target = build_targets(levels)[21]
        self.assertAlmostEqual(target.relative_log_return.iloc[0], (0.01 - 0.005) * 21)
        self.assertEqual(target.execution_date.iloc[0], dates[1])
        self.assertEqual(target.outcome_end_date.iloc[0], dates[22])

    def test_decision_return_is_first_return_after_next_close_execution(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=5)
        levels = pd.DataFrame(
            {"VOO": [100, 101, 103, 106, 110], "KMLM": [100] * 5}, index=dates
        )
        returns = decision_period_returns(levels, execution_lag=1)
        self.assertAlmostEqual(returns.loc[dates[0], "VOO"], 103 / 101 - 1)


if __name__ == "__main__":
    unittest.main()
