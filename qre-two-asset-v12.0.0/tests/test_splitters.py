from __future__ import annotations

import unittest

import pandas as pd

from qre_two_asset.splitters import (
    assert_fold_integrity,
    expanding_calendar_folds,
    inner_prequential_folds,
    weekly_decision_dates,
)


class SplitterTests(unittest.TestCase):
    def test_outer_fold_purges_overlapping_labels(self) -> None:
        dates = pd.bdate_range("2000-01-03", periods=3200)
        outcomes = pd.Series(dates, index=dates).shift(-127)
        folds = expanding_calendar_folds(
            dates,
            outcomes,
            minimum_train_years=5,
            test_years=1,
            step_years=1,
        )
        self.assertGreaterEqual(len(folds), 3)
        assert_fold_integrity(folds, outcomes)

    def test_inner_folds_are_chronological(self) -> None:
        dates = weekly_decision_dates(pd.bdate_range("2000-01-03", periods=1800))
        outcomes = pd.Series(dates, index=dates).shift(-10)
        folds = inner_prequential_folds(dates, outcomes, n_splits=3)
        self.assertTrue(all(fold.train_end < fold.test_start for fold in folds))


if __name__ == "__main__":
    unittest.main()
