"""Chronological expanding-window splits with exact label purging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import DataIntegrityError


@dataclass(frozen=True)
class TimeFold:
    fold_id: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex

    @property
    def train_start(self) -> pd.Timestamp:
        return self.train_dates[0]

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train_dates[-1]

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test_dates[0]

    @property
    def test_end(self) -> pd.Timestamp:
        return self.test_dates[-1]


def weekly_decision_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(index).sort_values().unique()
    series = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(series.groupby(dates.to_period("W-FRI")).last().to_numpy())


def purge_train_dates(
    candidate_train: pd.DatetimeIndex,
    outcome_end_dates: pd.Series,
    test_start: pd.Timestamp,
) -> pd.DatetimeIndex:
    realized = pd.to_datetime(outcome_end_dates.reindex(candidate_train), errors="coerce")
    keep = realized.notna() & realized.lt(pd.Timestamp(test_start))
    return candidate_train[keep.to_numpy()]


def expanding_calendar_folds(
    index: pd.DatetimeIndex,
    outcome_end_dates: pd.Series,
    *,
    minimum_train_years: int,
    test_years: int,
    step_years: int,
    final_boundary: str | pd.Timestamp | None = None,
) -> list[TimeFold]:
    dates = pd.DatetimeIndex(index).sort_values().unique()
    if final_boundary is not None:
        dates = dates[dates < pd.Timestamp(final_boundary).normalize()]
    if len(dates) < 504:
        raise DataIntegrityError("insufficient history for walk-forward evaluation")
    first_test = dates[0] + pd.DateOffset(years=minimum_train_years)
    first_candidates = dates[dates >= first_test]
    if len(first_candidates) == 0:
        raise DataIntegrityError("minimum training window consumes all available history")
    test_start = first_candidates[0]
    folds: list[TimeFold] = []
    fold_id = 0
    while test_start <= dates[-1]:
        test_end_limit = test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)
        test_dates = dates[(dates >= test_start) & (dates <= test_end_limit)]
        if len(test_dates) < 63:
            break
        candidate_train = dates[dates < test_start]
        train_dates = purge_train_dates(candidate_train, outcome_end_dates, test_dates[0])
        if len(train_dates) < 504:
            raise DataIntegrityError("purging leaves fewer than two years of training data")
        folds.append(TimeFold(fold_id, train_dates, test_dates))
        fold_id += 1
        next_limit = test_start + pd.DateOffset(years=step_years)
        future = dates[dates >= next_limit]
        if len(future) == 0:
            break
        test_start = future[0]
    if len(folds) < 3:
        raise DataIntegrityError("walk-forward protocol requires at least three outer folds")
    return folds


def inner_prequential_folds(
    index: pd.DatetimeIndex,
    outcome_end_dates: pd.Series,
    *,
    n_splits: int,
    minimum_train: int | None = None,
) -> list[TimeFold]:
    dates = pd.DatetimeIndex(index).sort_values().unique()
    if minimum_train is None:
        minimum_train = max(104, len(dates) // 2)
    minimum_train = min(minimum_train, max(52, len(dates) - n_splits * 26))
    remaining = len(dates) - minimum_train
    test_size = max(26, remaining // n_splits)
    folds: list[TimeFold] = []
    for fold_id in range(n_splits):
        test_begin = minimum_train + fold_id * test_size
        test_end = min(test_begin + test_size, len(dates))
        if test_begin >= len(dates) or test_end - test_begin < 13:
            continue
        test_dates = dates[test_begin:test_end]
        candidate_train = dates[:test_begin]
        train_dates = purge_train_dates(candidate_train, outcome_end_dates, test_dates[0])
        if len(train_dates) < 52:
            continue
        folds.append(TimeFold(fold_id, train_dates, test_dates))
    if len(folds) < 2:
        raise DataIntegrityError("not enough matured observations for inner prequential folds")
    return folds


def assert_fold_integrity(folds: list[TimeFold], outcome_end_dates: pd.Series) -> None:
    for fold in folds:
        if fold.train_end >= fold.test_start:
            raise DataIntegrityError("training dates overlap test dates")
        ends = pd.to_datetime(outcome_end_dates.reindex(fold.train_dates), errors="coerce")
        if ends.isna().any() or not ends.lt(fold.test_start).all():
            raise DataIntegrityError("training labels overlap the test information boundary")
        if not np.all(np.diff(fold.train_dates.view("i8")) > 0):
            raise DataIntegrityError("training dates must be strictly increasing")
