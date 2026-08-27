"""Decision-aligned forward relative-return targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import DataIntegrityError


@dataclass(frozen=True)
class HorizonTarget:
    horizon: int
    relative_log_return: pd.Series
    voo_log_return: pd.Series
    kmlm_log_return: pd.Series
    voo_outperforms: pd.Series
    execution_date: pd.Series
    outcome_end_date: pd.Series


def _future_dates(index: pd.DatetimeIndex, offset: int) -> pd.Series:
    values = np.full(len(index), np.datetime64("NaT"), dtype="datetime64[ns]")
    if offset < len(index):
        values[:-offset] = index.to_numpy()[offset:]
    return pd.Series(values, index=index, dtype="datetime64[ns]")


def build_targets(
    levels: pd.DataFrame, horizons: tuple[int, ...] = (21, 63, 126), execution_lag: int = 1
) -> dict[int, HorizonTarget]:
    if tuple(horizons) != (21, 63, 126):
        raise DataIntegrityError("target horizons are frozen at 21/63/126 sessions")
    if execution_lag < 1:
        raise DataIntegrityError("execution lag must be at least one session")
    if not {"VOO", "KMLM"}.issubset(levels.columns):
        raise DataIntegrityError("levels require VOO and KMLM")
    log_levels = np.log(levels[["VOO", "KMLM"]].astype(float))
    output: dict[int, HorizonTarget] = {}
    for horizon in horizons:
        start = log_levels.shift(-execution_lag)
        end = log_levels.shift(-(execution_lag + horizon))
        forward = end - start
        relative = (forward["VOO"] - forward["KMLM"]).rename(f"relative_{horizon}")
        direction = relative.gt(0.0).where(relative.notna()).astype("boolean")
        output[horizon] = HorizonTarget(
            horizon=horizon,
            relative_log_return=relative,
            voo_log_return=forward["VOO"].rename(f"voo_{horizon}"),
            kmlm_log_return=forward["KMLM"].rename(f"kmlm_{horizon}"),
            voo_outperforms=direction,
            execution_date=_future_dates(levels.index, execution_lag),
            outcome_end_date=_future_dates(levels.index, execution_lag + horizon),
        )
    return output


def decision_period_returns(levels: pd.DataFrame, execution_lag: int = 1) -> pd.DataFrame:
    """Return first earned after a next-close execution, indexed by signal date."""

    if execution_lag < 1:
        raise DataIntegrityError("execution lag must be at least one session")
    forward_one = levels.shift(-1) / levels - 1.0
    return forward_one.shift(-execution_lag).loc[:, ["VOO", "KMLM"]]
