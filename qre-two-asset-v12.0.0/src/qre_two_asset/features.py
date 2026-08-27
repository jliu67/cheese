"""Backward-looking VOO/KMLM features with a frozen 40-column ceiling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import MarketData, PointInTimePanel
from .errors import DataIntegrityError
from .splitters import TimeFold

FEATURE_VERSION = "12.0"


@dataclass(frozen=True)
class FeatureSet:
    values: pd.DataFrame
    variants: dict[str, tuple[str, ...]]
    external_available: bool

    def for_variant(self, variant: str) -> pd.DataFrame:
        if variant not in self.variants:
            raise DataIntegrityError(f"unknown feature variant: {variant}")
        return self.values.loc[:, list(self.variants[variant])]


def _log_return(level: pd.Series, periods: int) -> pd.Series:
    clean = np.log(pd.to_numeric(level, errors="coerce"))
    return clean.diff(periods)


def _vol(level: pd.Series, window: int, downside: bool = False) -> pd.Series:
    returns = np.log(pd.to_numeric(level, errors="coerce")).diff()
    if downside:
        returns = returns.where(returns < 0.0, 0.0)
    return returns.rolling(window, min_periods=max(10, window // 2)).std(ddof=0) * np.sqrt(252.0)


def _drawdown(level: pd.Series, window: int = 252) -> pd.Series:
    clean = pd.to_numeric(level, errors="coerce")
    peak = clean.rolling(window, min_periods=max(63, window // 2)).max()
    return clean / peak - 1.0


def _ma_gap(level: pd.Series, window: int) -> pd.Series:
    clean = pd.to_numeric(level, errors="coerce")
    mean = clean.rolling(window, min_periods=max(20, window // 2)).mean()
    return clean / mean - 1.0


def _downside_frequency(level: pd.Series, window: int = 21) -> pd.Series:
    returns = pd.to_numeric(level, errors="coerce").pct_change(fill_method=None)
    return returns.lt(0.0).rolling(window, min_periods=window).mean()


def _trailing_z(series: pd.Series, window: int = 252) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    mean = clean.rolling(window, min_periods=max(63, window // 2)).mean()
    std = clean.rolling(window, min_periods=max(63, window // 2)).std(ddof=0).replace(0, np.nan)
    return ((clean - mean) / std).clip(-8.0, 8.0)


def build_features(market: MarketData, external: PointInTimePanel | None = None) -> FeatureSet:
    levels = market.levels
    voo = levels["VOO"]
    kmlm = levels["KMLM"]
    columns: dict[str, pd.Series] = {}

    # Model A: VOO only (12 columns).
    for horizon in (21, 63, 126, 252):
        columns[f"voo__log_return_{horizon}"] = _log_return(voo, horizon)
    columns["voo__vol_21"] = _vol(voo, 21)
    columns["voo__vol_63"] = _vol(voo, 63)
    columns["voo__downside_vol_63"] = _vol(voo, 63, downside=True)
    columns["voo__drawdown_252"] = _drawdown(voo)
    columns["voo__ma_gap_63"] = _ma_gap(voo, 63)
    columns["voo__ma_gap_200"] = _ma_gap(voo, 200)
    columns["voo__downside_frequency_21"] = _downside_frequency(voo)
    columns["voo__momentum_accel_21_63"] = _log_return(voo, 21) - _log_return(voo, 63) / 3.0
    a_columns = tuple(columns)

    # Model B: add KMLM behavior (11 columns, 23 cumulative).
    for horizon in (21, 63, 126, 252):
        columns[f"kmlm__log_return_{horizon}"] = _log_return(kmlm, horizon)
    columns["kmlm__vol_21"] = _vol(kmlm, 21)
    columns["kmlm__vol_63"] = _vol(kmlm, 63)
    columns["kmlm__downside_vol_63"] = _vol(kmlm, 63, downside=True)
    columns["kmlm__drawdown_252"] = _drawdown(kmlm)
    columns["kmlm__ma_gap_63"] = _ma_gap(kmlm, 63)
    columns["kmlm__ma_gap_200"] = _ma_gap(kmlm, 200)
    columns["kmlm__downside_frequency_21"] = _downside_frequency(kmlm)
    b_columns = tuple(columns)

    # Model C: add relative state (12 columns, 35 cumulative).
    relative_log_level = np.log(voo) - np.log(kmlm)
    relative_daily = relative_log_level.diff()
    for horizon in (21, 63, 126, 252):
        columns[f"relative__log_return_{horizon}"] = relative_log_level.diff(horizon)
    columns["relative__ratio_slope_63"] = relative_log_level.diff(62) / 62.0
    columns["relative__ratio_slope_126"] = relative_log_level.diff(125) / 125.0
    columns["relative__vol_63"] = relative_daily.rolling(63, min_periods=32).std(ddof=0) * np.sqrt(252.0)
    voo_daily = np.log(voo).diff()
    kmlm_daily = np.log(kmlm).diff()
    columns["relative__correlation_63"] = voo_daily.rolling(63, min_periods=32).corr(kmlm_daily)
    columns["relative__correlation_change_21"] = columns["relative__correlation_63"].diff(21)
    ratio_level = np.exp(relative_log_level)
    columns["relative__drawdown_252"] = _drawdown(ratio_level)
    relative_vol = columns["relative__vol_63"].replace(0.0, np.nan)
    columns["relative__risk_adjusted_momentum_63"] = relative_log_level.diff(63) / relative_vol
    columns["relative__risk_adjusted_momentum_126"] = relative_log_level.diff(126) / relative_vol
    c_columns = tuple(columns)

    # Model D: exactly five optional external risk columns (40 cumulative).
    external_available = False
    external_values = external.values.reindex(levels.index) if external is not None else pd.DataFrame(index=levels.index)
    required_external = {"VIX", "HY_OAS", "NFCI"}
    if required_external.issubset(external_values.columns):
        vix = external_values["VIX"]
        hy = external_values["HY_OAS"]
        nfci = external_values["NFCI"]
        columns["external__vix_z_252"] = _trailing_z(vix)
        columns["external__vix_change_21"] = vix.diff(21)
        columns["external__hy_oas_z_252"] = _trailing_z(hy)
        columns["external__hy_oas_change_21"] = hy.diff(21)
        columns["external__nfci_change_20"] = nfci.diff(20)
        coverage = pd.DataFrame({key: columns[key] for key in list(columns)[-5:]}).notna().mean().min()
        external_available = bool(coverage >= 0.50)
    else:
        for name in (
            "external__vix_z_252",
            "external__vix_change_21",
            "external__hy_oas_z_252",
            "external__hy_oas_change_21",
            "external__nfci_change_20",
        ):
            columns[name] = pd.Series(np.nan, index=levels.index, dtype=float)
    d_columns = tuple(columns)

    values = pd.DataFrame(columns, index=levels.index).replace([np.inf, -np.inf], np.nan)
    variants = {"A": a_columns, "B": b_columns, "C": c_columns, "D": d_columns}
    if len(d_columns) != 40:
        raise AssertionError(f"frozen feature count changed: {len(d_columns)}")
    return FeatureSet(values=values, variants=variants, external_available=external_available)


def audit_feature_causality(
    original_market: MarketData,
    mutated_market: MarketData,
    cutoff: str | pd.Timestamp,
    original_external: PointInTimePanel | None = None,
    mutated_external: PointInTimePanel | None = None,
) -> None:
    """Future mutations must not alter any feature at or before the cutoff."""

    boundary = pd.Timestamp(cutoff).normalize()
    before = build_features(original_market, original_external).values.loc[:boundary]
    after = build_features(mutated_market, mutated_external).values.loc[:boundary]
    pd.testing.assert_frame_equal(before, after, check_exact=False, rtol=1e-12, atol=1e-12)


def feature_fold_stability(
    features: pd.DataFrame, folds: list[TimeFold]
) -> pd.DataFrame:
    """Availability and robust location/scale by outer training fold."""

    rows: list[dict[str, float | int | str]] = []
    for fold in folds:
        frame = features.reindex(fold.train_dates)
        for name in frame.columns:
            values = frame[name].dropna()
            rows.append(
                {
                    "fold": fold.fold_id,
                    "train_start": fold.train_start.date().isoformat(),
                    "train_end": fold.train_end.date().isoformat(),
                    "feature": name,
                    "observations": len(values),
                    "coverage": float(frame[name].notna().mean()),
                    "median": float(values.median()) if len(values) else float("nan"),
                    "iqr": (
                        float(values.quantile(0.75) - values.quantile(0.25))
                        if len(values)
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def feature_redundancy(
    features: pd.DataFrame, *, absolute_correlation_threshold: float = 0.95
) -> pd.DataFrame:
    """List highly correlated feature pairs without using future/holdout rows."""

    correlation = features.corr(min_periods=126)
    rows: list[dict[str, float | str]] = []
    for left_index, left in enumerate(correlation.columns):
        for right in correlation.columns[left_index + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= absolute_correlation_threshold:
                rows.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "correlation": float(value),
                        "absolute_correlation": abs(float(value)),
                    }
                )
    columns = ["feature_1", "feature_2", "correlation", "absolute_correlation"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        "absolute_correlation", ascending=False
    ).reset_index(drop=True)
