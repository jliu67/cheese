"""Market/proxy provenance and point-in-time external data handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .errors import DataIntegrityError, LookAheadError

ASSETS = ("VOO", "KMLM")
PIT_REQUIRED_COLUMNS = {
    "series_id",
    "observation_date",
    "release_date",
    "vintage_start",
    "vintage_end",
    "value",
    "source",
    "source_series",
}
PIT_DATE_COLUMNS = ("observation_date", "release_date", "vintage_start", "vintage_end")
APPROVED_HISTORY_KINDS = {"live_etf", "official_index", "proxy", "synthetic_test"}


@dataclass(frozen=True)
class ProvenanceSegment:
    asset: str
    start: pd.Timestamp
    end: pd.Timestamp | None
    source_name: str
    source_series: str
    history_kind: str
    return_basis: str
    expense_adjustment_bps: float
    hypothetical: bool
    primary_claim_allowed: bool
    notes: str = ""


@dataclass(frozen=True)
class MarketData:
    raw_levels: pd.DataFrame
    levels: pd.DataFrame
    returns: pd.DataFrame
    segments: tuple[ProvenanceSegment, ...]
    primary_claim_allowed: bool
    warnings: tuple[str, ...]

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.levels.index

    def provenance_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "asset": segment.asset,
                    "start": segment.start.date().isoformat(),
                    "end": segment.end.date().isoformat() if segment.end is not None else "",
                    "source_name": segment.source_name,
                    "source_series": segment.source_series,
                    "history_kind": segment.history_kind,
                    "return_basis": segment.return_basis,
                    "expense_adjustment_bps": segment.expense_adjustment_bps,
                    "hypothetical": segment.hypothetical,
                    "primary_claim_allowed": segment.primary_claim_allowed,
                    "notes": segment.notes,
                }
                for segment in self.segments
            ]
        )


@dataclass(frozen=True)
class PointInTimePanel:
    values: pd.DataFrame
    observation_dates: pd.DataFrame
    release_dates: pd.DataFrame
    release_age_days: pd.DataFrame

    def audit(self) -> None:
        decision = pd.Series(self.values.index, index=self.values.index)
        for column in self.values.columns:
            visible = self.values[column].notna()
            if (self.observation_dates.loc[visible, column] > decision.loc[visible]).any():
                raise LookAheadError(f"{column}: future observation in point-in-time panel")
            if (self.release_dates.loc[visible, column] > decision.loc[visible]).any():
                raise LookAheadError(f"{column}: future release in point-in-time panel")


def _as_date(value: Any, name: str, *, optional: bool = False) -> pd.Timestamp | None:
    if optional and (value is None or str(value).strip() == ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise DataIntegrityError(f"invalid {name}: {value!r}")
    return pd.Timestamp(parsed).normalize()


def load_provenance(path: str | Path) -> tuple[ProvenanceSegment, ...]:
    source = Path(path)
    if not source.exists():
        raise DataIntegrityError(f"provenance file does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    assets = raw.get("assets")
    if not isinstance(assets, dict):
        raise DataIntegrityError("provenance must contain an assets mapping")
    segments: list[ProvenanceSegment] = []
    for asset in ASSETS:
        rows = assets.get(asset)
        if not isinstance(rows, list) or not rows:
            raise DataIntegrityError(f"provenance requires at least one {asset} segment")
        for row in rows:
            if not isinstance(row, dict):
                raise DataIntegrityError(f"{asset} provenance entries must be mappings")
            kind = str(row.get("history_kind", "")).strip()
            if kind not in APPROVED_HISTORY_KINDS:
                raise DataIntegrityError(f"{asset}: unsupported history_kind {kind!r}")
            basis = str(row.get("return_basis", "")).strip()
            if basis not in {"adjusted_total_return", "gross_total_return", "net_total_return"}:
                raise DataIntegrityError(f"{asset}: invalid return_basis {basis!r}")
            fee = float(row.get("expense_adjustment_bps", 0.0))
            if basis == "gross_total_return" and fee <= 0:
                raise DataIntegrityError(
                    f"{asset}: gross index history requires a positive expense adjustment"
                )
            segments.append(
                ProvenanceSegment(
                    asset=asset,
                    start=_as_date(row.get("start"), "start"),
                    end=_as_date(row.get("end"), "end", optional=True),
                    source_name=str(row.get("source_name", "")).strip(),
                    source_series=str(row.get("source_series", "")).strip(),
                    history_kind=kind,
                    return_basis=basis,
                    expense_adjustment_bps=fee,
                    hypothetical=bool(row.get("hypothetical", False)),
                    primary_claim_allowed=bool(row.get("primary_claim_allowed", False)),
                    notes=str(row.get("notes", "")).strip(),
                )
            )
    for segment in segments:
        if not segment.source_name or not segment.source_series:
            raise DataIntegrityError(f"{segment.asset}: source name and series are required")
        if segment.end is not None and segment.end < segment.start:
            raise DataIntegrityError(f"{segment.asset}: provenance end precedes start")
        if segment.history_kind in {"proxy", "synthetic_test"} and segment.primary_claim_allowed:
            raise DataIntegrityError(
                f"{segment.asset}: proxy/synthetic history cannot authorize a primary alpha claim"
            )
    return tuple(sorted(segments, key=lambda item: (item.asset, item.start)))


def _segment_mask(index: pd.DatetimeIndex, segment: ProvenanceSegment) -> np.ndarray:
    mask = index >= segment.start
    if segment.end is not None:
        mask &= index <= segment.end
    return np.asarray(mask, dtype=bool)


def _validate_coverage(index: pd.DatetimeIndex, segments: tuple[ProvenanceSegment, ...]) -> None:
    for asset in ASSETS:
        count = np.zeros(len(index), dtype=int)
        for segment in segments:
            if segment.asset == asset:
                count += _segment_mask(index, segment).astype(int)
        if (count == 0).any():
            missing = index[count == 0][:5].strftime("%Y-%m-%d").tolist()
            raise DataIntegrityError(f"{asset}: dates without provenance coverage: {missing}")
        if (count > 1).any():
            overlap = index[count > 1][:5].strftime("%Y-%m-%d").tolist()
            raise DataIntegrityError(f"{asset}: overlapping provenance segments: {overlap}")


def load_market_data(levels_path: str | Path, provenance_path: str | Path) -> MarketData:
    """Load continuous total-return levels and apply only documented index fee drag."""

    source = Path(levels_path)
    if not source.exists():
        raise DataIntegrityError(f"market data file does not exist: {source}")
    raw = pd.read_csv(source)
    required = {"date", *ASSETS}
    missing = required - set(raw.columns)
    if missing:
        raise DataIntegrityError(f"market data missing columns: {sorted(missing)}")
    frame = raw.loc[:, ["date", *ASSETS]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise DataIntegrityError("market data contain invalid dates")
    if frame["date"].duplicated().any():
        raise DataIntegrityError("market data contain duplicate dates")
    frame = frame.sort_values("date").set_index("date")
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    for asset in ASSETS:
        frame[asset] = pd.to_numeric(frame[asset], errors="coerce")
        if frame[asset].isna().any() or (frame[asset] <= 0).any():
            raise DataIntegrityError(f"{asset}: levels must be positive and complete")
    if len(frame) < 504:
        raise DataIntegrityError("market history must contain at least two trading years")
    calendar_gaps = frame.index.to_series().diff().dt.days.dropna()
    if (calendar_gaps > 10).any():
        dates = frame.index[1:][calendar_gaps.to_numpy() > 10][:5]
        raise DataIntegrityError(f"market history contains unexplained long gaps near {list(dates)}")

    raw_returns = frame.pct_change(fill_method=None)
    if (raw_returns.abs() > 0.30).any().any():
        bad = np.argwhere((raw_returns.abs() > 0.30).to_numpy())[0]
        raise DataIntegrityError(
            f"implausible one-day level jump for {raw_returns.columns[bad[1]]} "
            f"on {raw_returns.index[bad[0]].date()}; stitch source levels before use"
        )

    segments = load_provenance(provenance_path)
    _validate_coverage(frame.index, segments)
    log_returns = np.log(frame).diff()
    warnings: list[str] = []
    for segment in segments:
        mask = _segment_mask(frame.index, segment)
        if segment.expense_adjustment_bps > 0:
            daily_fee = (segment.expense_adjustment_bps / 10_000.0) / 252.0
            log_returns.loc[mask, segment.asset] -= daily_fee
        if segment.hypothetical:
            warnings.append(
                f"{segment.asset} {segment.start.date()}.."
                f"{segment.end.date() if segment.end is not None else 'latest'} is hypothetical"
            )
        if not segment.primary_claim_allowed:
            warnings.append(
                f"{segment.asset} segment {segment.source_series} is diagnostic-only for alpha claims"
            )
    net_levels = np.exp(log_returns.fillna(0.0).cumsum()) * 100.0
    net_levels.index = frame.index
    returns = net_levels.pct_change(fill_method=None)
    claim_allowed = all(segment.primary_claim_allowed for segment in segments)
    return MarketData(
        raw_levels=frame,
        levels=net_levels,
        returns=returns,
        segments=segments,
        primary_claim_allowed=claim_allowed,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def validate_holdout_provenance(
    market: MarketData, holdout_start: str | pd.Timestamp, require_kmlm_live_etf: bool
) -> None:
    start = pd.Timestamp(holdout_start).normalize()
    if start not in market.dates:
        future = market.dates[market.dates >= start]
        if len(future) == 0:
            raise DataIntegrityError("holdout begins after available market data")
        start = future[0]
    if not require_kmlm_live_etf:
        return
    for segment in market.segments:
        if segment.asset != "KMLM":
            continue
        overlap_end = segment.end if segment.end is not None else market.dates[-1]
        if overlap_end >= start and segment.start <= market.dates[-1]:
            if segment.history_kind != "live_etf":
                raise DataIntegrityError(
                    "final holdout must use live KMLM ETF history, not index/proxy history"
                )


def validate_pit_observations(frame: pd.DataFrame) -> pd.DataFrame:
    missing = PIT_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise DataIntegrityError(f"point-in-time input missing columns: {sorted(missing)}")
    clean = frame.copy()
    for column in PIT_DATE_COLUMNS:
        raw = clean[column]
        parsed = pd.to_datetime(raw, errors="coerce")
        invalid = raw.notna() & raw.astype(str).str.strip().ne("") & parsed.isna()
        if invalid.any():
            raise DataIntegrityError(f"invalid {column} values")
        if column != "vintage_end" and parsed.isna().any():
            raise DataIntegrityError(f"{column} may not be blank")
        clean[column] = parsed.dt.normalize()
    clean["series_id"] = clean["series_id"].astype(str).str.strip().str.upper()
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    if clean["value"].isna().any():
        raise DataIntegrityError("point-in-time values must be numeric and nonmissing")
    if (clean["release_date"] < clean["observation_date"]).any():
        raise DataIntegrityError("release date may not precede observation date")
    if (clean["vintage_start"] < clean["release_date"]).any():
        raise DataIntegrityError("vintage start may not precede release date")
    finite = clean["vintage_end"].notna()
    if (clean.loc[finite, "vintage_end"] < clean.loc[finite, "vintage_start"]).any():
        raise DataIntegrityError("vintage end may not precede vintage start")
    key = ["series_id", "observation_date", "vintage_start"]
    duplicate = clean.duplicated(key, keep=False)
    if duplicate.any():
        for _, group in clean.loc[duplicate].groupby(key, dropna=False):
            if group[["value", "release_date", "vintage_end"]].drop_duplicates().shape[0] > 1:
                raise DataIntegrityError("conflicting duplicate vintages")
        clean = clean.drop_duplicates(key, keep="first")
    return clean.sort_values(["series_id", "release_date", "observation_date"]).reset_index(drop=True)


def load_pit_data(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise DataIntegrityError(f"point-in-time data file does not exist: {source}")
    return validate_pit_observations(pd.read_csv(source))


def _series_asof(
    rows: pd.DataFrame, decision_dates: pd.DatetimeIndex
) -> tuple[pd.Series, pd.Series, pd.Series]:
    data = rows.copy().reset_index(drop=True)
    data["effective_date"] = data[["release_date", "vintage_start"]].max(axis=1)
    data["row_id"] = np.arange(len(data), dtype=int)
    starts = list(
        data.sort_values(["effective_date", "observation_date", "vintage_start"]).itertuples(
            index=False
        )
    )
    expiries = list(data[data["vintage_end"].notna()].sort_values("vintage_end").itertuples(index=False))
    start_pointer = 0
    end_pointer = 0
    active: dict[pd.Timestamp, object] = {}
    values: list[float] = []
    observations: list[pd.Timestamp | pd.NaT] = []
    releases: list[pd.Timestamp | pd.NaT] = []
    for decision in decision_dates:
        while end_pointer < len(expiries) and expiries[end_pointer].vintage_end < decision:
            expired = expiries[end_pointer]
            current = active.get(expired.observation_date)
            if current is not None and current.row_id == expired.row_id:
                del active[expired.observation_date]
            end_pointer += 1
        while start_pointer < len(starts) and starts[start_pointer].effective_date <= decision:
            candidate = starts[start_pointer]
            if candidate.observation_date > decision:
                raise LookAheadError("effective macro row contains a future observation")
            previous = active.get(candidate.observation_date)
            if previous is None or candidate.vintage_start >= previous.vintage_start:
                active[candidate.observation_date] = candidate
            start_pointer += 1
        if not active:
            values.append(np.nan)
            observations.append(pd.NaT)
            releases.append(pd.NaT)
            continue
        row = active[max(active)]
        values.append(float(row.value))
        observations.append(row.observation_date)
        releases.append(row.release_date)
    return (
        pd.Series(values, index=decision_dates, dtype=float),
        pd.Series(observations, index=decision_dates, dtype="datetime64[ns]"),
        pd.Series(releases, index=decision_dates, dtype="datetime64[ns]"),
    )


def build_pit_panel(frame: pd.DataFrame, decision_dates: pd.DatetimeIndex) -> PointInTimePanel:
    clean = validate_pit_observations(frame)
    dates = pd.DatetimeIndex(pd.to_datetime(decision_dates)).normalize().sort_values().unique()
    if dates.empty:
        raise DataIntegrityError("decision calendar may not be empty")
    values: dict[str, pd.Series] = {}
    observations: dict[str, pd.Series] = {}
    releases: dict[str, pd.Series] = {}
    for series_id, rows in clean.groupby("series_id", sort=True):
        value, observation, release = _series_asof(rows, dates)
        values[series_id] = value
        observations[series_id] = observation
        releases[series_id] = release
    value_frame = pd.DataFrame(values, index=dates)
    observation_frame = pd.DataFrame(observations, index=dates)
    release_frame = pd.DataFrame(releases, index=dates)
    date_matrix = pd.DataFrame(
        np.repeat(dates.to_numpy()[:, None], len(release_frame.columns), axis=1),
        index=dates,
        columns=release_frame.columns,
    )
    age = (date_matrix - release_frame).apply(lambda column: column.dt.days)
    panel = PointInTimePanel(value_frame, observation_frame, release_frame, age)
    panel.audit()
    return panel
