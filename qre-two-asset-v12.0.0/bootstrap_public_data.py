#!/usr/bin/env python3
"""Bootstrap QRE v12 with immediately-available public research data.

VOO side: SPY Yahoo Finance auto-adjusted daily prices before VOO inception,
          then VOO Yahoo Finance auto-adjusted daily prices thereafter.
KMLM:     public KMLMX_rev.csv proxy before KMLM inception,
       Yahoo Finance auto-adjusted KMLM thereafter.

The pre-inception KMLM history is deliberately marked as a proxy and
primary_claim_allowed=false.  This is suitable for getting QRE running and
for diagnostic/development research, not for claiming a definitive result.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

HOLDOUT = pd.Timestamp("2020-12-02")
PROXY_URL = (
    "https://gist.githubusercontent.com/chrswrmr/"
    "3a7522de7c9429e6dc2c8d29627e6e33/raw/"
    "510aeeaa5dc64dbd2e59ee5138ac66edff9018ca/KMLMX_rev.csv"
)


def download_adjusted_close(ticker: str, start: str) -> pd.Series:
    print(f"Downloading {ticker} from Yahoo Finance...")
    df = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")

    # yfinance versions differ on whether columns are flat or MultiIndex.
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df.xs("Close", axis=1, level=0)
        elif "Close" in df.columns.get_level_values(-1):
            close = df.xs("Close", axis=1, level=-1)
        else:
            raise RuntimeError(f"Could not find adjusted Close for {ticker}: {df.columns}")
        if isinstance(close, pd.DataFrame):
            if ticker in close.columns:
                close = close[ticker]
            else:
                close = close.iloc[:, 0]
    else:
        if "Close" not in df.columns:
            raise RuntimeError(f"Could not find adjusted Close for {ticker}: {df.columns}")
        close = df["Close"]

    close = pd.Series(close, copy=True).dropna().astype(float)
    idx = pd.to_datetime(close.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    close.index = pd.DatetimeIndex(idx).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close.name = ticker
    if (close <= 0).any():
        raise RuntimeError(f"{ticker} contains non-positive prices")
    return close


def download_kmlm_proxy() -> pd.Series:
    print("Downloading public pre-2020 KMLM proxy (KMLMX_rev.csv)...")
    r = requests.get(PROXY_URL, timeout=60)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text))
    if raw.shape[1] < 2:
        raise RuntimeError("KMLMX proxy CSV does not contain at least two columns")

    # The public file is date,value.  Use the first two columns so harmless
    # header/name changes do not break the bootstrap.
    proxy = raw.iloc[:, :2].copy()
    proxy.columns = ["date", "KMLM_PROXY"]
    proxy["date"] = pd.to_datetime(proxy["date"], errors="coerce")
    proxy["KMLM_PROXY"] = pd.to_numeric(proxy["KMLM_PROXY"], errors="coerce")
    proxy = proxy.dropna().drop_duplicates("date", keep="last").sort_values("date")
    proxy = proxy.set_index("date")["KMLM_PROXY"].astype(float)
    proxy.index = pd.DatetimeIndex(proxy.index).normalize()
    proxy = proxy[proxy > 0]

    if proxy.empty:
        raise RuntimeError("KMLMX proxy parsed to an empty series")
    if proxy.index.min() > pd.Timestamp("2012-12-01"):
        raise RuntimeError(
            f"Proxy starts too late ({proxy.index.min().date()}); QRE needs >=8 pre-holdout years"
        )
    if proxy.index.max() < pd.Timestamp("2020-11-20"):
        raise RuntimeError(
            f"Proxy ends too early ({proxy.index.max().date()}) to stitch to live KMLM"
        )
    if proxy.pct_change(fill_method=None).abs().max() > 0.30:
        raise RuntimeError("Proxy has a >30% one-day jump; refusing to feed it to QRE")

    print(
        f"  proxy coverage: {proxy.index.min().date()} -> {proxy.index.max().date()} "
        f"({len(proxy):,} rows)"
    )
    return proxy


def main() -> None:
    # Use SPY to extend the VOO/S&P-500 sleeve back to SPY inception, then
    # switch to actual VOO once VOO exists.  The VOO series is scaled at the
    # first shared SPY/VOO session so the stitched level is continuous.
    spy = download_adjusted_close("SPY", "1993-01-01")
    voo_live = download_adjusted_close("VOO", "2010-09-01")
    kmlm_live = download_adjusted_close("KMLM", "2020-12-01")
    proxy = download_kmlm_proxy()

    overlap = spy.index.intersection(voo_live.index)
    if overlap.empty:
        raise RuntimeError("SPY and VOO have no overlapping sessions for the stitch")
    voo_switch = overlap.min()

    spy_level = spy / spy.iloc[0] * 100.0
    voo_scaled = voo_live / voo_live.loc[voo_switch] * float(spy_level.loc[voo_switch])
    equity = pd.concat([
        spy_level.loc[spy_level.index < voo_switch],
        voo_scaled.loc[voo_scaled.index >= voo_switch],
    ]).sort_index()
    equity = equity[~equity.index.duplicated(keep="last")].rename("VOO")

    print(
        f"  SPY/VOO stitched coverage: {equity.index.min().date()} -> "
        f"{equity.index.max().date()} ({len(equity):,} rows); "
        f"VOO switch={voo_switch.date()}"
    )

    # Pre-holdout uses only dates shared by the SPY/VOO sleeve and KMLM proxy.
    pre_idx = equity.index.intersection(proxy.index)
    pre_idx = pre_idx[(pre_idx < HOLDOUT)]
    if len(pre_idx) < 8 * 240:
        raise RuntimeError(
            f"Only {len(pre_idx):,} shared pre-holdout sessions; not enough for QRE's 8-year minimum"
        )

    # Holdout/live era uses only dates shared by the equity sleeve and actual KMLM.
    live_idx = equity.index.intersection(kmlm_live.index)
    live_idx = live_idx[(live_idx >= HOLDOUT)]
    if len(live_idx) < 3 * 240:
        raise RuntimeError(
            f"Only {len(live_idx):,} shared live-KMLM sessions; QRE requires >=3 holdout years"
        )

    # Normalize proxy at 100.  Then return-stitch live KMLM onto the last proxy
    # level.  KMLM's first ETF observation has no prior ETF close, so the stitch
    # treats the transition observation as the anchor and uses live ETF returns
    # from the following session onward.
    proxy_level = proxy.loc[pre_idx] / proxy.loc[pre_idx].iloc[0] * 100.0
    anchor = float(proxy_level.iloc[-1])
    live_level = kmlm_live.loc[live_idx] / kmlm_live.loc[live_idx].iloc[0] * anchor
    kmlm = pd.concat([proxy_level.rename("KMLM"), live_level.rename("KMLM")])

    all_idx = pre_idx.append(live_idx)
    voo_level = equity.loc[all_idx]

    out = pd.DataFrame({"VOO": voo_level, "KMLM": kmlm.loc[all_idx]}, index=all_idx)
    out.index.name = "date"
    out = out.dropna().sort_index()

    jumps = out.pct_change(fill_method=None).abs()
    if (jumps > 0.30).any().any():
        where = (jumps > 0.30).stack()
        bad = where[where].index[0]
        raise RuntimeError(f">30% one-day jump after stitch at {bad}; refusing output")

    market_path = DATA / "market_levels.csv"
    out.to_csv(market_path, float_format="%.10f")

    first = out.index.min().date().isoformat()
    pre_end = out.index[out.index < HOLDOUT].max().date().isoformat()
    live_start = out.index[out.index >= HOLDOUT].min().date().isoformat()

    provenance = {
        "assets": {
            "VOO": [
                {
                    "start": first,
                    "end": (voo_switch - pd.Timedelta(days=1)).date().isoformat(),
                    "source_name": "Yahoo Finance",
                    "source_series": "SPY auto-adjusted daily Close",
                    "history_kind": "proxy",
                    "return_basis": "adjusted_total_return",
                    "expense_adjustment_bps": 0.0,
                    "hypothetical": True,
                    "primary_claim_allowed": False,
                    "notes": (
                        "SPY is used as the pre-inception proxy for the VOO/S&P 500 sleeve. "
                        "SPY and VOO both track the S&P 500, but their fees/tracking differ slightly."
                    ),
                },
                {
                    "start": voo_switch.date().isoformat(),
                    "end": None,
                    "source_name": "Yahoo Finance",
                    "source_series": "VOO auto-adjusted daily Close",
                    "history_kind": "live_etf",
                    "return_basis": "adjusted_total_return",
                    "expense_adjustment_bps": 0.0,
                    "hypothetical": False,
                    "primary_claim_allowed": False,
                    "notes": (
                        "Actual VOO ETF era. Public bootstrap source; verify against an authoritative "
                        "provider before a primary claim."
                    ),
                },
            ],
            "KMLM": [
                {
                    "start": first,
                    "end": pre_end,
                    "source_name": "Public GitHub Gist referenced by QuantConnect community",
                    "source_series": "KMLMX_rev.csv",
                    "history_kind": "proxy",
                    "return_basis": "net_total_return",
                    "expense_adjustment_bps": 0.0,
                    "hypothetical": True,
                    "primary_claim_allowed": False,
                    "notes": (
                        "Diagnostic pre-inception KMLM proxy. Public KMLMX history; provenance "
                        "is not authoritative enough for a primary alpha claim. Replace with official "
                        "KFA/MLM history when available."
                    ),
                },
                {
                    "start": live_start,
                    "end": None,
                    "source_name": "Yahoo Finance",
                    "source_series": "KMLM auto-adjusted daily Close",
                    "history_kind": "live_etf",
                    "return_basis": "adjusted_total_return",
                    "expense_adjustment_bps": 0.0,
                    "hypothetical": False,
                    "primary_claim_allowed": False,
                    "notes": (
                        "Actual KMLM ETF era. Public bootstrap source; verify against an authoritative "
                        "provider before a primary claim."
                    ),
                },
            ],
        }
    }
    prov_path = DATA / "provenance.yaml"
    with prov_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(provenance, f, sort_keys=False)

    print("\nDONE")
    print(f"  wrote: {market_path}")
    print(f"  wrote: {prov_path}")
    print(f"  QRE rows: {len(out):,}")
    print(f"  coverage: {out.index.min().date()} -> {out.index.max().date()}")
    print(f"  pre-holdout sessions: {(out.index < HOLDOUT).sum():,}")
    print(f"  live-KMLM sessions:   {(out.index >= HOLDOUT).sum():,}")
    print("\nNext command:")
    print("  qre2 --config config/research.yaml inspect-data")
    print("\nIMPORTANT: This gets QRE working now, but the pre-2020 KMLM segment is a proxy.")
    print("Do not treat a successful result as final/official until that proxy is replaced or validated.")


if __name__ == "__main__":
    main()
