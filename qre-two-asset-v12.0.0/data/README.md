# Required research data

The package intentionally ships without a fabricated VOO/KMLM backtest.

Create `market_levels.csv` from continuous, positive, daily **total-return levels**:

```csv
date,VOO,KMLM
1988-01-04,100.000000,100.000000
1988-01-05,100.420000,99.910000
```

The two columns may cross ETF/index source boundaries only after the source returns have
been stitched into one continuous level. The loader rejects large discontinuities but
cannot prove a smaller splice is correct; reconcile every boundary against provider data.

Copy `provenance.template.yaml` to `provenance.yaml` and replace every placeholder with
the actual provider, series, dates, return basis, fee adjustment, and backfill status.
Proxy and synthetic segments are always diagnostic-only. To permit a primary alpha
claim, every segment must be an official index or live ETF history explicitly marked
`primary_claim_allowed: true` on the strength of its provider documentation.

Optional point-in-time external data use the long-form layout in
`macro_pit.template.csv`. Current-vintage macro histories are not acceptable for a
historical claim.
