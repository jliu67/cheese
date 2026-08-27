# Data contract and provenance

## Market levels

`market_levels.csv` must contain:

| Column | Type | Meaning |
|---|---|---|
| `date` | unique trading date | Common decision calendar |
| `VOO` | positive number | Continuous total-return level for VOO economics |
| `KMLM` | positive number | Continuous total-return level for KMLM economics |

The level scale is arbitrary; returns are not. Dividends/distributions must be included.
Rows must be sorted uniquely after parsing, both assets must be complete, and unexplained
calendar gaps or one-day moves above 30% are fatal. Provider source transitions must be
return-stitched before ingestion; the engine does not rescale or splice them.

## Provenance YAML

Every date for each asset must be covered by exactly one provenance segment. Required
fields are:

| Field | Meaning |
|---|---|
| `start`, `end` | Inclusive source coverage; `end: null` means latest |
| `source_name`, `source_series` | Provider and exact identifier |
| `history_kind` | `live_etf`, `official_index`, `proxy`, or `synthetic_test` |
| `return_basis` | `adjusted_total_return`, `gross_total_return`, or `net_total_return` |
| `expense_adjustment_bps` | Annual fee drag applied to gross index returns |
| `hypothetical` | Whether provider history is hypothetical/backfilled |
| `primary_claim_allowed` | Explicit evidence-governance permission |
| `notes` | Methodology, licensing, and caveats |

Gross index history requires a positive fee adjustment. Proxy and synthetic history can
never set `primary_claim_allowed: true`. The final holdout defaults to requiring every
overlapping KMLM segment to be `live_etf`.

`primary_claim_allowed` is deliberately conservative. It is not inferred from a ticker or
URL. Set it true only after documenting source authority, total-return basis, fees,
inception, backfill status, and stitch methodology. The code reports warnings but does not
convert an unapproved history into approved evidence.

## Defensible historical roles

- **VOO since inception:** adjusted total-return ETF series.
- **VOO before inception:** official S&P 500 total-return history, with VOO-equivalent fee
  drag if the input is gross.
- **KMLM final holdout:** adjusted total-return live ETF history beginning 2020-12-02.
- **KMLM earlier research:** official KFA/MLM index history if licensed/authorized and
  fully documented, with KMLM fee drag. Any other trend-following series remains a proxy.

Do not relabel the MLM Index, MLM Index EV, a CTA index, or a reconstructed trend rule as
live KMLM. The KMLM provider itself describes multiple historical index eras, so source
segments should preserve those distinctions rather than flattening them into an ETF label.

## Point-in-time external data

`macro_pit.csv` uses long form:

| Column | Meaning |
|---|---|
| `series_id` | `VIX`, `HY_OAS`, or `NFCI` |
| `observation_date` | Economic/market observation date |
| `release_date` | First public availability date |
| `vintage_start` | Date this recorded value became the active vintage |
| `vintage_end` | Last date it remained active; blank if open |
| `value` | Numeric value |
| `source`, `source_series` | Provider audit identifiers |

The PIT replay chooses, for every decision date, the latest observation whose release and
vintage start are not in the future and whose vintage has not expired. It records the
selected observation date, release date, and release age. Conflicting duplicate vintages,
future observations, and future releases are fatal.

The official [FRED real-time-period documentation](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)
and [vintage-date endpoint](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html)
describe the concepts required for revision-aware histories. A current download is not a
historical vintage database.

## Optional risk series

- VIX: use an authorized daily close with same-day availability documented; see Cboe's
  [VIX historical data page](https://www.cboe.com/tradable_products/vix/vix_historical_data/).
- HY OAS: FRED series `BAMLH0A0HYM2`, with its publication timing represented explicitly.
- NFCI: Chicago Fed National Financial Conditions Index, including its weekly release lag
  and revisions where applicable.

Variant D requires all three. Its five transforms are trailing VIX z-score and 21-session
change, trailing HY OAS z-score and 21-session change, and 20-session NFCI change.

## Dataset versioning

The engine hashes the normalized raw level frame and the provenance file. A frozen
candidate refuses evaluation if either pre-holdout history or provenance changes. Source
licenses and provider retrieval timestamps should additionally be recorded in `notes` or
an external data manifest maintained with the licensed data.
