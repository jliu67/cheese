# Final untouched holdout protocol

## Boundary

The default final holdout starts **2020-12-02**, the live KMLM ETF era encoded by this
project's data template. The exact first trading row at or after the configured boundary is
used. A minimum of three years is required; the default available period is intentionally
not used during candidate development.

The holdout must use `history_kind: live_etf` for every KMLM provenance segment that
overlaps it. Pre-holdout KFA/MLM index economics may support development only when their
identity, methodology eras, hypothetical status, and fee treatment are explicit.

## What development can access

`qre2 develop` loads the market file to validate structure and source coverage, then slices
at the boundary before constructing point-in-time panels, features, targets, folds,
forecasts, allocations, or metrics. Only the pre-holdout slice enters the frozen bundle.

The freeze records:

- configuration digest;
- raw pre-holdout market hash;
- provenance-file hash;
- feature version and names;
- selected A/B/C/D variant;
- selected allocator and parameters;
- pre-holdout metrics, rolling windows, Reality Check, and bootstrap;
- exact model objects fitted only on mature pre-holdout labels.

Changing the configuration, pre-holdout history, or provenance after freeze blocks final
evaluation.

## One-time evaluation sequence

1. Require the exact acknowledgement string.
2. Refuse if `FINAL_HOLDOUT_USED.json` already exists.
3. Validate data, KMLM live-ETF provenance, frozen artifact, hashes, and holdout length.
4. Atomically write the marker with `state: evaluation_in_progress`.
5. Only then calculate holdout features, targets, predictions, allocations, and returns.
6. Write every result, benchmark, diagnostic, and audit table.
7. Build a deployment bundle containing either frozen-architecture refit models or no
   models plus a fail-close reason.
8. Update the marker to `state: evaluation_completed`.

If a process fails after step 4, the holdout remains consumed. This is deliberate. Repair
the operational error without treating subsequent results as an untouched final test.
Deleting the marker or copying the project to manufacture another "first" look invalidates
the protocol.

## Frozen alpha gate

Deployment passes only when every check is true:

- all source segments authorize a primary evidence claim;
- holdout QRE CAGR exceeds holdout VOO CAGR after costs;
- holdout QRE terminal wealth exceeds holdout VOO terminal wealth;
- pre-holdout rolling five-year VOO-beating rate is at least 50%;
- the pre-holdout Reality Check p-value is at most 0.10.

The thresholds are intentionally visible and frozen. The final holdout does not choose a
new feature variant, model parameter, horizon weight, allocation rule, cadence, threshold,
or cost assumption.

## After the holdout

Permutation importance, regime contribution, false-signal analysis, calibration, and
stress tables are reporting diagnostics only. They may explain a result but cannot alter
the pass/fail conclusion.

When the gate passes, models are refit using all mature data under the identical frozen
architecture for subsequent live inference. When it fails, live inference always emits
100% VOO and the failure reason. A new research version requires a genuinely new future
holdout; it cannot reuse this one as untouched evidence.
