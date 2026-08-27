# Prior QRE v11 reuse audit

The latest available prior archive inspected was `quant-regime-engine-forward-v11.0.0`.
It was treated as a source of infrastructure patterns, not as a modeling foundation.

## Reused as patterns

| Prior capability | v12 treatment |
|---|---|
| Explicit observation/release metadata | Rebuilt as the smaller PIT vintage contract in `data.py` |
| Chronological forward-testing concepts | Rebuilt with exact label-end purging and nested folds |
| CLI and configuration shape | Reduced to five commands and a small frozen YAML surface |
| Structured outputs and SQLite auditability | Reimplemented with atomic CSV/JSON/joblib/SQLite writes |
| Status/fail-safe operational pattern | Reimplemented as freeze/holdout/deployment states and VOO fail-close |
| Point-in-time and contract tests | Rebuilt as targeted standard-library unit and integration tests |

No source module was copied wholesale. The new package namespace, schemas, models, and
artifacts are independent.

## Explicitly discarded

- recession → recovery → regime → ranking → optimizer hierarchy;
- defensive-risk and recovery-state scoring as central allocation logic;
- large forward-model subsystems and many intermediate labels;
- broad multi-asset allocation and unavailable asset fallbacks;
- large macro feature catalogs;
- inherited trained model files and their historical performance claims;
- service, alerting, scheduler, and web layers unrelated to the primary research question.

## Why

The prior architecture optimized and combined many intermediate concepts that were not the
new objective. Reusing it would preserve both complexity and historical research degrees of
freedom. v12 instead predicts the actual VOO-minus-KMLM decision target and selects a
transparent two-weight allocation on net CAGR.

## Data audit result

The inspected v11 and trained v10 archives contained no defensible continuous VOO/KMLM
total-return dataset suitable for the requested long-history test. No stored return table or
trained artifact was promoted into v12. Consequently, v12 is delivered with an explicit
`NOT TESTED` alpha status and rejects fabricated or unlabeled proxy evidence.
