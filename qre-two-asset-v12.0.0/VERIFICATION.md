# Verification record

## Scope

This record verifies software behavior, not investment performance. No approved historical
VOO/KMLM input was present, so CAGR and terminal-wealth success remain untested.

## Commands

```bash
PYTHONPATH=src python -m compileall -q src tests
PYTHONPATH=src python -m unittest discover -s tests -v
QRE_RUN_INTEGRATION=1 PYTHONPATH=src python -m unittest tests.test_pipeline_integration -v
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

## Results

- Compilation: passed.
- Fast suite: 18 tests passed; one opt-in integration test skipped by default.
- Full synthetic integration: passed in approximately 22 seconds in the build environment.
- Wheel build and isolated `qre2 --help`/version import smoke: passed.
- Wheel SHA-256: `8955c1d622974733514a7de8f2dbffa71dc9298d101f8ac165d64a0d62a7c17a`.

The integration workflow generated synthetic total-return levels, ran pre-holdout nested
walk-forward development, froze one candidate, evaluated a single-use holdout, refused a
second evaluation, failed the provenance/alpha gate, and emitted a live 100% VOO / 0% KMLM
allocation. Synthetic performance was neither recorded as research evidence nor used to
alter architecture.

## Covered invariants

- frozen target horizons and deterministic configuration hash;
- market/provenance validation and proxy claim prohibition;
- point-in-time release invisibility and conflicting-vintage rejection;
- exact 12/23/35/40 feature counts and future-mutation causality;
- next-close execution alignment and forward-label end dates;
- purged expanding and prequential folds;
- compact model fit, probability bounds, and four-family structure;
- identical predictions across alternate seeds for the deliberately deterministic estimators;
- fully invested bounded allocations;
- transaction costs reduce wealth;
- CAGR remains primary over smoother underperformance;
- deterministic block bootstrap and Reality Check;
- final-holdout single-use marker and live fail-close.

## Deliberately not asserted

- that QRE beats VOO;
- that a particular KMLM prehistory is an investable ETF history;
- that default transaction costs match a particular broker or trade size;
- that historical calibration or alpha will persist live.
