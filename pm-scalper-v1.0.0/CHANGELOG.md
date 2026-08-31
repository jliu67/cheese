# Changelog

## 1.0.0 — 2026-08-30

- Added a public-data recorder and paper-trading engine with no live-order path.
- Added Gamma full-market discovery, liquidity/volume filtering, and detail hydration fallback.
- Added CLOB batch snapshots, periodic reconciliation, and market WebSocket ingestion.
- Added raw normalized JSONL capture and a SQLite audit trail.
- Added momentum, depth imbalance, trade flow, volume acceleration, microprice, spread, and volatility features.
- Added a transparent YES/NO opportunity score with maker-first entry simulation.
- Added queue-ahead fills from qualifying trade prints and tick-aware profit targets.
- Added fee-aware bid-book risk exits and current USDC-notional minimum-order enforcement.
- Added side-specific depth-state guards so unrelated events cannot replenish consumed liquidity.
- Added stale-response protection so slower REST snapshots cannot roll state backward.
- Added UTC-daily trade/loss resets, consecutive-loss controls, resolution buffers, and kill-switch entry cancellation.
- Added public `market_resolved` handling with 1/0 paper settlement and re-entry blocking.
- Added conservative cancellation when a new tick size invalidates a resting entry or an updated minimum notional invalidates an untouched entry.
- Added reports that preserve the bankroll stored with each run.
- Added Rich terminal monitoring, CSV/JSON exports, shell launchers, and 35 offline tests.
