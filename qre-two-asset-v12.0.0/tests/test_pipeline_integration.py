from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from qre_two_asset.errors import HoldoutError
from qre_two_asset.pipeline import (
    evaluate_final_holdout,
    run_development,
    run_live_inference,
)


@unittest.skipUnless(
    os.environ.get("QRE_RUN_INTEGRATION") == "1",
    "set QRE_RUN_INTEGRATION=1 to run the full synthetic workflow",
)
class PipelineIntegrationTests(unittest.TestCase):
    def test_freeze_one_time_holdout_and_fail_closed_live(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            dates = pd.bdate_range("2005-01-03", periods=3800)
            rng = np.random.default_rng(123)
            phase = np.sin(np.arange(len(dates)) / 150.0)
            voo = 100 * np.exp(
                np.cumsum(0.00035 + 0.0004 * phase + rng.normal(0, 0.009, len(dates)))
            )
            kmlm = 100 * np.exp(
                np.cumsum(0.00010 - 0.0003 * phase + rng.normal(0, 0.007, len(dates)))
            )
            pd.DataFrame({"date": dates, "VOO": voo, "KMLM": kmlm}).to_csv(
                data / "market.csv", index=False
            )
            (data / "provenance.yaml").write_text(
                """assets:
  VOO:
    - start: '2005-01-03'
      end: null
      source_name: integration_test
      source_series: VOO_SYNTHETIC
      history_kind: synthetic_test
      return_basis: net_total_return
      expense_adjustment_bps: 0
      hypothetical: true
      primary_claim_allowed: false
  KMLM:
    - start: '2005-01-03'
      end: null
      source_name: integration_test
      source_series: KMLM_SYNTHETIC
      history_kind: synthetic_test
      return_basis: net_total_return
      expense_adjustment_bps: 0
      hypothetical: true
      primary_claim_allowed: false
""",
                encoding="utf-8",
            )
            config = root / "research.yaml"
            config.write_text(
                f"""market_data: {data / 'market.csv'}
provenance: {data / 'provenance.yaml'}
output_directory: {root / 'outputs'}
artifact_directory: {root / 'artifacts'}
feature_variants: [A]
walk_forward:
  minimum_train_years: 8
  test_years: 1
  step_years: 1
  inner_folds: 3
  sampling: weekly
models:
  horizons: [21, 63, 126]
  horizon_weights: [0.2, 0.35, 0.45]
  random_seed: 1729
  ridge_alphas: [1.0]
  logistic_c: [0.1]
  boosted_leaf_nodes: [7]
  maximum_features: 40
allocation:
  neutral_voo_weight: 0.7
  maximum_tilt: 0.7
  methods: [continuous]
  cadences: [weekly]
  minimum_changes: [0.1]
  buckets: [0.0, 0.5, 1.0]
  execution_lag_sessions: 1
  transaction_cost_bps: 5
  selection_tolerance_annual_bps: 25
holdout:
  start: '2016-01-04'
  minimum_years: 3
  require_kmlm_live_etf: false
  acknowledgement: I_UNDERSTAND_THIS_USES_THE_FINAL_HOLDOUT
deployment:
  minimum_preholdout_rolling_5y_win_rate: 0.5
  maximum_reality_check_pvalue: 0.1
""",
                encoding="utf-8",
            )

            development = run_development(config)
            self.assertEqual(development["manifest"]["status"], "frozen_preholdout")
            for relative in (
                "outputs/development/all_candidate_returns.csv",
                "outputs/development/benchmark_comparison.csv",
                "outputs/development/execution_ledger.csv",
                "outputs/development/historical_episodes.csv",
                "outputs/development/variant_A/calibration_reliability.csv",
                "outputs/development/variant_A/feature_fold_stability.csv",
                "artifacts/frozen_preholdout_manifest.json",
            ):
                self.assertTrue((root / relative).exists(), relative)
            marker = root / "artifacts" / "FINAL_HOLDOUT_USED.json"
            self.assertFalse(marker.exists())

            holdout = evaluate_final_holdout(
                config,
                acknowledgement="I_UNDERSTAND_THIS_USES_THE_FINAL_HOLDOUT",
            )
            self.assertEqual(holdout["result"]["primary_objective"], "FAILED")
            self.assertTrue(marker.exists())
            for relative in (
                "outputs/final_holdout/benchmarks.csv",
                "outputs/final_holdout/calibration_summary.csv",
                "outputs/final_holdout/bear_market_timing.csv",
                "outputs/final_holdout/execution_ledger.csv",
                "artifacts/deployment_bundle.joblib",
            ):
                self.assertTrue((root / relative).exists(), relative)
            with self.assertRaises(HoldoutError):
                evaluate_final_holdout(
                    config,
                    acknowledgement="I_UNDERSTAND_THIS_USES_THE_FINAL_HOLDOUT",
                )
            with self.assertRaises(HoldoutError):
                run_development(config)

            live = run_live_inference(config)
            self.assertEqual(live["allocation"], {"VOO": 1.0, "KMLM": 0.0})
            self.assertEqual(live["deployment_status"], "FAILED_ALPHA_GATE_FAIL_CLOSED")
            self.assertTrue((root / "outputs/live/allocation_history.csv").exists())


if __name__ == "__main__":
    unittest.main()
