from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from qre_two_asset.data import (
    build_pit_panel,
    load_market_data,
    load_provenance,
    validate_holdout_provenance,
    validate_pit_observations,
)
from qre_two_asset.errors import DataIntegrityError


def _write_market(path: Path) -> None:
    dates = pd.bdate_range("2018-01-02", periods=800)
    frame = pd.DataFrame(
        {
            "date": dates,
            "VOO": 100.0 * (1.0002 ** pd.Series(range(len(dates)))),
            "KMLM": 100.0 * (1.0001 ** pd.Series(range(len(dates)))),
        }
    )
    frame.to_csv(path, index=False)


def _write_provenance(path: Path, *, kmlm_kind: str = "live_etf") -> None:
    path.write_text(
        f"""assets:
  VOO:
    - start: '2018-01-02'
      end: null
      source_name: test
      source_series: VOO
      history_kind: live_etf
      return_basis: adjusted_total_return
      expense_adjustment_bps: 0
      hypothetical: false
      primary_claim_allowed: true
  KMLM:
    - start: '2018-01-02'
      end: null
      source_name: test
      source_series: KMLM
      history_kind: {kmlm_kind}
      return_basis: adjusted_total_return
      expense_adjustment_bps: 0
      hypothetical: false
      primary_claim_allowed: {'false' if kmlm_kind == 'proxy' else 'true'}
""",
        encoding="utf-8",
    )


class DataTests(unittest.TestCase):
    def test_market_and_provenance_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market_path = root / "market.csv"
            provenance_path = root / "provenance.yaml"
            _write_market(market_path)
            _write_provenance(provenance_path)
            market = load_market_data(market_path, provenance_path)
            self.assertTrue(market.primary_claim_allowed)
            validate_holdout_provenance(market, "2019-01-01", True)

    def test_proxy_cannot_authorize_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provenance_path = Path(directory) / "provenance.yaml"
            _write_provenance(provenance_path, kmlm_kind="proxy")
            segments = load_provenance(provenance_path)
            self.assertFalse(all(item.primary_claim_allowed for item in segments))

    def test_pit_value_is_invisible_until_release(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "series_id": "VIX",
                    "observation_date": "2008-01-01",
                    "release_date": "2008-01-03",
                    "vintage_start": "2008-01-03",
                    "vintage_end": "2008-01-06",
                    "value": 20.0,
                    "source": "test",
                    "source_series": "VIX",
                },
                {
                    "series_id": "VIX",
                    "observation_date": "2008-01-01",
                    "release_date": "2008-01-07",
                    "vintage_start": "2008-01-07",
                    "vintage_end": "",
                    "value": 21.0,
                    "source": "test",
                    "source_series": "VIX",
                },
            ]
        )
        panel = build_pit_panel(frame, pd.date_range("2008-01-01", "2008-01-08"))
        self.assertTrue(pd.isna(panel.values.loc["2008-01-02", "VIX"]))
        self.assertEqual(panel.values.loc["2008-01-03", "VIX"], 20.0)
        self.assertEqual(panel.values.loc["2008-01-07", "VIX"], 21.0)

    def test_conflicting_vintages_are_fatal(self) -> None:
        base = {
            "series_id": "NFCI",
            "observation_date": "2008-01-01",
            "release_date": "2008-01-03",
            "vintage_start": "2008-01-03",
            "vintage_end": "",
            "source": "test",
            "source_series": "NFCI",
        }
        with self.assertRaisesRegex(DataIntegrityError, "conflicting"):
            validate_pit_observations(
                pd.DataFrame([{**base, "value": 1}, {**base, "value": 2}])
            )


if __name__ == "__main__":
    unittest.main()
