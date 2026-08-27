from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from qre_two_asset.artifacts import atomic_write_json


class ArtifactTests(unittest.TestCase):
    def test_json_is_strict_and_replaces_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            atomic_write_json(path, {"finite": np.float64(1.25), "missing": float("nan")})
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", text)
            self.assertEqual(json.loads(text), {"finite": 1.25, "missing": None})


if __name__ == "__main__":
    unittest.main()
