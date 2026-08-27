from __future__ import annotations

import unittest

from qre_two_asset.config import EngineConfig, ModelConfig
from qre_two_asset.errors import ConfigError


class ConfigTests(unittest.TestCase):
    def test_frozen_horizons_are_enforced(self) -> None:
        config = EngineConfig(models=ModelConfig(horizons=(21, 63)))
        with self.assertRaisesRegex(ConfigError, "21/63/126"):
            config.validate()

    def test_config_digest_is_deterministic(self) -> None:
        self.assertEqual(EngineConfig().digest(), EngineConfig().digest())


if __name__ == "__main__":
    unittest.main()
