from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from qre_two_asset.config import ModelConfig
from qre_two_asset.features import build_features
from qre_two_asset.models import fit_horizon_model
from qre_two_asset.targets import build_targets
from tests.helpers import synthetic_market


class ModelTests(unittest.TestCase):
    def test_compact_model_fits_and_predicts(self) -> None:
        market = synthetic_market(periods=3000)
        features = build_features(market).for_variant("A")
        targets = build_targets(market.levels)
        train = features.iloc[:2600]
        model = fit_horizon_model(
            train,
            targets[21],
            ModelConfig(
                ridge_alphas=(1.0,),
                logistic_c=(0.1,),
                boosted_leaf_nodes=(7,),
            ),
        )
        prediction = model.predict(features.iloc[[2700]])
        self.assertGreater(prediction["probability"].iloc[0], 0)
        self.assertLess(prediction["probability"].iloc[0], 1)
        self.assertEqual(len(model.regressors), 2)
        self.assertEqual(len(model.classifiers), 2)
        alternate_seed = fit_horizon_model(
            train,
            targets[21],
            replace(
                ModelConfig(
                    ridge_alphas=(1.0,),
                    logistic_c=(0.1,),
                    boosted_leaf_nodes=(7,),
                ),
                random_seed=999,
            ),
        ).predict(features.iloc[[2700]])
        np.testing.assert_allclose(prediction, alternate_seed, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
