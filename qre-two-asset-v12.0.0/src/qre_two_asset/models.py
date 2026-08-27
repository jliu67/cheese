"""Compact four-model ensemble and nested chronological evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig
from .errors import DataIntegrityError
from .splitters import TimeFold, inner_prequential_folds, weekly_decision_dates
from .targets import HorizonTarget


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _expit(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


@dataclass
class ProbabilityCalibrator:
    slope: float = 1.0
    intercept: float = 0.0

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        return np.clip(_expit(self.intercept + self.slope * _logit(raw_probability)), 1e-4, 1 - 1e-4)


@dataclass
class HorizonModel:
    horizon: int
    feature_names: tuple[str, ...]
    regressors: tuple[object, object]
    classifiers: tuple[object, object]
    calibrator: ProbabilityCalibrator
    residual_quantiles: tuple[float, float]
    residual_scale: float
    feature_medians: pd.Series
    oof_metrics: dict[str, float]
    selected_parameters: dict[str, float | int]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame.reindex(columns=list(self.feature_names))
        reg_values = np.column_stack([model.predict(X) for model in self.regressors])
        cls_values = np.column_stack([model.predict_proba(X)[:, 1] for model in self.classifiers])
        expected = reg_values.mean(axis=1)
        raw_probability = cls_values.mean(axis=1)
        probability = self.calibrator.predict(raw_probability)
        return pd.DataFrame(
            {
                "expected": expected,
                "probability": probability,
                "lower": expected + self.residual_quantiles[0],
                "upper": expected + self.residual_quantiles[1],
                "reg_disagreement": reg_values.std(axis=1, ddof=0),
                "cls_disagreement": cls_values.std(axis=1, ddof=0),
                "residual_scale": self.residual_scale,
            },
            index=frame.index,
        )

    def local_contributions(self, row: pd.DataFrame) -> pd.Series:
        """Mechanical prediction changes when one feature is reset to its training median."""

        if len(row) != 1:
            raise ValueError("local contributions require exactly one row")
        base = float(self.predict(row)["expected"].iloc[0])
        contributions: dict[str, float] = {}
        for feature in self.feature_names:
            perturbed = row.copy()
            perturbed.loc[:, feature] = self.feature_medians.get(feature, np.nan)
            contributions[feature] = base - float(self.predict(perturbed)["expected"].iloc[0])
        return pd.Series(contributions).sort_values(key=np.abs, ascending=False)


@dataclass(frozen=True)
class VariantPredictions:
    variant: str
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    selected_parameters: pd.DataFrame


def _linear_regressor(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=float(alpha))),
        ]
    )


def _boosted_regressor(leaves: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.04,
                    max_iter=120,
                    max_leaf_nodes=int(leaves),
                    min_samples_leaf=30,
                    l2_regularization=2.0,
                    early_stopping=False,
                    random_state=seed,
                ),
            ),
        ]
    )


def _linear_classifier(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(C=float(c_value), max_iter=2000, random_state=seed),
            ),
        ]
    )


def _boosted_classifier(leaves: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_iter=120,
                    max_leaf_nodes=int(leaves),
                    min_samples_leaf=30,
                    l2_regularization=2.0,
                    early_stopping=False,
                    random_state=seed,
                ),
            ),
        ]
    )


def _valid_rows(X: pd.DataFrame, y: pd.Series, minimum_coverage: float = 0.65) -> pd.Series:
    return y.notna() & X.notna().mean(axis=1).ge(minimum_coverage)


def _candidate_score(
    factory: Callable[[float | int], object],
    parameter: float | int,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[TimeFold],
    *,
    classification: bool,
) -> float:
    losses: list[float] = []
    weights: list[int] = []
    for fold in folds:
        train_valid = _valid_rows(X.loc[fold.train_dates], y.loc[fold.train_dates])
        test_valid = _valid_rows(X.loc[fold.test_dates], y.loc[fold.test_dates])
        train_dates = fold.train_dates[train_valid.to_numpy()]
        test_dates = fold.test_dates[test_valid.to_numpy()]
        if len(train_dates) < 52 or len(test_dates) < 10:
            continue
        if classification and y.loc[train_dates].nunique() < 2:
            continue
        model = factory(parameter)
        model.fit(X.loc[train_dates], y.loc[train_dates].astype(int if classification else float))
        if classification:
            prediction = model.predict_proba(X.loc[test_dates])[:, 1]
            loss = brier_score_loss(y.loc[test_dates].astype(int), prediction)
        else:
            prediction = model.predict(X.loc[test_dates])
            loss = mean_squared_error(y.loc[test_dates].astype(float), prediction)
        losses.append(float(loss))
        weights.append(len(test_dates))
    if not losses:
        return float("inf")
    return float(np.average(losses, weights=weights))


def _select_parameter(
    factory: Callable[[float | int], object],
    candidates: tuple[float | int, ...],
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[TimeFold],
    *,
    classification: bool,
) -> float | int:
    scores = [
        _candidate_score(factory, candidate, X, y, folds, classification=classification)
        for candidate in candidates
    ]
    if not np.isfinite(scores).any():
        raise DataIntegrityError("every inner model candidate failed")
    return candidates[int(np.nanargmin(scores))]


def _fit_calibrator(raw_probability: np.ndarray, truth: np.ndarray) -> ProbabilityCalibrator:
    raw = np.asarray(raw_probability, dtype=float)
    y = np.asarray(truth, dtype=int)
    if len(raw) < 50 or np.unique(y).size < 2:
        return ProbabilityCalibrator()
    model = LogisticRegression(C=1000.0, max_iter=2000)
    model.fit(_logit(raw).reshape(-1, 1), y)
    slope = float(np.clip(model.coef_[0, 0], 0.0, 3.0))
    intercept = float(np.clip(model.intercept_[0], -5.0, 5.0))
    return ProbabilityCalibrator(slope=slope, intercept=intercept)


def _expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= left) & (p < right if right < 1.0 else p <= right)
        if mask.any():
            error += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error if total else np.nan)


def fit_horizon_model(
    X: pd.DataFrame,
    target: HorizonTarget,
    config: ModelConfig,
    *,
    sample_weekly: bool = True,
    inner_folds: int = 3,
) -> HorizonModel:
    feature_names = tuple(X.columns)
    if len(feature_names) > config.maximum_features:
        raise DataIntegrityError("feature ceiling exceeded")
    decision_dates = X.index
    if sample_weekly:
        decision_dates = weekly_decision_dates(decision_dates)
    y_reg = target.relative_log_return.reindex(decision_dates)
    y_cls = target.voo_outperforms.astype("Float64").reindex(decision_dates).astype(float)
    outcome_end = target.outcome_end_date.reindex(decision_dates)
    X_sample = X.reindex(decision_dates)
    common_valid = _valid_rows(X_sample, y_reg) & y_cls.notna() & outcome_end.notna()
    X_sample = X_sample.loc[common_valid]
    y_reg = y_reg.loc[common_valid].astype(float)
    y_cls = y_cls.loc[common_valid].astype(int)
    outcome_end = outcome_end.loc[common_valid]
    if len(X_sample) < 156 or y_cls.nunique() < 2:
        raise DataIntegrityError("insufficient mature observations for horizon model")
    folds = inner_prequential_folds(X_sample.index, outcome_end, n_splits=inner_folds)
    seed = config.random_seed + target.horizon
    ridge_alpha = _select_parameter(
        _linear_regressor, config.ridge_alphas, X_sample, y_reg, folds, classification=False
    )
    boost_reg_leaves = _select_parameter(
        lambda value: _boosted_regressor(int(value), seed),
        config.boosted_leaf_nodes,
        X_sample,
        y_reg,
        folds,
        classification=False,
    )
    logistic_c = _select_parameter(
        lambda value: _linear_classifier(float(value), seed),
        config.logistic_c,
        X_sample,
        y_cls,
        folds,
        classification=True,
    )
    boost_cls_leaves = _select_parameter(
        lambda value: _boosted_classifier(int(value), seed),
        config.boosted_leaf_nodes,
        X_sample,
        y_cls,
        folds,
        classification=True,
    )

    reg_factories = (
        lambda: _linear_regressor(float(ridge_alpha)),
        lambda: _boosted_regressor(int(boost_reg_leaves), seed),
    )
    cls_factories = (
        lambda: _linear_classifier(float(logistic_c), seed),
        lambda: _boosted_classifier(int(boost_cls_leaves), seed),
    )
    oof_reg: list[float] = []
    oof_cls: list[float] = []
    oof_reg_truth: list[float] = []
    oof_cls_truth: list[int] = []
    for fold in folds:
        train_dates = fold.train_dates.intersection(X_sample.index)
        test_dates = fold.test_dates.intersection(X_sample.index)
        if len(train_dates) < 52 or len(test_dates) < 10 or y_cls.loc[train_dates].nunique() < 2:
            continue
        reg_models = [factory() for factory in reg_factories]
        cls_models = [factory() for factory in cls_factories]
        for model in reg_models:
            model.fit(X_sample.loc[train_dates], y_reg.loc[train_dates])
        for model in cls_models:
            model.fit(X_sample.loc[train_dates], y_cls.loc[train_dates])
        reg_prediction = np.column_stack(
            [model.predict(X_sample.loc[test_dates]) for model in reg_models]
        ).mean(axis=1)
        cls_prediction = np.column_stack(
            [model.predict_proba(X_sample.loc[test_dates])[:, 1] for model in cls_models]
        ).mean(axis=1)
        oof_reg.extend(reg_prediction.tolist())
        oof_cls.extend(cls_prediction.tolist())
        oof_reg_truth.extend(y_reg.loc[test_dates].tolist())
        oof_cls_truth.extend(y_cls.loc[test_dates].tolist())
    if len(oof_reg) < 50:
        raise DataIntegrityError("too few prequential predictions for calibration")
    oof_reg_array = np.asarray(oof_reg)
    oof_cls_array = np.asarray(oof_cls)
    reg_truth = np.asarray(oof_reg_truth)
    cls_truth = np.asarray(oof_cls_truth, dtype=int)
    # Evaluate calibration on the chronological tail only. The deployable
    # calibrator is then refit on all prequential predictions.
    calibration_cut = max(50, int(len(oof_cls_array) * 0.67))
    if (
        len(oof_cls_array) - calibration_cut >= 20
        and np.unique(cls_truth[:calibration_cut]).size == 2
        and np.unique(cls_truth[calibration_cut:]).size == 2
    ):
        evaluation_calibrator = _fit_calibrator(
            oof_cls_array[:calibration_cut], cls_truth[:calibration_cut]
        )
        metric_truth = cls_truth[calibration_cut:]
        metric_probability = evaluation_calibrator.predict(oof_cls_array[calibration_cut:])
    else:
        evaluation_calibrator = _fit_calibrator(oof_cls_array, cls_truth)
        metric_truth = cls_truth
        metric_probability = evaluation_calibrator.predict(oof_cls_array)
    calibrator = _fit_calibrator(oof_cls_array, cls_truth)
    residuals = reg_truth - oof_reg_array
    low, high = np.quantile(residuals, [0.10, 0.90])
    residual_scale = float(max(np.std(residuals, ddof=0), 1e-6))
    metrics: dict[str, float] = {
        "rmse": float(np.sqrt(mean_squared_error(reg_truth, oof_reg_array))),
        "brier": float(brier_score_loss(metric_truth, metric_probability)),
        "log_loss": float(log_loss(metric_truth, metric_probability, labels=[0, 1])),
        "ece": _expected_calibration_error(metric_truth, metric_probability),
        "prequential_samples": float(len(oof_reg_array)),
        "calibration_test_samples": float(len(metric_truth)),
    }
    if np.unique(metric_truth).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(metric_truth, metric_probability))

    regressors = tuple(factory() for factory in reg_factories)
    classifiers = tuple(factory() for factory in cls_factories)
    for model in regressors:
        model.fit(X_sample, y_reg)
    for model in classifiers:
        model.fit(X_sample, y_cls)
    medians = X_sample.median(axis=0, skipna=True)
    parameters: dict[str, float | int] = {
        "ridge_alpha": float(ridge_alpha),
        "boosted_reg_leaf_nodes": int(boost_reg_leaves),
        "logistic_c": float(logistic_c),
        "boosted_cls_leaf_nodes": int(boost_cls_leaves),
    }
    return HorizonModel(
        horizon=target.horizon,
        feature_names=feature_names,
        regressors=regressors,  # type: ignore[arg-type]
        classifiers=classifiers,  # type: ignore[arg-type]
        calibrator=calibrator,
        residual_quantiles=(float(low), float(high)),
        residual_scale=residual_scale,
        feature_medians=medians,
        oof_metrics=metrics,
        selected_parameters=parameters,
    )


def walk_forward_variant(
    variant: str,
    features: pd.DataFrame,
    targets: dict[int, HorizonTarget],
    outer_folds: list[TimeFold],
    config: ModelConfig,
    *,
    inner_folds: int = 3,
) -> VariantPredictions:
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []
    parameter_rows: list[dict[str, float | int | str]] = []
    for fold in outer_folds:
        fold_frame = pd.DataFrame(index=fold.test_dates)
        fold_frame["fold"] = fold.fold_id
        for horizon in config.horizons:
            model = fit_horizon_model(
                features.loc[fold.train_dates],
                targets[horizon],
                config,
                sample_weekly=True,
                inner_folds=inner_folds,
            )
            predicted = model.predict(features.loc[fold.test_dates])
            for column in predicted.columns:
                fold_frame[f"h{horizon}_{column}"] = predicted[column]
            truth_reg = targets[horizon].relative_log_return.reindex(fold.test_dates)
            truth_cls = targets[horizon].voo_outperforms.astype("Float64").reindex(fold.test_dates)
            valid = truth_reg.notna() & truth_cls.notna()
            if valid.any():
                metric_rows.append(
                    {
                        "variant": variant,
                        "fold": fold.fold_id,
                        "horizon": horizon,
                        "test_start": fold.test_start.date().isoformat(),
                        "test_end": fold.test_end.date().isoformat(),
                        "samples": int(valid.sum()),
                        "rmse": float(
                            np.sqrt(
                                mean_squared_error(
                                    truth_reg.loc[valid], predicted.loc[valid, "expected"]
                                )
                            )
                        ),
                        "brier": float(
                            brier_score_loss(
                                truth_cls.loc[valid].astype(int),
                                predicted.loc[valid, "probability"],
                            )
                        ),
                    }
                )
            parameter_rows.append(
                {
                    "variant": variant,
                    "fold": fold.fold_id,
                    "horizon": horizon,
                    **model.selected_parameters,
                }
            )
        prediction_parts.append(fold_frame)
    predictions = pd.concat(prediction_parts).sort_index()
    return VariantPredictions(
        variant=variant,
        predictions=predictions,
        fold_metrics=pd.DataFrame(metric_rows),
        selected_parameters=pd.DataFrame(parameter_rows),
    )


def aggregate_horizons(predictions: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    expected_63 = np.zeros(len(predictions), dtype=float)
    probability = np.zeros(len(predictions), dtype=float)
    standardized = np.zeros(len(predictions), dtype=float)
    reg_uncertainty = np.zeros(len(predictions), dtype=float)
    cls_uncertainty = np.zeros(len(predictions), dtype=float)
    lower_63 = np.zeros(len(predictions), dtype=float)
    upper_63 = np.zeros(len(predictions), dtype=float)
    for horizon, weight in zip(config.horizons, config.horizon_weights, strict=True):
        scale = predictions[f"h{horizon}_residual_scale"].clip(lower=1e-6)
        expected = predictions[f"h{horizon}_expected"]
        expected_63 += weight * expected.to_numpy() * (63.0 / horizon)
        lower_63 += weight * predictions[f"h{horizon}_lower"].to_numpy() * (63.0 / horizon)
        upper_63 += weight * predictions[f"h{horizon}_upper"].to_numpy() * (63.0 / horizon)
        probability += weight * predictions[f"h{horizon}_probability"].to_numpy()
        standardized += weight * (expected / scale).to_numpy()
        reg_uncertainty += weight * (
            predictions[f"h{horizon}_reg_disagreement"] / scale
        ).to_numpy()
        cls_uncertainty += weight * predictions[f"h{horizon}_cls_disagreement"].to_numpy()
    directional = 2.0 * probability - 1.0
    raw_score = 0.5 * np.tanh(standardized) + 0.5 * directional
    shrink = np.clip(1.0 - 0.5 * reg_uncertainty - cls_uncertainty, 0.25, 1.0)
    score = np.clip(raw_score * shrink, -1.0, 1.0)
    return pd.DataFrame(
        {
            "expected_relative_log_return_63_equivalent": expected_63,
            "probability_voo_outperforms": probability,
            "relative_return_lower_80": lower_63,
            "relative_return_upper_80": upper_63,
            "model_agreement_shrinkage": shrink,
            "allocation_score": score,
            "directional_confidence": np.maximum(probability, 1.0 - probability),
        },
        index=predictions.index,
    )


def fit_final_models(
    features: pd.DataFrame,
    targets: dict[int, HorizonTarget],
    config: ModelConfig,
    *,
    inner_folds: int = 3,
) -> dict[int, HorizonModel]:
    return {
        horizon: fit_horizon_model(
            features,
            targets[horizon],
            config,
            sample_weekly=True,
            inner_folds=inner_folds,
        )
        for horizon in config.horizons
    }


def calibration_diagnostics(
    predictions: pd.DataFrame,
    targets: dict[int, HorizonTarget],
    config: ModelConfig,
    *,
    bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reliability curves and proper scoring rules for chronological predictions."""

    curve_rows: list[dict[str, float | int]] = []
    summary_rows: list[dict[str, float | int]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for horizon in config.horizons:
        probability = predictions[f"h{horizon}_probability"].astype(float)
        truth = targets[horizon].voo_outperforms.astype("Float64").reindex(predictions.index)
        valid = probability.notna() & truth.notna()
        p = probability.loc[valid].clip(1e-6, 1 - 1e-6).to_numpy(dtype=float)
        y = truth.loc[valid].astype(int).to_numpy(dtype=int)
        if len(y) == 0:
            continue
        bin_ids = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
        ece = 0.0
        for bin_id in range(bins):
            mask = bin_ids == bin_id
            if not mask.any():
                continue
            mean_prediction = float(p[mask].mean())
            observed = float(y[mask].mean())
            gap = abs(mean_prediction - observed)
            ece += float(mask.mean()) * gap
            curve_rows.append(
                {
                    "horizon": horizon,
                    "bin": bin_id + 1,
                    "lower": float(edges[bin_id]),
                    "upper": float(edges[bin_id + 1]),
                    "samples": int(mask.sum()),
                    "mean_prediction": mean_prediction,
                    "observed_frequency": observed,
                    "absolute_gap": gap,
                }
            )
        summary: dict[str, float | int] = {
            "horizon": horizon,
            "samples": len(y),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "ece": float(ece),
        }
        summary["roc_auc"] = (
            float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
        )
        summary_rows.append(summary)
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


def predict_final_models(
    models: dict[int, HorizonModel], features: pd.DataFrame, config: ModelConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(index=features.index)
    for horizon in config.horizons:
        predicted = models[horizon].predict(features)
        for column in predicted.columns:
            raw[f"h{horizon}_{column}"] = predicted[column]
    return raw, aggregate_horizons(raw, config)


def holdout_permutation_importance(
    models: dict[int, HorizonModel],
    features: pd.DataFrame,
    targets: dict[int, HorizonTarget],
    config: ModelConfig,
) -> pd.DataFrame:
    """Frozen-model importance on holdout data; reporting only, never selection."""

    rows: list[dict[str, float | int | str]] = []
    for horizon in config.horizons:
        model = models[horizon]
        truth_reg = targets[horizon].relative_log_return.reindex(features.index)
        truth_cls = targets[horizon].voo_outperforms.astype("Float64").reindex(features.index)
        valid = truth_reg.notna() & truth_cls.notna()
        X = features.loc[valid, list(model.feature_names)]
        if len(X) < 20:
            continue
        y_reg = truth_reg.loc[valid].astype(float)
        y_cls = truth_cls.loc[valid].astype(int)
        baseline = model.predict(X)
        baseline_rmse = float(np.sqrt(mean_squared_error(y_reg, baseline["expected"])))
        baseline_brier = float(brier_score_loss(y_cls, baseline["probability"]))
        rng = np.random.default_rng(config.random_seed + horizon)
        for feature in model.feature_names:
            shuffled = X.copy()
            shuffled.loc[:, feature] = rng.permutation(shuffled[feature].to_numpy())
            prediction = model.predict(shuffled)
            rows.append(
                {
                    "horizon": horizon,
                    "feature": feature,
                    "rmse_increase": float(
                        np.sqrt(mean_squared_error(y_reg, prediction["expected"]))
                        - baseline_rmse
                    ),
                    "brier_increase": float(
                        brier_score_loss(y_cls, prediction["probability"]) - baseline_brier
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["horizon", "rmse_increase"], ascending=[True, False]
    ).reset_index(drop=True)
