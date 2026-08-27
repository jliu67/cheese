"""Development freeze, one-time holdout evaluation, and live inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .analysis import (
    bear_market_timing_analysis,
    cost_and_threshold_robustness,
    deployment_gate,
    false_signal_analysis,
    horizon_weight_robustness,
    historical_episode_analysis,
    regime_contribution,
)
from .artifacts import (
    append_ledger,
    atomic_joblib_dump,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    sha256_frame,
    utc_now,
    write_audit_database,
)
from .config import EngineConfig, load_config, resolve_path
from .data import (
    MarketData,
    PointInTimePanel,
    build_pit_panel,
    load_market_data,
    load_pit_data,
    validate_holdout_provenance,
)
from .errors import DataIntegrityError, HoldoutError
from .features import (
    FEATURE_VERSION,
    FeatureSet,
    build_features,
    feature_fold_stability,
    feature_redundancy,
)
from .models import (
    aggregate_horizons,
    calibration_diagnostics,
    fit_final_models,
    holdout_permutation_importance,
    predict_final_models,
    walk_forward_variant,
)
from .portfolio import (
    AllocatorSpec,
    allocate,
    allocator_grid,
    backtest,
    benchmark_table,
    execution_ledger,
    performance_metrics,
    rolling_summary,
    select_allocator,
)
from .reporting import development_report, holdout_report, live_report
from .splitters import assert_fold_integrity, expanding_calendar_folds
from .statistics import block_bootstrap_cagr_excess, reality_check
from .targets import HorizonTarget, build_targets, decision_period_returns


@dataclass(frozen=True)
class ProjectPaths:
    config: Path
    market: Path
    provenance: Path
    macro: Path | None
    outputs: Path
    artifacts: Path


@dataclass(frozen=True)
class ResearchContext:
    config: EngineConfig
    paths: ProjectPaths
    market: MarketData
    external: PointInTimePanel | None
    features: FeatureSet
    targets: dict[int, HorizonTarget]


def project_paths(config_path: str | Path, config: EngineConfig) -> ProjectPaths:
    source = Path(config_path).resolve()
    market = resolve_path(source, config.market_data)
    provenance = resolve_path(source, config.provenance)
    outputs = resolve_path(source, config.output_directory)
    artifacts = resolve_path(source, config.artifact_directory)
    if market is None or provenance is None or outputs is None or artifacts is None:
        raise DataIntegrityError("market, provenance, output, and artifact paths are required")
    return ProjectPaths(
        config=source,
        market=market,
        provenance=provenance,
        macro=resolve_path(source, config.macro_data),
        outputs=outputs,
        artifacts=artifacts,
    )


def _slice_market(market: MarketData, dates: pd.DatetimeIndex) -> MarketData:
    return MarketData(
        raw_levels=market.raw_levels.loc[dates],
        levels=market.levels.loc[dates],
        returns=market.returns.loc[dates],
        segments=market.segments,
        primary_claim_allowed=market.primary_claim_allowed,
        warnings=market.warnings,
    )


def _context_from_market(
    config: EngineConfig, paths: ProjectPaths, market: MarketData
) -> ResearchContext:
    external = None
    if paths.macro is not None:
        external = build_pit_panel(load_pit_data(paths.macro), market.dates)
    features = build_features(market, external)
    targets = build_targets(
        market.levels,
        horizons=config.models.horizons,
        execution_lag=config.allocation.execution_lag_sessions,
    )
    return ResearchContext(config, paths, market, external, features, targets)


def load_research_context(
    config_path: str | Path, *, preholdout_only: bool = False
) -> ResearchContext:
    config = load_config(config_path)
    paths = project_paths(config_path, config)
    market = load_market_data(paths.market, paths.provenance)
    if not preholdout_only:
        return _context_from_market(config, paths, market)
    validate_holdout_provenance(
        market, config.holdout.start, config.holdout.require_kmlm_live_etf
    )
    boundary = pd.Timestamp(config.holdout.start).normalize()
    dates = market.dates[market.dates < boundary]
    if len(dates) < 252 * config.walk_forward.minimum_train_years:
        raise DataIntegrityError("not enough pre-holdout history for the frozen protocol")
    return _context_from_market(config, paths, _slice_market(market, dates))


def inspect_data(config_path: str | Path) -> dict[str, Any]:
    context = load_research_context(config_path)
    validate_holdout_provenance(
        context.market,
        context.config.holdout.start,
        context.config.holdout.require_kmlm_live_etf,
    )
    return {
        "start": context.market.dates[0].date().isoformat(),
        "end": context.market.dates[-1].date().isoformat(),
        "sessions": len(context.market.dates),
        "primary_claim_allowed": context.market.primary_claim_allowed,
        "external_features_available": context.features.external_available,
        "feature_counts": {
            key: len(value) for key, value in context.features.variants.items()
        },
        "warnings": list(context.market.warnings),
        "provenance": context.market.provenance_table().to_dict(orient="records"),
    }


def _variant_selection(ablation: pd.DataFrame, tolerance_bps: float) -> str:
    best = float(ablation["cagr"].max())
    finalists = ablation[ablation["cagr"] >= best - tolerance_bps / 10_000.0].copy()
    return str(
        finalists.sort_values(
            ["feature_count", "annualized_turnover", "variant"], ascending=[True, True, True]
        ).iloc[0]["variant"]
    )


def _candidate_backtests(
    variant: str,
    signals: pd.DataFrame,
    decision_returns: pd.DataFrame,
    config: EngineConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    returns: dict[str, pd.Series] = {}
    ledger: list[dict[str, Any]] = []
    for spec in allocator_grid(config.allocation):
        weights = allocate(signals, config.allocation, spec)
        result = backtest(
            weights,
            decision_returns,
            transaction_cost_bps=config.allocation.transaction_cost_bps,
        )
        name = f"{variant}__{spec.name}"
        returns[name] = result.returns
        ledger.append(
            {
                "kind": "allocator_candidate",
                "name": name,
                "variant": variant,
                "features": int(signals.attrs.get("feature_count", 0)),
                "allocator": asdict(spec),
            }
        )
    return pd.DataFrame(returns), ledger


def run_development(config_path: str | Path) -> dict[str, Any]:
    preflight_config = load_config(config_path)
    preflight_paths = project_paths(config_path, preflight_config)
    if (preflight_paths.artifacts / "FINAL_HOLDOUT_USED.json").exists():
        raise HoldoutError("development is locked because the final holdout has been consumed")
    # No features, labels, forecasts, or metrics are calculated on final-holdout rows.
    context = load_research_context(config_path, preholdout_only=True)
    config = context.config
    development_dir = context.paths.outputs / "development"
    ledger_path = development_dir / "experiment_ledger.jsonl"
    max_target = context.targets[max(config.models.horizons)]
    folds = expanding_calendar_folds(
        context.market.dates,
        max_target.outcome_end_date,
        minimum_train_years=config.walk_forward.minimum_train_years,
        test_years=config.walk_forward.test_years,
        step_years=config.walk_forward.step_years,
    )
    assert_fold_integrity(folds, max_target.outcome_end_date)
    decision_returns = decision_period_returns(
        context.market.levels, config.allocation.execution_lag_sessions
    )
    append_ledger(
        ledger_path,
        {
            "kind": "research_protocol",
            "feature_version": FEATURE_VERSION,
            "feature_variants": list(config.feature_variants),
            "feature_definitions": list(context.features.variants["D"]),
            "price_lookbacks_sessions": [21, 63, 126, 200, 252],
            "target_variants": ["relative_log_return", "voo_outperforms_binary"],
            "forecast_horizons_sessions": list(config.models.horizons),
            "model_families": [
                "ridge_regression",
                "hist_gradient_boosting_regression",
                "logistic_regression",
                "hist_gradient_boosting_classification",
            ],
            "hyperparameter_candidates": {
                "ridge_alphas": list(config.models.ridge_alphas),
                "logistic_c": list(config.models.logistic_c),
                "boosted_leaf_nodes": list(config.models.boosted_leaf_nodes),
            },
            "allocator_methods": list(config.allocation.methods),
            "rebalance_cadences": list(config.allocation.cadences),
            "minimum_changes": list(config.allocation.minimum_changes),
            "execution_lag_sessions": config.allocation.execution_lag_sessions,
        },
    )

    runs: dict[str, dict[str, Any]] = {}
    all_candidate_returns: list[pd.DataFrame] = []
    all_ledger: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    for variant in config.feature_variants:
        if variant == "D" and not context.features.external_available:
            append_ledger(ledger_path, {"kind": "ablation", "variant": "D", "status": "skipped_missing_point_in_time_external_data"})
            continue
        feature_frame = context.features.for_variant(variant)
        stability = feature_fold_stability(feature_frame, folds)
        redundancy = feature_redundancy(feature_frame)
        walk_forward = walk_forward_variant(
            variant,
            feature_frame,
            context.targets,
            folds,
            config.models,
            inner_folds=config.walk_forward.inner_folds,
        )
        calibration_curve, calibration_summary = calibration_diagnostics(
            walk_forward.predictions, context.targets, config.models
        )
        signals = aggregate_horizons(walk_forward.predictions, config.models)
        signals.attrs["feature_count"] = feature_frame.shape[1]
        selected_spec, selected_result, allocator_table = select_allocator(
            signals, decision_returns, config.allocation
        )
        metrics = performance_metrics(selected_result, decision_returns["VOO"])
        rolling = rolling_summary(selected_result.returns, decision_returns["VOO"])
        robustness = cost_and_threshold_robustness(
            signals, decision_returns, config.allocation, selected_spec
        )
        horizon_robustness = horizon_weight_robustness(
            walk_forward.predictions,
            decision_returns,
            config.models,
            config.allocation,
            selected_spec,
        )
        candidate_returns, ledger_records = _candidate_backtests(
            variant, signals, decision_returns, config
        )
        all_candidate_returns.append(candidate_returns)
        all_ledger.extend(ledger_records)
        rolling_lookup = rolling.set_index("years")
        ablation_rows.append(
            {
                "variant": variant,
                "feature_count": feature_frame.shape[1],
                "allocator": selected_spec.name,
                **metrics,
                "rolling_3y_win_rate": rolling_lookup.loc[3, "win_rate"],
                "rolling_5y_win_rate": rolling_lookup.loc[5, "win_rate"],
                "rolling_10y_win_rate": rolling_lookup.loc[10, "win_rate"],
            }
        )
        runs[variant] = {
            "walk_forward": walk_forward,
            "signals": signals,
            "spec": selected_spec,
            "result": selected_result,
            "allocator_table": allocator_table,
            "metrics": metrics,
            "rolling": rolling,
            "robustness": robustness,
            "horizon_robustness": horizon_robustness,
            "calibration_curve": calibration_curve,
            "calibration_summary": calibration_summary,
            "feature_stability": stability,
            "feature_redundancy": redundancy,
        }
        variant_dir = development_dir / f"variant_{variant}"
        atomic_write_csv(variant_dir / "walk_forward_predictions.csv", walk_forward.predictions)
        atomic_write_csv(variant_dir / "walk_forward_metrics.csv", walk_forward.fold_metrics, index=False)
        atomic_write_csv(variant_dir / "selected_parameters.csv", walk_forward.selected_parameters, index=False)
        atomic_write_csv(variant_dir / "allocator_grid.csv", allocator_table, index=False)
        atomic_write_csv(variant_dir / "rolling_outperformance.csv", rolling, index=False)
        atomic_write_csv(variant_dir / "robustness.csv", robustness, index=False)
        atomic_write_csv(
            variant_dir / "horizon_weight_robustness.csv",
            horizon_robustness,
            index=False,
        )
        atomic_write_csv(
            variant_dir / "calibration_reliability.csv", calibration_curve, index=False
        )
        atomic_write_csv(
            variant_dir / "calibration_summary.csv", calibration_summary, index=False
        )
        atomic_write_csv(
            variant_dir / "feature_fold_stability.csv", stability, index=False
        )
        atomic_write_csv(
            variant_dir / "feature_redundancy.csv", redundancy, index=False
        )

    if not runs:
        raise DataIntegrityError("no feature variant could be evaluated")
    for record in all_ledger:
        append_ledger(ledger_path, record)
    ablation = pd.DataFrame(ablation_rows)
    selected_variant = _variant_selection(
        ablation, config.allocation.selection_tolerance_annual_bps
    )
    selected_run = runs[selected_variant]
    selected_spec: AllocatorSpec = selected_run["spec"]
    selected_result = selected_run["result"]
    episodes = historical_episode_analysis(selected_result, decision_returns)
    bear_timing = bear_market_timing_analysis(selected_result, decision_returns)
    development_benchmarks, _ = benchmark_table(
        decision_returns.reindex(selected_result.returns.index),
        config.allocation.transaction_cost_bps,
    )
    candidates = pd.concat(all_candidate_returns, axis=1).dropna()
    rc = reality_check(
        candidates,
        decision_returns["VOO"],
        block_length=21,
        simulations=1000,
        seed=config.models.random_seed,
    )
    bootstrap = block_bootstrap_cagr_excess(
        selected_result.returns,
        decision_returns["VOO"],
        block_length=21,
        simulations=1000,
        seed=config.models.random_seed,
    )
    selected_features = context.features.for_variant(selected_variant)
    final_models = fit_final_models(
        selected_features,
        context.targets,
        config.models,
        inner_folds=config.walk_forward.inner_folds,
    )
    preholdout_hash = sha256_frame(context.market.raw_levels)
    rolling_records = selected_run["rolling"].to_dict(orient="records")
    manifest: dict[str, Any] = {
        "schema_version": "12.0",
        "generated_at": utc_now(),
        "status": "frozen_preholdout",
        "config_digest": config.digest(),
        "feature_version": FEATURE_VERSION,
        "preholdout_data_hash": preholdout_hash,
        "provenance_file_hash": sha256_file(context.paths.provenance),
        "development_period": {
            "start": context.market.dates[0].date().isoformat(),
            "end": context.market.dates[-1].date().isoformat(),
            "outer_folds": len(folds),
        },
        "holdout": {
            "start": config.holdout.start,
            "evaluated": False,
            "marker": str(context.paths.artifacts / "FINAL_HOLDOUT_USED.json"),
        },
        "selected": {
            "variant": selected_variant,
            "feature_names": list(selected_features.columns),
            "allocator": selected_spec.name,
            "allocator_spec": asdict(selected_spec),
            "metrics": selected_run["metrics"],
            "rolling": rolling_records,
            "calibration": selected_run["calibration_summary"].to_dict(orient="records"),
        },
        "statistics": {"reality_check": rc, "bootstrap_cagr_excess": bootstrap},
        "testing_ledger": {
            "feature_variants": len(runs),
            "forecast_horizons": len(config.models.horizons),
            "model_families": 4,
            "allocator_candidates_per_variant": len(allocator_grid(config.allocation)),
            "total_portfolio_candidates": candidates.shape[1],
        },
        "data_warnings": list(context.market.warnings),
    }
    bundle = {
        "manifest": manifest,
        "models": final_models,
        "allocator_spec": selected_spec,
        "variant": selected_variant,
        "feature_names": tuple(selected_features.columns),
    }
    frozen_path = context.paths.artifacts / "frozen_preholdout.joblib"
    atomic_joblib_dump(frozen_path, bundle)
    manifest["frozen_bundle_sha256"] = sha256_file(frozen_path)
    atomic_write_json(context.paths.artifacts / "frozen_preholdout_manifest.json", manifest)
    atomic_write_csv(development_dir / "ablation_results.csv", ablation, index=False)
    atomic_write_csv(development_dir / "selected_returns.csv", selected_result.returns.to_frame())
    atomic_write_csv(development_dir / "selected_weights.csv", selected_result.weights)
    atomic_write_csv(development_dir / "selected_equity.csv", selected_result.equity.to_frame("equity"))
    atomic_write_csv(
        development_dir / "all_candidate_returns.csv", candidates
    )
    atomic_write_csv(
        development_dir / "benchmark_comparison.csv",
        development_benchmarks,
        index=False,
    )
    selected_ledger = execution_ledger(
        selected_result,
        decision_returns,
        context.targets[min(config.models.horizons)].execution_date,
    )
    atomic_write_csv(development_dir / "execution_ledger.csv", selected_ledger)
    atomic_write_csv(development_dir / "historical_episodes.csv", episodes, index=False)
    atomic_write_csv(development_dir / "bear_market_timing.csv", bear_timing, index=False)
    write_audit_database(
        development_dir / "development_audit.sqlite",
        {
            "ablation_results": ablation,
            "selected_signals": selected_run["signals"],
            "selected_weights": selected_result.weights,
            "selected_returns": selected_result.returns.to_frame(),
            "execution_ledger": selected_ledger,
            "benchmarks": development_benchmarks,
            "historical_episodes": episodes,
            "bear_market_timing": bear_timing,
        },
    )
    report = development_report(manifest, ablation, development_benchmarks)
    atomic_write_text(development_dir / "development_report.md", report)
    return {
        "manifest": manifest,
        "frozen_bundle": str(frozen_path),
        "report": str(development_dir / "development_report.md"),
    }


def _rolling_5y_from_manifest(manifest: dict[str, Any]) -> float:
    for row in manifest["selected"]["rolling"]:
        if int(row["years"]) == 5:
            return float(row["win_rate"])
    return float("nan")


def evaluate_final_holdout(
    config_path: str | Path,
    *,
    frozen_bundle: str | Path | None = None,
    acknowledgement: str,
) -> dict[str, Any]:
    config = load_config(config_path)
    paths = project_paths(config_path, config)
    if acknowledgement != config.holdout.acknowledgement:
        raise HoldoutError("exact final-holdout acknowledgement was not supplied")
    marker_path = paths.artifacts / "FINAL_HOLDOUT_USED.json"
    if marker_path.exists():
        raise HoldoutError(
            "the final holdout has already been evaluated; the protocol forbids a second look"
        )
    market = load_market_data(paths.market, paths.provenance)
    validate_holdout_provenance(
        market, config.holdout.start, config.holdout.require_kmlm_live_etf
    )
    bundle_path = (
        Path(frozen_bundle).resolve()
        if frozen_bundle is not None
        else paths.artifacts / "frozen_preholdout.joblib"
    )
    if not bundle_path.exists():
        raise HoldoutError(f"frozen bundle does not exist: {bundle_path}")
    bundle = joblib.load(bundle_path)
    manifest: dict[str, Any] = bundle["manifest"]
    if manifest["config_digest"] != config.digest():
        raise HoldoutError("configuration changed after the candidate was frozen")
    boundary = pd.Timestamp(config.holdout.start).normalize()
    prefix = market.raw_levels.loc[market.dates < boundary]
    if sha256_frame(prefix) != manifest["preholdout_data_hash"]:
        raise HoldoutError("pre-holdout market history changed after freezing")
    if sha256_file(paths.provenance) != manifest["provenance_file_hash"]:
        raise HoldoutError("provenance metadata changed after freezing")

    holdout_dates = market.dates[market.dates >= boundary]
    if len(holdout_dates) / 252.0 < config.holdout.minimum_years:
        raise HoldoutError("final holdout is shorter than the configured minimum")
    # Reserve the final look before any holdout feature, target, forecast, or
    # performance calculation. A failed evaluation remains a consumed holdout.
    atomic_write_json(
        marker_path,
        {
            "state": "evaluation_in_progress",
            "used_at": utc_now(),
            "holdout_start": config.holdout.start,
            "frozen_bundle_sha256": sha256_file(bundle_path),
        },
    )
    context = _context_from_market(config, paths, market)
    feature_frame = context.features.for_variant(bundle["variant"])
    models = bundle["models"]
    raw, all_signals = predict_final_models(models, feature_frame, config.models)
    signals = all_signals.loc[holdout_dates]
    spec: AllocatorSpec = bundle["allocator_spec"]
    weights = allocate(signals, config.allocation, spec)
    decision_returns = decision_period_returns(
        context.market.levels, config.allocation.execution_lag_sessions
    ).loc[holdout_dates]
    result = backtest(
        weights,
        decision_returns,
        transaction_cost_bps=config.allocation.transaction_cost_bps,
    )
    metrics = performance_metrics(result, decision_returns["VOO"])
    rolling = rolling_summary(result.returns, decision_returns["VOO"])
    benchmarks, _ = benchmark_table(
        decision_returns, config.allocation.transaction_cost_bps
    )
    bootstrap = block_bootstrap_cagr_excess(
        result.returns,
        decision_returns["VOO"],
        block_length=21,
        simulations=2000,
        seed=config.models.random_seed,
    )
    gate = deployment_gate(
        metrics,
        _rolling_5y_from_manifest(manifest),
        float(manifest["statistics"]["reality_check"]["pvalue"]),
        context.market.primary_claim_allowed,
        minimum_rolling_win_rate=config.deployment.minimum_preholdout_rolling_5y_win_rate,
        maximum_pvalue=config.deployment.maximum_reality_check_pvalue,
    )
    importance = holdout_permutation_importance(
        models, feature_frame.loc[holdout_dates], context.targets, config.models
    )
    calibration_curve, calibration_summary = calibration_diagnostics(
        raw.loc[holdout_dates], context.targets, config.models
    )
    regime = regime_contribution(
        result, decision_returns, context.market.levels["VOO"]
    )
    episodes = historical_episode_analysis(result, decision_returns)
    bear_timing = bear_market_timing_analysis(result, decision_returns)
    false_signals = false_signal_analysis(result, decision_returns)
    robustness = cost_and_threshold_robustness(
        signals, decision_returns, config.allocation, spec
    )
    horizon_robustness = horizon_weight_robustness(
        raw.loc[holdout_dates],
        decision_returns,
        config.models,
        config.allocation,
        spec,
    )
    holdout_payload: dict[str, Any] = {
        "schema_version": "12.0",
        "evaluated_at": utc_now(),
        "frozen_bundle_sha256": sha256_file(bundle_path),
        "period": {
            "start": result.returns.index[0].date().isoformat(),
            "end": result.returns.index[-1].date().isoformat(),
            "sessions": len(result.returns),
        },
        "metrics": metrics,
        "bootstrap_cagr_excess": bootstrap,
        "calibration": calibration_summary.to_dict(orient="records"),
        "deployment_gate": gate,
        "primary_objective": "PASSED" if gate["passed"] else "FAILED",
    }
    holdout_dir = context.paths.outputs / "final_holdout"
    atomic_write_json(holdout_dir / "final_holdout_result.json", holdout_payload)
    atomic_write_csv(holdout_dir / "predictions.csv", raw.loc[holdout_dates])
    atomic_write_csv(holdout_dir / "signals.csv", signals)
    atomic_write_csv(holdout_dir / "weights.csv", result.weights)
    atomic_write_csv(holdout_dir / "returns.csv", result.returns.to_frame())
    atomic_write_csv(holdout_dir / "equity.csv", result.equity.to_frame("equity"))
    holdout_ledger = execution_ledger(
        result,
        decision_returns,
        context.targets[min(config.models.horizons)].execution_date,
    )
    atomic_write_csv(holdout_dir / "execution_ledger.csv", holdout_ledger)
    atomic_write_csv(holdout_dir / "rolling_outperformance.csv", rolling, index=False)
    atomic_write_csv(holdout_dir / "benchmarks.csv", benchmarks, index=False)
    atomic_write_csv(holdout_dir / "permutation_importance.csv", importance, index=False)
    atomic_write_csv(
        holdout_dir / "calibration_reliability.csv", calibration_curve, index=False
    )
    atomic_write_csv(
        holdout_dir / "calibration_summary.csv", calibration_summary, index=False
    )
    atomic_write_csv(holdout_dir / "regime_contribution.csv", regime, index=False)
    atomic_write_csv(holdout_dir / "historical_episodes.csv", episodes, index=False)
    atomic_write_csv(holdout_dir / "bear_market_timing.csv", bear_timing, index=False)
    atomic_write_csv(holdout_dir / "false_signal_analysis.csv", false_signals, index=False)
    atomic_write_csv(holdout_dir / "robustness.csv", robustness, index=False)
    atomic_write_csv(
        holdout_dir / "horizon_weight_robustness.csv",
        horizon_robustness,
        index=False,
    )
    report = holdout_report(holdout_payload, benchmarks, rolling)
    atomic_write_text(holdout_dir / "final_holdout_report.md", report)
    write_audit_database(
        holdout_dir / "holdout_audit.sqlite",
        {
            "signals": signals,
            "weights": result.weights,
            "returns": result.returns.to_frame(),
            "execution_ledger": holdout_ledger,
            "benchmarks": benchmarks,
            "rolling": rolling,
            "importance": importance,
            "historical_episodes": episodes,
            "bear_market_timing": bear_timing,
        },
    )

    if gate["passed"]:
        live_models = fit_final_models(
            feature_frame,
            context.targets,
            config.models,
            inner_folds=config.walk_forward.inner_folds,
        )
    else:
        live_models = None
    deployment_bundle = {
        "created_at": utc_now(),
        "deployment_allowed": bool(gate["passed"]),
        "failure_reason": gate["failure_message"],
        "config_digest": config.digest(),
        "variant": bundle["variant"],
        "feature_names": bundle["feature_names"],
        "allocator_spec": spec,
        "models": live_models,
        "holdout_result": holdout_payload,
        "training_data_end": context.market.dates[-1].date().isoformat(),
        "training_data_hash": sha256_frame(context.market.raw_levels),
        "provenance_file_hash": sha256_file(context.paths.provenance),
    }
    deployment_path = context.paths.artifacts / "deployment_bundle.joblib"
    atomic_joblib_dump(deployment_path, deployment_bundle)
    holdout_payload["deployment_bundle_sha256"] = sha256_file(deployment_path)
    atomic_write_json(holdout_dir / "final_holdout_result.json", holdout_payload)
    marker = {
        "state": "evaluation_completed",
        "used_at": utc_now(),
        "holdout_result": str(holdout_dir / "final_holdout_result.json"),
        "report": str(holdout_dir / "final_holdout_report.md"),
        "primary_objective": holdout_payload["primary_objective"],
        "frozen_bundle_sha256": sha256_file(bundle_path),
    }
    atomic_write_json(marker_path, marker)
    return {"result": holdout_payload, "report": str(holdout_dir / "final_holdout_report.md")}


def run_live_inference(
    config_path: str | Path, deployment_bundle: str | Path | None = None
) -> dict[str, Any]:
    context = load_research_context(config_path)
    bundle_path = (
        Path(deployment_bundle).resolve()
        if deployment_bundle is not None
        else context.paths.artifacts / "deployment_bundle.joblib"
    )
    if not bundle_path.exists():
        raise DataIntegrityError("deployment bundle is missing; evaluate the final holdout first")
    bundle = joblib.load(bundle_path)
    if bundle["config_digest"] != context.config.digest():
        raise DataIntegrityError("live configuration does not match the deployment bundle")
    training_end = pd.Timestamp(bundle["training_data_end"]).normalize()
    frozen_history = context.market.raw_levels.loc[
        context.market.raw_levels.index <= training_end
    ]
    if sha256_frame(frozen_history) != bundle["training_data_hash"]:
        raise DataIntegrityError("historical market data changed after deployment training")
    if sha256_file(context.paths.provenance) != bundle["provenance_file_hash"]:
        raise DataIntegrityError("provenance changed after deployment training")
    latest_date = context.market.dates[-1]
    payload: dict[str, Any]
    if not bundle["deployment_allowed"]:
        payload = {
            "schema_version": "12.0",
            "signal_date": latest_date.date().isoformat(),
            "deployment_status": "FAILED_ALPHA_GATE_FAIL_CLOSED",
            "allocation": {"VOO": 1.0, "KMLM": 0.0},
            "probability_voo_outperforms": 0.5,
            "directional_confidence": 0.5,
            "expected_relative_return_63_equivalent": 0.0,
            "prediction_interval_80": [None, None],
            "forecast_uncertainty": {
                "prediction_interval_width_80": None,
                "model_agreement_shrinkage": 0.0,
            },
            "horizon_forecasts": {},
            "major_signals": [],
            "failure_reason": bundle["failure_reason"],
        }
    else:
        feature_frame = context.features.for_variant(bundle["variant"])
        latest = feature_frame.loc[[latest_date]]
        raw, signals = predict_final_models(bundle["models"], latest, context.config.models)
        previous_weight = 1.0
        prior_path = context.paths.outputs / "live" / "latest_allocation.json"
        if prior_path.exists():
            with prior_path.open("r", encoding="utf-8") as handle:
                previous_weight = float(json.load(handle)["allocation"]["VOO"])
        weights = allocate(
            signals,
            context.config.allocation,
            bundle["allocator_spec"],
            initial_voo_weight=previous_weight,
        ).iloc[0]
        contributions = pd.Series(dtype=float)
        for horizon, horizon_weight in zip(
            context.config.models.horizons,
            context.config.models.horizon_weights,
            strict=True,
        ):
            local = bundle["models"][horizon].local_contributions(latest)
            contributions = contributions.add(
                local * horizon_weight * (63.0 / horizon), fill_value=0.0
            )
        major = [
            {"feature": name, "contribution": float(value)}
            for name, value in contributions.sort_values(key=np.abs, ascending=False).head(8).items()
        ]
        aggregate = signals.iloc[0]
        horizon_forecasts = {
            str(horizon): {
                "expected_relative_return": float(np.expm1(raw[f"h{horizon}_expected"].iloc[0])),
                "probability_voo_outperforms": float(raw[f"h{horizon}_probability"].iloc[0]),
                "prediction_interval_80": [
                    float(np.expm1(raw[f"h{horizon}_lower"].iloc[0])),
                    float(np.expm1(raw[f"h{horizon}_upper"].iloc[0])),
                ],
            }
            for horizon in context.config.models.horizons
        }
        interval_low = float(np.expm1(aggregate["relative_return_lower_80"]))
        interval_high = float(np.expm1(aggregate["relative_return_upper_80"]))
        payload = {
            "schema_version": "12.0",
            "signal_date": latest_date.date().isoformat(),
            "deployment_status": "PASSED_ALPHA_GATE",
            "allocation": {"VOO": float(weights["VOO"]), "KMLM": float(weights["KMLM"])},
            "probability_voo_outperforms": float(aggregate["probability_voo_outperforms"]),
            "directional_confidence": float(aggregate["directional_confidence"]),
            "expected_relative_return_63_equivalent": float(
                np.expm1(aggregate["expected_relative_log_return_63_equivalent"])
            ),
            "prediction_interval_80": [
                interval_low,
                interval_high,
            ],
            "forecast_uncertainty": {
                "prediction_interval_width_80": interval_high - interval_low,
                "model_agreement_shrinkage": float(
                    aggregate["model_agreement_shrinkage"]
                ),
            },
            "horizon_forecasts": horizon_forecasts,
            "major_signals": major,
            "failure_reason": "",
        }
    live_dir = context.paths.outputs / "live"
    atomic_write_json(live_dir / "latest_allocation.json", payload)
    latest_frame = pd.DataFrame(
        [
            {
                "signal_date": payload["signal_date"],
                "VOO": payload["allocation"]["VOO"],
                "KMLM": payload["allocation"]["KMLM"],
                "probability_voo_outperforms": payload["probability_voo_outperforms"],
                "directional_confidence": payload["directional_confidence"],
                "expected_relative_return_63_equivalent": payload[
                    "expected_relative_return_63_equivalent"
                ],
                "prediction_interval_80_lower": payload["prediction_interval_80"][0],
                "prediction_interval_80_upper": payload["prediction_interval_80"][1],
                "deployment_status": payload["deployment_status"],
            }
        ]
    )
    atomic_write_csv(live_dir / "latest_allocation.csv", latest_frame, index=False)
    history_path = live_dir / "allocation_history.csv"
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    history = pd.concat([history, latest_frame], ignore_index=True)
    history = history.drop_duplicates("signal_date", keep="last").sort_values("signal_date")
    atomic_write_csv(history_path, history, index=False)
    atomic_write_text(live_dir / "latest_report.md", live_report(payload))
    return payload
