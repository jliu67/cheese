"""Small, explicit configuration surface for the two-asset engine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError


@dataclass(frozen=True)
class WalkForwardConfig:
    minimum_train_years: int = 8
    test_years: int = 1
    step_years: int = 1
    inner_folds: int = 3
    sampling: str = "weekly"


@dataclass(frozen=True)
class ModelConfig:
    horizons: tuple[int, ...] = (21, 63, 126)
    horizon_weights: tuple[float, ...] = (0.20, 0.35, 0.45)
    random_seed: int = 1729
    ridge_alphas: tuple[float, ...] = (1.0, 10.0)
    logistic_c: tuple[float, ...] = (0.1, 1.0)
    boosted_leaf_nodes: tuple[int, ...] = (7, 15)
    maximum_features: int = 40


@dataclass(frozen=True)
class AllocationConfig:
    neutral_voo_weight: float = 0.70
    maximum_tilt: float = 0.70
    methods: tuple[str, ...] = ("continuous", "bucketed", "probability")
    cadences: tuple[str, ...] = ("daily", "weekly")
    minimum_changes: tuple[float, ...] = (0.05, 0.10)
    buckets: tuple[float, ...] = (0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0)
    execution_lag_sessions: int = 1
    transaction_cost_bps: float = 5.0
    selection_tolerance_annual_bps: float = 25.0


@dataclass(frozen=True)
class HoldoutConfig:
    start: str = "2020-12-02"
    minimum_years: float = 3.0
    require_kmlm_live_etf: bool = True
    acknowledgement: str = "I_UNDERSTAND_THIS_USES_THE_FINAL_HOLDOUT"


@dataclass(frozen=True)
class DeploymentConfig:
    minimum_preholdout_rolling_5y_win_rate: float = 0.50
    maximum_reality_check_pvalue: float = 0.10


@dataclass(frozen=True)
class EngineConfig:
    project_name: str = "QRE VOO-KMLM"
    market_data: str = "data/market_levels.csv"
    provenance: str = "data/provenance.yaml"
    macro_data: str | None = None
    output_directory: str = "outputs"
    artifact_directory: str = "artifacts"
    feature_variants: tuple[str, ...] = ("A", "B", "C", "D")
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    holdout: HoldoutConfig = field(default_factory=HoldoutConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    def validate(self) -> None:
        if self.models.horizons != (21, 63, 126):
            raise ConfigError("the frozen research protocol permits only 21/63/126-session horizons")
        if len(self.models.horizon_weights) != len(self.models.horizons):
            raise ConfigError("horizon_weights must match horizons")
        if abs(sum(self.models.horizon_weights) - 1.0) > 1e-9:
            raise ConfigError("horizon_weights must sum to one")
        if any(weight < 0 for weight in self.models.horizon_weights):
            raise ConfigError("horizon_weights may not be negative")
        if not self.feature_variants or len(set(self.feature_variants)) != len(
            self.feature_variants
        ):
            raise ConfigError("feature_variants must be nonempty and unique")
        if not set(self.feature_variants).issubset({"A", "B", "C", "D"}):
            raise ConfigError("feature_variants must be drawn from A/B/C/D")
        if self.walk_forward.minimum_train_years < 5:
            raise ConfigError("minimum_train_years must be at least five")
        if self.walk_forward.test_years < 1 or self.walk_forward.step_years < 1:
            raise ConfigError("walk-forward test and step years must be positive")
        if self.walk_forward.inner_folds < 2:
            raise ConfigError("inner_folds must be at least two")
        if self.walk_forward.sampling != "weekly":
            raise ConfigError("the frozen model-fitting sample is weekly")
        if not self.models.ridge_alphas or any(value <= 0 for value in self.models.ridge_alphas):
            raise ConfigError("ridge alphas must be positive")
        if not self.models.logistic_c or any(value <= 0 for value in self.models.logistic_c):
            raise ConfigError("logistic C values must be positive")
        if not self.models.boosted_leaf_nodes or any(
            value < 2 for value in self.models.boosted_leaf_nodes
        ):
            raise ConfigError("boosted leaf-node candidates must be at least two")
        if self.allocation.execution_lag_sessions < 1:
            raise ConfigError("execution lag must be at least one session")
        if not 0 <= self.allocation.neutral_voo_weight <= 1:
            raise ConfigError("neutral VOO weight must be between zero and one")
        if not 0 <= self.allocation.maximum_tilt <= 1:
            raise ConfigError("maximum tilt must be between zero and one")
        if not self.allocation.methods or not set(self.allocation.methods).issubset(
            {"continuous", "bucketed", "probability"}
        ):
            raise ConfigError("allocation methods are invalid")
        if not self.allocation.cadences or not set(self.allocation.cadences).issubset(
            {"daily", "weekly"}
        ):
            raise ConfigError("allocation cadences are invalid")
        if not self.allocation.minimum_changes or any(
            not 0 <= value <= 1 for value in self.allocation.minimum_changes
        ):
            raise ConfigError("minimum allocation changes must be between zero and one")
        if not self.allocation.buckets or any(
            not 0 <= value <= 1 for value in self.allocation.buckets
        ):
            raise ConfigError("allocation buckets must be between zero and one")
        if tuple(sorted(set(self.allocation.buckets))) != self.allocation.buckets:
            raise ConfigError("allocation buckets must be sorted and unique")
        if self.allocation.transaction_cost_bps < 0:
            raise ConfigError("transaction cost may not be negative")
        if self.allocation.selection_tolerance_annual_bps < 0:
            raise ConfigError("selection tolerance may not be negative")
        try:
            date.fromisoformat(str(self.holdout.start))
        except ValueError as error:
            raise ConfigError("holdout start must be an ISO date") from error
        if self.holdout.minimum_years <= 0 or not self.holdout.acknowledgement:
            raise ConfigError("holdout years and acknowledgement must be positive/nonempty")
        if not 0 <= self.deployment.minimum_preholdout_rolling_5y_win_rate <= 1:
            raise ConfigError("rolling win-rate gate must be between zero and one")
        if not 0 <= self.deployment.maximum_reality_check_pvalue <= 1:
            raise ConfigError("Reality Check p-value gate must be between zero and one")
        if not 1 <= self.models.maximum_features <= 40:
            raise ConfigError("maximum_features must be between one and 40")

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _construct(raw: dict[str, Any]) -> EngineConfig:
    known = {
        "project_name",
        "market_data",
        "provenance",
        "macro_data",
        "output_directory",
        "artifact_directory",
        "feature_variants",
        "walk_forward",
        "models",
        "allocation",
        "holdout",
        "deployment",
    }
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown configuration keys: {sorted(unknown)}")
    payload = dict(raw)
    payload["walk_forward"] = WalkForwardConfig(**payload.get("walk_forward", {}))
    payload["models"] = ModelConfig(**payload.get("models", {}))
    payload["allocation"] = AllocationConfig(**payload.get("allocation", {}))
    holdout_raw = dict(payload.get("holdout", {}))
    if "start" in holdout_raw:
        value = holdout_raw["start"]
        holdout_raw["start"] = value.isoformat() if isinstance(value, date) else str(value)
    payload["holdout"] = HoldoutConfig(**holdout_raw)
    payload["deployment"] = DeploymentConfig(**payload.get("deployment", {}))
    for key in ("feature_variants",):
        if key in payload:
            payload[key] = tuple(payload[key])
    model_raw = dict(raw.get("models", {}))
    for key in ("horizons", "horizon_weights", "ridge_alphas", "logistic_c", "boosted_leaf_nodes"):
        if key in model_raw:
            model_raw[key] = tuple(model_raw[key])
    payload["models"] = ModelConfig(**model_raw)
    allocation_raw = dict(raw.get("allocation", {}))
    for key in ("methods", "cadences", "minimum_changes", "buckets"):
        if key in allocation_raw:
            allocation_raw[key] = tuple(allocation_raw[key])
    payload["allocation"] = AllocationConfig(**allocation_raw)
    config = EngineConfig(**payload)
    config.validate()
    return config


def load_config(path: str | Path) -> EngineConfig:
    source = Path(path)
    if not source.exists():
        raise ConfigError(f"configuration does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    return _construct(raw)


def resolve_path(config_path: str | Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path(config_path).resolve().parent / path
