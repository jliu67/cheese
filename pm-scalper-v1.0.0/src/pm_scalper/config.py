from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ApiConfig:
    gamma_url: str = "https://gamma-api.polymarket.com"
    clob_url: str = "https://clob.polymarket.com"
    websocket_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    request_timeout_seconds: float = 20.0
    discovery_page_size: int = 100
    discovery_pages: int = 5
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    heartbeat_seconds: float = 10.0
    book_refresh_seconds: float = 30.0


@dataclass(slots=True)
class UniverseConfig:
    max_markets: int = 30
    include_outcomes: list[str] = field(default_factory=lambda: ["Yes", "No"])
    min_liquidity_usdc: float = 10_000.0
    min_volume_24h_usdc: float = 2_500.0
    min_price: float = 0.05
    max_price: float = 0.95
    max_spread_absolute: float = 0.03
    min_top_depth_usdc: float = 1_000.0
    depth_levels: int = 5
    min_hours_to_end: float = 2.0
    max_seconds_delay: int = 0
    hydrate_market_details: bool = True


@dataclass(slots=True)
class StrategyConfig:
    warmup_seconds: int = 120
    decision_interval_seconds: float = 2.0
    dashboard_interval_seconds: float = 1.0
    equity_sample_seconds: float = 10.0
    signal_log_seconds: float = 10.0
    entry_score: float = 69.0
    exit_score: float = 43.0
    target_return: float = 0.01
    stop_loss: float = 0.0125
    minimum_hold_seconds: int = 45
    maximum_hold_seconds: int = 2_700
    entry_timeout_seconds: int = 90
    cooldown_after_exit_seconds: int = 120
    stale_book_seconds: int = 30
    momentum_15s_scale: float = 0.003
    momentum_60s_scale: float = 0.008
    momentum_300s_scale: float = 0.02
    microprice_scale: float = 0.003
    volatility_scale: float = 0.01
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "momentum": 0.30,
            "book_imbalance": 0.20,
            "trade_flow": 0.18,
            "volume_acceleration": 0.10,
            "microprice": 0.14,
            "spread_penalty": 0.16,
            "volatility_penalty": 0.08,
        }
    )


@dataclass(slots=True)
class ExecutionConfig:
    mode: str = "paper"
    entry_style: str = "maker"
    improve_bid_ticks: int = 1
    queue_ahead_fraction: float = 1.0
    target_queue_ahead_fraction: float = 1.0
    default_taker_fee_rate: float = 0.07
    default_fee_exponent: float = 1.0
    taker_slippage_buffer_ticks: int = 0
    quantity_decimals: int = 4
    flatten_on_shutdown: bool = True


@dataclass(slots=True)
class RiskConfig:
    starting_cash_usdc: float = 10_000.0
    target_position_usdc: float = 1_000.0
    max_position_pct_equity: float = 0.15
    max_total_exposure_pct_equity: float = 0.50
    max_open_positions: int = 3
    max_trades_per_day: int = 30
    max_daily_loss_pct: float = 0.02
    max_consecutive_losses: int = 5
    depth_utilization: float = 0.20
    flatten_on_kill_switch: bool = True


@dataclass(slots=True)
class StorageConfig:
    data_dir: str = "data"
    sqlite_path: str = "data/pm_scalper.sqlite3"
    raw_event_dir: str = "data/raw"
    export_dir: str = "exports"
    log_dir: str = "logs"
    raw_flush_every: int = 25


@dataclass(slots=True)
class AppConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    def validate(self) -> None:
        errors: list[str] = []
        if self.execution.mode.lower() != "paper":
            errors.append("execution.mode must remain 'paper'; this build has no live-order path")
        if self.execution.entry_style.lower() != "maker":
            errors.append("execution.entry_style must be 'maker' in V1")
        if not 0 < self.strategy.target_return < 1:
            errors.append("strategy.target_return must be between 0 and 1")
        if not 0 < self.strategy.stop_loss < 1:
            errors.append("strategy.stop_loss must be between 0 and 1")
        if self.strategy.exit_score >= self.strategy.entry_score:
            errors.append("strategy.exit_score must be below strategy.entry_score")
        if self.risk.starting_cash_usdc <= 0:
            errors.append("risk.starting_cash_usdc must be positive")
        if self.risk.target_position_usdc <= 0:
            errors.append("risk.target_position_usdc must be positive")
        if not 0 < self.risk.max_position_pct_equity <= 1:
            errors.append("risk.max_position_pct_equity must be in (0, 1]")
        if not 0 < self.risk.max_total_exposure_pct_equity <= 1:
            errors.append("risk.max_total_exposure_pct_equity must be in (0, 1]")
        if self.risk.max_position_pct_equity > self.risk.max_total_exposure_pct_equity:
            errors.append("max_position_pct_equity cannot exceed max_total_exposure_pct_equity")
        if self.api.book_refresh_seconds < 0:
            errors.append("api.book_refresh_seconds cannot be negative")
        if not 0 <= self.execution.queue_ahead_fraction <= 5:
            errors.append("execution.queue_ahead_fraction must be between 0 and 5")
        if not 0 <= self.execution.target_queue_ahead_fraction <= 5:
            errors.append("execution.target_queue_ahead_fraction must be between 0 and 5")
        if not 0 < self.risk.depth_utilization <= 1:
            errors.append("risk.depth_utilization must be in (0, 1]")
        if self.universe.min_price <= 0 or self.universe.max_price >= 1:
            errors.append("universe price bounds must stay inside (0, 1)")
        if self.universe.min_price >= self.universe.max_price:
            errors.append("universe.min_price must be below universe.max_price")
        if errors:
            raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type[Any], raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    known = {item.name for item in cls.__dataclass_fields__.values()}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown keys in {cls.__name__}: {', '.join(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    allowed = {"api", "universe", "strategy", "execution", "risk", "storage"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {', '.join(unknown)}")
    config = AppConfig(
        api=_construct(ApiConfig, raw.get("api")),
        universe=_construct(UniverseConfig, raw.get("universe")),
        strategy=_construct(StrategyConfig, raw.get("strategy")),
        execution=_construct(ExecutionConfig, raw.get("execution")),
        risk=_construct(RiskConfig, raw.get("risk")),
        storage=_construct(StorageConfig, raw.get("storage")),
    )
    config.validate()
    return config
