"""Command-line interface for the frozen research workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import json_safe
from .config import load_config
from .pipeline import (
    evaluate_final_holdout,
    inspect_data,
    project_paths,
    run_development,
    run_live_inference,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qre2", description="VOO/KMLM relative-return engine")
    parser.add_argument("--config", default="config/research.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect-data", help="validate price/proxy provenance and coverage")
    subparsers.add_parser("develop", help="run pre-holdout walk-forward research and freeze once")
    evaluate = subparsers.add_parser(
        "evaluate-holdout", help="evaluate the untouched final holdout exactly once"
    )
    evaluate.add_argument("--frozen-bundle")
    evaluate.add_argument("--acknowledgement", required=True)
    live = subparsers.add_parser("live", help="write the latest allocation report")
    live.add_argument("--deployment-bundle")
    subparsers.add_parser("status", help="show freeze/holdout/deployment status")
    return parser


def _status(config_path: str) -> dict[str, object]:
    config = load_config(config_path)
    paths = project_paths(config_path, config)
    manifest = paths.artifacts / "frozen_preholdout_manifest.json"
    marker = paths.artifacts / "FINAL_HOLDOUT_USED.json"
    deployment = paths.artifacts / "deployment_bundle.joblib"
    output: dict[str, object] = {
        "frozen_candidate_exists": manifest.exists(),
        "final_holdout_used": marker.exists(),
        "deployment_bundle_exists": deployment.exists(),
    }
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as handle:
            frozen = json.load(handle)
        output["selected_variant"] = frozen["selected"]["variant"]
        output["selected_allocator"] = frozen["selected"]["allocator"]
        output["holdout_start"] = frozen["holdout"]["start"]
    if marker.exists():
        with marker.open("r", encoding="utf-8") as handle:
            used = json.load(handle)
        output["holdout_state"] = used.get("state", "evaluation_completed")
        output["primary_objective"] = used.get("primary_objective", "NOT_AVAILABLE")
        output["holdout_used_at"] = used["used_at"]
    return output


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect-data":
        result = inspect_data(args.config)
    elif args.command == "develop":
        result = run_development(args.config)
    elif args.command == "evaluate-holdout":
        result = evaluate_final_holdout(
            args.config,
            frozen_bundle=args.frozen_bundle,
            acknowledgement=args.acknowledgement,
        )
    elif args.command == "live":
        result = run_live_inference(args.config, args.deployment_bundle)
    elif args.command == "status":
        result = _status(args.config)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(json_safe(result), indent=2, sort_keys=True, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
