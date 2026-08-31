#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/pm-scalper record --config config/default.yaml "$@"
