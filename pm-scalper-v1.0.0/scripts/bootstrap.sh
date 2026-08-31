#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3.11 -m venv .venv
.venv/bin/python3.11 -m pip install --upgrade pip setuptools wheel
.venv/bin/python3.11 -m pip install -e '.[dev]'
.venv/bin/pm-scalper validate --config config/default.yaml

printf '\nInstalled successfully. Next command:\n'
printf '  .venv/bin/pm-scalper discover --config config/default.yaml\n'
