#!/usr/bin/env bash
# Production regressions and fixed development only; never starts Test134.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?Usage: bash scripts/validate_release5.sh RELEASE_ROOT}"
ASG_PY="${ASG_PY:-/home/yangchengyu/asg_alfworld_venv/bin/python}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONUNBUFFERED=1
"$ASG_PY" -m pytest -q
git diff --check
bash -n scripts/request_released_stop.sh scripts/validate_release5.sh scripts/run_released_frozen_parallel.sh
bash scripts/validate_release4.sh "$OUT"
"$ASG_PY" -m experiments.released_stream --release-root "$OUT"
