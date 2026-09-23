#!/usr/bin/env bash
# Development validation only; never starts Test134.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?Usage: bash scripts/validate_release4.sh RELEASE_ROOT}"
ASG_PY="${ASG_PY:-/home/yangchengyu/asg_alfworld_venv/bin/python}"
cd "$REPO"
set -a
source "$REPO/.env"
set +a
: "${MODEL_API_KEY:?MODEL_API_KEY is missing in .env}"
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONUNBUFFERED=1
exec 9>"$OUT/dev_validation.lock"
flock -n 9 || { echo 'Release4 dev validation is already running' >&2; exit 1; }
"$ASG_PY" -m experiments.release4_coverage --release-root "$OUT" --workers 3
"$ASG_PY" -m experiments.run_v3_bank_release make-configs --output-root "$OUT" \
  --seeds 42 43 44 --profile current --repeats 42:3 43:1 44:1
echo 'Development acceptance complete. Formal configs are ready; Test134 was NOT started.'
