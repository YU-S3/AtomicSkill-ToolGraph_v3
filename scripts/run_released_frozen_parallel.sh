#!/usr/bin/env bash
# Fixed matrix; completed runs are preserved, unfinished runs need explicit resume.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?Usage: bash scripts/run_released_frozen_parallel.sh RELEASE_ROOT [all|42-first|42|43|44] [--resume]}"
MODE="${2:-all}"
ASG_PY="${ASG_PY:-/home/yangchengyu/asg_alfworld_venv/bin/python}"
cd "$REPO"
test -x "$ASG_PY"
test -f "$OUT/evaluation_plan.json"
test -f "$REPO/.env"
set -a
source "$REPO/.env"
set +a
: "${MODEL_API_KEY:?MODEL_API_KEY is missing in .env}"
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONUNBUFFERED=1
extra=()
if [[ ${3:-} == --resume ]]; then extra+=(--resume)
elif [[ $# -gt 2 ]]; then echo 'Only --resume is supported' >&2; exit 2; fi
case "$MODE" in
  all) seeds=(42 43 44) ;;
  42-first) seeds=(42); extra+=(--first-only) ;;
  42|43|44) seeds=("$MODE") ;;
  *) echo 'Select all, 42-first, 42, 43 or 44' >&2; exit 2 ;;
esac
exec 8>"$OUT/launch.lock"
flock -n 8 || { echo 'Release launch already in progress' >&2; exit 1; }
"$ASG_PY" -m experiments.released_stream --release-root "$OUT"
for seed in "${seeds[@]}"; do
  nohup "$ASG_PY" -m experiments.released_stream --release-root "$OUT" --seed "$seed" "${extra[@]}" \
    >> "$OUT/seed${seed}.log" 2>&1 < /dev/null 8>&- &
  printf 'seed%s supervisor PID=%s; log=%s/seed%s.log\n' "$seed" "$!" "$OUT" "$seed"
done
