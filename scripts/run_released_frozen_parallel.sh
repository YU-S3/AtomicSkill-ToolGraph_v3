#!/usr/bin/env bash
# Three independent streams; seed42 repetitions are sequential, never resume aliases.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?Usage: bash scripts/run_released_frozen_parallel.sh RELEASE_ROOT}"
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
if [[ $# == 1 ]]; then
  for seed in 42 43 44; do
    test -f "$OUT/configs/seed${seed}_rep01.json"
    test ! -e "$OUT/eval/seed${seed}/rep01"
  done
  for seed in 42 43 44; do
    nohup bash "${BASH_SOURCE[0]}" "$OUT" "$seed" > "$OUT/seed${seed}.log" 2>&1 < /dev/null &
    printf 'seed%s supervisor PID=%s; log=%s/seed%s.log\n' "$seed" "$!" "$OUT" "$seed"
  done
  exit 0
fi
seed="$2"
case "$seed" in
  42) repeats=(01 02 03) ;;
  43|44) repeats=(01) ;;
  *) echo 'Seed must be 42, 43 or 44' >&2; exit 2 ;;
esac
exec 9>"$OUT/seed${seed}.lock"
flock -n 9 || { echo "seed${seed} is already running" >&2; exit 1; }
for rep in "${repeats[@]}"; do
  "$ASG_PY" -m experiments.run_v3_released_frozen run --config "$OUT/configs/seed${seed}_rep${rep}.json"
done
