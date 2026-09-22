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
  exec 8>"$OUT/launch.lock"
  flock -n 8 || { echo 'Release streams are already running' >&2; exit 1; }
  "$ASG_PY" -m experiments.released_dev_checks --release-root "$OUT"
  "$ASG_PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
from experiments.protocol import hash_config
root=Path(sys.argv[1]).resolve()
plan=json.loads((root/'evaluation_plan.json').read_text())
assert len(plan['runs']) == 5 and {(r['seed'],r['rep']) for r in plan['runs']} == {(42,1),(42,2),(42,3),(43,1),(44,1)}
for row in plan['runs']:
    path=root/'configs'/f"seed{row['seed']}_rep{row['rep']:02}.json"
    config=json.loads(path.read_text())
    output=root/'eval'/f"seed{row['seed']}"/f"rep{row['rep']:02}"
    assert Path(row['config']).resolve()==path and hash_config(config)==row['config_hash']
    assert Path(config['experiment']['output_dir']).resolve()==output and not output.exists()
    assert config['deployment']['presentation_profile']==plan['profile']
    assert Path(config['bank_release']['release_manifest']).resolve()==root/f"seed{row['seed']}"/'frozen/release_manifest.json'
PY
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
