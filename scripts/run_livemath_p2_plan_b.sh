#!/usr/bin/env bash
# Prepared versioned derivation: saved Train60 -> one P2 -> Frozen -> conditional Val.
set -euo pipefail
mode=${1:---check};revision=${2:-}
case "$mode" in --check|--execute) ;; *) echo 'Use --check or --execute with the pinned commit' >&2;exit 2;;esac
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${LIVEMATH_P2_PLAN_B_OUTPUT:-/home/yangchengyu/livemath_p2_plan_b_derived_seed42_20261013}
PYTHON=/home/yangchengyu/asg_alfworld_venv/bin/python
cd "$CODE";export PYTHONPATH="$CODE/src"
[[ $(git rev-parse HEAD) == "$revision" ]] && [[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'Pinned clean commit required' >&2;exit 2; }
[[ -f "$OUTPUT/derivation_manifest.json" ]] || { echo 'Use the already prepared Plan B directory' >&2;exit 2; }
exec 9>"$OUTPUT/run.lock";flock -n 9 || { echo 'This derived run is already running' >&2;exit 2; }
"$PYTHON" - "$OUTPUT" <<'PY'
import json,sys
from atomic_skillgraph.experiments.run_derived_choice import check
result=check(sys.argv[1])
print(json.dumps(result,ensure_ascii=False),flush=True)
if not result['billing_ready']:raise SystemExit('Billing gate blocked; no model request issued')
PY
if [[ $mode == --execute ]];then
  exec "$PYTHON" -u -m atomic_skillgraph.experiments.run_derived_choice execute \
    --output "$OUTPUT" --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
fi
