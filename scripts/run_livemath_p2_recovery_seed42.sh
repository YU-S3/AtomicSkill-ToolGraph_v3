#!/usr/bin/env bash
set -euo pipefail
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PLAN=${1:-/home/yangchengyu/livemath_p2_contract_recovery_seed42_20261012/plan.json}
PY=/home/yangchengyu/asg_alfworld_venv/bin/python
ENV_FILE=/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
cd "$CODE"
export PYTHONPATH="$CODE/src"
test -f "$PLAN"
exec 9>"$(dirname -- "$PLAN")/.execute.lock"
flock -n 9 || { echo 'This P2 recovery is already running' >&2; exit 2; }
"$PY" -m atomic_skillgraph.experiments.run_choice_capability_batch check --plan "$PLAN"
exec "$PY" -m atomic_skillgraph.experiments.run_choice_capability_batch execute \
  --plan "$PLAN" --env-file "$ENV_FILE"
