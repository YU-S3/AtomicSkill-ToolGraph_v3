#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 || ($# -eq 1 && "${1:-}" == --check-only) ]]
ROOT=/home/yangchengyu/asg_livemath_p2_r4_paid_seed42_20261012
RUN=/home/yangchengyu/livemath_p2_r4_paid_seed42_20261012
PY=/home/yangchengyu/asg_alfworld_venv/bin/python
cd "$ROOT"
[[ "$(git rev-parse HEAD)" == 3c38842efe13ad6dace2818f5d3e7087fc7611f2 ]]
export PYTHONPATH="$ROOT/src"
"$PY" -m atomic_skillgraph.experiments.run_choice_capability_batch check --plan "$RUN/plan.json"
if [[ "${1:-}" == --check-only ]]; then exit 0; fi
exec flock -n "$RUN/.execute.lock" "$PY" -u -m atomic_skillgraph.experiments.run_choice_capability_batch \
  execute --plan "$RUN/plan.json" --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
