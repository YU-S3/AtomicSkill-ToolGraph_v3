#!/usr/bin/env bash
# Usage: bash experiments/baselines/launch_embodiskill.sh smoke|load-probe|formal|resume OUTPUT
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASG_PY="${ASG_PY:-/home/yangchengyu/asg_alfworld_venv/bin/python}"
ASG_ENV="${ASG_ENV:-/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env}"
MODE="${1:?Specify smoke, formal, or resume}"
OUTPUT="${2:?Specify a unique output directory, or the original directory for resume}"
cd "$REPO"
set -a
. "$ASG_ENV"
set +a
: "${MODEL_API_KEY:?MODEL_API_KEY is missing}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-/home/yangchengyu/.cache/alfworld}"
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONUNBUFFERED=1
ARGS=()
case "$MODE" in
  smoke) ARGS+=(--smoke) ;;
  load-probe) ARGS+=(--load-probe-only) ;;
  formal) ;;
  resume) ARGS+=(--resume) ;;
  *) echo "Expected smoke, load-probe, formal, or resume" >&2; exit 2 ;;
esac
mkdir -p runs/baselines/launch_logs
"$ASG_PY" -m experiments.baselines.b4_embodiskill.campaign \
  --output "$OUTPUT" "${ARGS[@]}" \
  2>&1 | tee "runs/baselines/launch_logs/b4_${MODE}_$(date -u +%Y%m%dT%H%M%SZ).log"
