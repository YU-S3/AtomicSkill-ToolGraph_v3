#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BASELINE_HOME="${BASELINE_HOME:-/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline}"
# Reuse the proven dependency runtime, without installing or modifying it.
B0_PY="${B0_PY:-$BASELINE_HOME/.venv_b5_gepa/bin/python}"
SKILLOPT_ROOT="${SKILLOPT_ROOT:-$BASELINE_HOME/.external/skillopt}"
ENV_FILE="${ENV_FILE:-/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env}"
if [[ ! -x "$B0_PY" ]]; then
  echo "Missing Python runtime: $B0_PY" >&2
  exit 1
fi
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
: "${MODEL_API_KEY:?Fill MODEL_API_KEY in ENV_FILE first}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-/home/yangchengyu/.cache/alfworld}"
export PYTHONPATH="$REPO/src:$REPO:$SKILLOPT_ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
MODE="${1:-}"
OUTPUT="${2:-}"
if [[ -z "$OUTPUT" || ! "$MODE" =~ ^(smoke|formal|resume)$ ]]; then
  echo "Usage: bash experiments/baselines/launch_dynamic.sh smoke|formal|resume OUTPUT [SMOKE_QUALIFICATION]" >&2
  exit 2
fi
ARGS=("$MODE" --output "$OUTPUT" --skillopt-root "$SKILLOPT_ROOT")
if [[ "$MODE" == formal ]]; then
  : "${3:?Formal requires the passing smoke_qualification.json path}"
  ARGS+=(--qualification "$3")
fi
exec "$B0_PY" -m experiments.baselines.b0_dynamic.campaign "${ARGS[@]}"
