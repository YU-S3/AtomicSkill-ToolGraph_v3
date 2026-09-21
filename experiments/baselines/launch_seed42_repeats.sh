#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REPEAT_PY="$REPO/.venv_b5_gepa/bin/python"
ENV_FILE="${ENV_FILE:-/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env}"
[[ -x "$REPEAT_PY" ]] || { echo "Missing runtime: $REPEAT_PY" >&2; exit 1; }
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
: "${MODEL_API_KEY:?Fill MODEL_API_KEY in ENV_FILE}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-/home/yangchengyu/.cache/alfworld}"
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
MODE="${1:-}"
OUTPUT="${2:-}"
if [[ "$MODE" == resume ]]; then
  : "${3:?Specify method, e.g. b3_skillopt}"
  : "${4:?Specify repetition number 1 or 2}"
  : "${5:?Specify smoke_qualification.json}"
  exec "$REPEAT_PY" -m experiments.baselines.repeat_test.run --resume \
    --output "$OUTPUT" --method "$3" --repeat "$4" --qualification "$5"
fi
[[ "$MODE" =~ ^(smoke|formal)$ && -n "$OUTPUT" ]] || {
  echo "Usage: bash experiments/baselines/launch_seed42_repeats.sh smoke|formal OUTPUT [QUALIFICATION]" >&2
  exit 2
}
ARGS=("$MODE" --output "$OUTPUT")
if [[ "$MODE" == formal ]]; then
  : "${3:?Specify smoke_qualification.json}"
  ARGS+=(--qualification "$3")
fi
exec "$REPEAT_PY" -m experiments.baselines.repeat_test.launch "${ARGS[@]}"
