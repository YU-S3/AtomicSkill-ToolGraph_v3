#!/usr/bin/env bash
# smoke OUTPUT | preflight OUTPUT SMOKE_RECEIPT | formal PREPARED_OUTPUT
# recover SOURCE_CAMPAIGN RECOVERY_OUTPUT SMOKE_RECEIPT
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
B5_PY="$REPO/.venv_b5_gepa/bin/python"
MODE="${1:?Specify smoke, preflight, formal, or recover}"
OUTPUT="${2:?Specify a unique output, or the already prepared directory for formal}"
cd "$REPO"
set -a
. "${ASG_ENV:-/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env}"
set +a
: "${MODEL_API_KEY:?MODEL_API_KEY is missing}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-/home/yangchengyu/.cache/alfworld}"
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONUNBUFFERED=1
ARGS=()
case "$MODE" in
  recover)
    MODULE=experiments.baselines.b5_gepa.recover_campaign
    ARGS+=(--source-campaign "$OUTPUT"
      --smoke-receipt "${4:?Provide current passing smoke_qualification.json}")
    OUTPUT="${3:?Provide a separate recovery directory}"
    ;;
  smoke)
    MODULE=experiments.baselines.b5_gepa.controller
    ARGS+=(--phase smoke --seed 42
      --train-manifest data/baseline_manifests/train_6_smoke.json
      --validation-manifest data/baseline_manifests/validation_6_smoke.json
      --test-manifest data/baseline_manifests/test_6_smoke.json
      --config configs/baselines/b5_gepa_smoke.yaml)
    ;;
  preflight|formal)
    MODULE=experiments.baselines.b5_gepa.run_seed_campaign
    ARGS+=(--seeds 42 43 44
      --train-manifest data/baseline_manifests/train_120.json
      --validation-manifest data/baseline_manifests/validation_24.json
      --test-manifest data/baseline_manifests/test_ood_full_134.json
      --config configs/baselines/b5_gepa.yaml)
    if [[ "$MODE" == preflight ]]; then
      ARGS+=(--preflight-only --smoke-receipt "${3:?Provide current passing smoke_qualification.json}")
    else
      ARGS+=(--prepared)
    fi
    ;;
  *) echo 'Expected smoke, preflight, formal, or recover' >&2; exit 2 ;;
esac
mkdir -p runs/baselines/launch_logs
"$B5_PY" -m "$MODULE" "${ARGS[@]}" --output-dir "$OUTPUT" \
  2>&1 | tee "runs/baselines/launch_logs/b5_${MODE}_$(date -u +%Y%m%dT%H%M%SZ).log"
