#!/usr/bin/env bash
# Fixed source; fresh independent Bank per seed; existing canonical campaign.
set -euo pipefail
umask 077
MODE=${1:---check}
SEED=${2:-42}
case "$MODE" in --check|--run|--resume) ;; *) echo 'Use --check, --run or --resume [42|43|44]' >&2; exit 2;; esac
case "$SEED" in 42|43|44) ;; *) echo 'Seed must be 42, 43 or 44' >&2; exit 2;; esac
[[ $# -le 2 ]] || exit 2
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/home/yangchengyu/asg_alfworld_venv/bin/python}
DATASETS=${DATASETS:-/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1}
ENV_FILE=${ENV_FILE:-/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env}
: "${EXPECTED_SHA:?Set the reviewed full source commit}"
cd "$CODE"
[[ $(git rev-parse HEAD) == "$EXPECTED_SHA" && -z $(git status --porcelain --untracked-files=no) ]] || {
  echo 'Source identity changed; no launch' >&2; exit 2;
}
export PYTHONPATH="$CODE/src" PYTHONUNBUFFERED=1
"$PYTHON" scripts/check_docvqa_v41.py --datasets "$DATASETS"
[[ "$MODE" != --check ]] || exit 0
: "${OUTPUT_ROOT:?Set an independent formal output root}"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT=$(cd "$OUTPUT_ROOT" && pwd)
exec 9>"$OUTPUT_ROOT/seed$SEED.lock"
flock -n 9 || { echo 'This seed is already running' >&2; exit 2; }
OUTPUT="$OUTPUT_ROOT/seed$SEED"
ARGS=()
if [[ "$MODE" == --resume ]]; then
  [[ -f "$OUTPUT/run_manifest.json" ]] || { echo 'No run to resume' >&2; exit 2; }
  ARGS+=(--resume)
else
  mkdir "$OUTPUT" || { echo 'Fresh output already exists; inspect before resuming' >&2; exit 2; }
fi
exec "$PYTHON" -m atomic_skillgraph.experiments.run_single_cell \
  --config configs/docvqa_deepseek_v41.yaml --benchmark docvqa --seed "$SEED" \
  --datasets "$DATASETS" --output "$OUTPUT" --env-file "$ENV_FILE" --model-key deepseek-flash "${ARGS[@]}"
