#!/usr/bin/env bash
# Separate Linux checkouts/banks/logs; each seed trains before its own frozen eval.
set -euo pipefail
MODE=${1:---probe}
ASG_PY=${ASG_PY:-/home/yangchengyu/asg_alfworld_venv/bin/python}
if [[ "$MODE" == --worker ]]; then
  cd "$2"
  SEED=$3
  export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
  export TMPDIR="$PWD/.tmp"
  "$ASG_PY" -m experiments.run_v3_train --config "configs/alfworld_train_full_120_r1021_seed${SEED}.yaml" 2>&1 | tee train.log
  "$ASG_PY" -m experiments.run_v3_frozen_eval --config "configs/alfworld_frozen_eval_134_r1021_seed${SEED}.yaml" 2>&1 | tee test.log
  exit 0
fi
[[ "$MODE" == --probe || "$MODE" == --launch ]] || { echo 'Use --probe or --launch' >&2; exit 2; }
REPO=$(cd "$(dirname "$0")/.." && pwd)
[[ -x "$ASG_PY" && -f "$REPO/.env" ]]
set -a
. "$REPO/.env"
set +a
: "${MODEL_API_KEY:?MODEL_API_KEY missing}"
export ASG_PY
BATCH=$(mktemp -d "$HOME/asg_r1021_parallel_XXXXXX")
echo "Batch: $BATCH"
PROBE_PIDS=()
for SEED in 43 44; do
  RUN="$BATCH/seed$SEED"
  git clone --quiet --no-hardlinks --single-branch --branch main "$REPO" "$RUN"
  mkdir "$RUN/.tmp"
  (
    cd "$RUN"
    export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 TMPDIR="$PWD/.tmp"
    "$ASG_PY" -m experiments.run_v3_smoke --provider-probe --config "configs/alfworld_train_full_120_r1021_seed${SEED}.yaml"
  ) >"$RUN/provider_probe.log" 2>&1 &
  PROBE_PIDS+=("$!")
done
FAILED=0
for PID in "${PROBE_PIDS[@]}"; do wait "$PID" || FAILED=1; done
[[ "$FAILED" == 0 ]] || { echo "Probe failed; no training started. See $BATCH/seed*/provider_probe.log" >&2; exit 1; }
echo 'Both concurrent provider probes passed.'
[[ "$MODE" == --launch ]] || exit 0
for SEED in 43 44; do
  RUN="$BATCH/seed$SEED"
  nohup bash "$RUN/scripts/run_r1021_remaining_seeds.sh" --worker "$RUN" "$SEED" >"$RUN/launcher.log" 2>&1 </dev/null &
  PID=$!
  printf '%s\n' "$PID" >"$RUN/launcher.pid"
  echo "seed$SEED supervisor PID=$PID; directory=$RUN"
done
