#!/usr/bin/env bash
# Finish seed43's completed train with an audited report-only overlay, then test.
set -euo pipefail
REPAIR_REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN=/home/yangchengyu/asg_r1021_parallel_18EnMl/seed43
ASG_PY=/home/yangchengyu/asg_alfworld_venv/bin/python
if [[ "${1:-}" != --worker ]]; then
  nohup bash "$REPAIR_REPO/scripts/resume_seed43_report.sh" --worker >>"$RUN/report_recovery_launcher.log" 2>&1 </dev/null &
  echo "seed43 supervisor PID=$!; log=$RUN/report_recovery_launcher.log"
  exit 0
fi
cd "$RUN"
exec 9>runs/.resume.lock
flock -n 9 || { echo 'seed43 already running via a recovery launcher'; exit 1; }
set -a
. "$REPAIR_REPO/.env"
set +a
: "${MODEL_API_KEY:?MODEL_API_KEY missing}"
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
if [[ ! -f runs/alfworld_train_full_120_r1021_seed43/frozen/data_v3/freeze_manifest.json ]]; then
  "$ASG_PY" "$REPAIR_REPO/scripts/finalize_budget_report.py" \
    --root "$RUN" --config configs/alfworld_train_full_120_r1021_seed43.yaml --finalize \
    2>&1 | tee -a train_report_resume.log
fi
RESUME=()
if [[ -f runs/alfworld_frozen_eval_134_r1021_seed43/run_manifest.json ]]; then RESUME=(--resume); fi
"$ASG_PY" -m experiments.run_v3_frozen_eval \
  --config configs/alfworld_frozen_eval_134_r1021_seed43.yaml "${RESUME[@]}" \
  2>&1 | tee -a test.log
