#!/usr/bin/env bash
# Thin launcher for existing run_single_cell. Default is DRY RUN.
# Does not patch the repository, change models, migrate Banks, or run ALFWorld/DocVQA.
set -euo pipefail
umask 077

EXECUTE=0
case "${1:-}" in
  "") ;;
  --execute) EXECUTE=1 ;;
  -h|--help)
    cat <<'HELP'
Usage: bash run_non_alfworld_formal.sh [--execute]
Export CODE PYTHON DATASETS CORPUS_ROOT ENV_FILE OUTPUT_ROOT EXPECTED_SHA first.
Optional MODEL_KEY (default deepseek-v4-flash).
Default prints commands only. --execute requires a clean CF4-R1 checkout and a new
output root, then starts four independent seed42 Train->Frozen->Val cells.
No automatic retries/resume/Test. Inspect completion.json in each cell afterwards.
HELP
    exit 0 ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo "Unexpected extra arguments" >&2; exit 2; fi
for key in CODE PYTHON DATASETS CORPUS_ROOT ENV_FILE OUTPUT_ROOT EXPECTED_SHA; do
  if [[ -z ${!key:-} ]]; then echo "Required environment variable is missing: $key" >&2; exit 2; fi
done
MODEL_KEY=${MODEL_KEY:-deepseek-v4-flash}
# Restrict this delivered launcher to the already approved model; no automatic swap.
if [[ "$MODEL_KEY" != deepseek-v4-flash ]]; then
  echo "This launch plan is locked to deepseek-v4-flash; do not silently switch model." >&2; exit 2
fi
BENCHMARKS=(searchqa livemath officeqa spreadsheetbench)
command_for() {
  local b=$1
  COMMAND=("$PYTHON" -m atomic_skillgraph.experiments.run_single_cell
    --config configs/main_experiment_v1.yaml --benchmark "$b" --seed 42
    --datasets "$DATASETS" --output "$OUTPUT_ROOT/$b/seed42"
    --model-key "$MODEL_KEY" --env-file "$ENV_FILE" --stop-after-val)
  if [[ "$b" == officeqa ]]; then COMMAND+=(--corpus-root "$CORPUS_ROOT"); fi
}
printf 'Mode: %s\n' "$([[ $EXECUTE == 1 ]] && echo EXECUTE || echo DRY_RUN)"
printf 'ALFWorld: excluded_by_user_cost; no process.\nDocVQA: unsupported under current text-only lock; no process.\n'
for b in "${BENCHMARKS[@]}"; do
  command_for "$b"
  printf '(cd %q && PYTHONPATH=%q TMPDIR=%q ' "$CODE" "$CODE/src" "$OUTPUT_ROOT/tmp/$b"
  printf '%q ' "${COMMAND[@]}"
  printf ')\n'
done
if (( ! EXECUTE )); then
  echo 'No preflight or model execution has occurred. Apply and test CF4-R1 before --execute.'
  exit 0
fi
for tool in git flock; do command -v "$tool" >/dev/null || { echo "Missing command: $tool" >&2; exit 2; }; done
[[ -d "$CODE" && -x "$PYTHON" && -d "$DATASETS" && -d "$CORPUS_ROOT" && -f "$ENV_FILE" ]] || {
  echo 'A required path is unavailable. No cell started.' >&2; exit 2;
}
[[ -f "$DATASETS/materialization.json" ]] || { echo 'Missing materialization.json' >&2; exit 2; }
ACTUAL_SHA=$(git -C "$CODE" rev-parse HEAD)
[[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]] || { echo 'Unexpected source commit; no cell started.' >&2; exit 2; }
[[ -z $(git -C "$CODE" status --porcelain --untracked-files=no) ]] || {
  echo 'Tracked source is dirty; no cell started.' >&2; exit 2;
}
cd "$CODE"
# These imports and checks have no model calls. Canonical hashes are rechecked by campaign.
CODE="$CODE" MODEL_KEY="$MODEL_KEY" PYTHONPATH="$CODE/src" "$PYTHON" - <<'PY'
from pathlib import Path
import json, os
import yaml
import atomic_skillgraph.empirical as empirical
from atomic_skillgraph.experiments.run_formal import configured_models
expected = Path(os.environ['CODE']).resolve() / 'src'
actual = Path(empirical.__file__).resolve()
if not actual.is_relative_to(expected):
    raise SystemExit('Import resolves outside the chosen checkout')
if empirical.IMPLEMENTATION_REVISION != 'empirical-v3.1-CF4-R1':
    raise SystemExit('CF4-R1 production patch not installed; refusing paid launch')
spec = yaml.safe_load(Path('configs/main_experiment_v1.yaml').read_text())
models = configured_models(json.loads(Path(spec['model_lock']).read_text()))
selected = [m for m in models if os.environ['MODEL_KEY'] in (m.get('model_id'), m.get('display_name'))]
if len(selected) != 1:
    raise SystemExit('The existing CLI must resolve exactly one model')
if selected[0].get('input_modalities') != ['text']:
    raise SystemExit('Model capability lock changed; revise the launch plan explicitly')
print('Source/import/revision/model preflight passed; no model called.')
PY
PYTHONPATH="$CODE/src" "$PYTHON" -m atomic_skillgraph.experiments.run_single_cell --help >/dev/null
# mkdir must fail on an existing root: do not merge with a previous run accidentally.
mkdir "$OUTPUT_ROOT" || { echo 'OUTPUT_ROOT already exists or parent is missing; no cell started.' >&2; exit 2; }
mkdir "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/status" "$OUTPUT_ROOT/locks" "$OUTPUT_ROOT/tmp"
printf 'benchmark\tseed\tplanned_status\n' > "$OUTPUT_ROOT/launch_plan.tsv"
for b in "${BENCHMARKS[@]}"; do printf '%s\t42\tqueued\n' "$b" >> "$OUTPUT_ROOT/launch_plan.tsv"; done
printf 'alfworld\t42\texcluded_by_user_cost\ndocvqa\t42\tunsupported_text_lock\n' >> "$OUTPUT_ROOT/launch_plan.tsv"
printf '%s\n' "$ACTUAL_SHA" > "$OUTPUT_ROOT/source_commit.txt"
run_cell() {
  local b=$1 rc
  exec 9>"$OUTPUT_ROOT/locks/$b.seed42.lock"
  if ! flock -n 9; then echo "Lock held for $b" >&2; return 2; fi
  mkdir -p "$OUTPUT_ROOT/tmp/$b"
  command_for "$b"
  if TMPDIR="$OUTPUT_ROOT/tmp/$b" PYTHONPATH="$CODE/src" "${COMMAND[@]}" >"$OUTPUT_ROOT/logs/$b.seed42.log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  printf '%s\n' "$rc" > "$OUTPUT_ROOT/status/$b.seed42.exit_code"
  return "$rc"
}
pids=()
for b in "${BENCHMARKS[@]}"; do
  run_cell "$b" &
  pids+=("$!")
  printf '%s\n' "$!" > "$OUTPUT_ROOT/status/$b.seed42.supervisor_pid"
done
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    printf '%s process exited normally; inspect completion.json for awaiting_test.\n' "${BENCHMARKS[$i]}"
  else
    printf '%s process failed/interrupted; no automatic retry. Other cells are not restarted.\n' "${BENCHMARKS[$i]}" >&2
    failed=1
  fi
done
exit "$failed"
