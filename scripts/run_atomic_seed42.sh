#!/usr/bin/env bash
# Four independent cells. Prefix is part of formal Train; resume uses the same run.
set -euo pipefail
mode=${1:---print}
case "$mode" in --print|--execute|--resume) ;; *) echo 'Use --print, --execute or --resume' >&2; exit 2;; esac
for key in CODE PYTHON DATASETS CORPUS_ROOT ENV_FILE OUTPUT_ROOT EXPECTED_SHA; do
  [[ -n ${!key:-} ]] || { echo "Missing $key" >&2; exit 2; }
done
cd "$CODE"
[[ $(git rev-parse HEAD) == "$EXPECTED_SHA" && -z $(git status --porcelain --untracked-files=no) ]] || {
  echo 'Source identity changed; no cell started' >&2; exit 2;
}
benchmarks=(searchqa livemath officeqa spreadsheetbench)
commands() {
  command=("$PYTHON" -m atomic_skillgraph.experiments.run_single_cell
    --config configs/atomic_unified_seed42.yaml --benchmark "$1" --seed 42
    --model-key deepseek-v4-flash --datasets "$DATASETS" --output "$OUTPUT_ROOT/$1/seed42" --env-file "$ENV_FILE")
  if [[ $1 == officeqa ]]; then command+=(--corpus-root "$CORPUS_ROOT"); fi
  if [[ $mode == --resume ]]; then command+=(--resume); else command+=(--max-new-tasks 5 --stop-after-val); fi
}
for benchmark in "${benchmarks[@]}"; do
  commands "$benchmark"; printf '%q ' "${command[@]}"; printf '\n'
done
[[ $mode != --print ]] || exit 0
if [[ $mode == --execute ]]; then mkdir "$OUTPUT_ROOT"; else [[ -d $OUTPUT_ROOT ]] || exit 2; fi
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks" "$OUTPUT_ROOT/tmp"
cp configs/atomic_unified_seed42_budget.json "$OUTPUT_ROOT/budget_allocation.json"
printf '%s\n' "$EXPECTED_SHA" > "$OUTPUT_ROOT/source_commit.txt"
pids=()
for benchmark in "${benchmarks[@]}"; do
  commands "$benchmark"
  mkdir -p "$OUTPUT_ROOT/tmp/$benchmark"
  (
    exec 9>"$OUTPUT_ROOT/locks/$benchmark.lock"
    flock -n 9 || exit 2
    export PYTHONPATH="$CODE/src" TMPDIR="$OUTPUT_ROOT/tmp/$benchmark"
    exec "${command[@]}"
  ) >>"$OUTPUT_ROOT/logs/$benchmark.seed42.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" > "$OUTPUT_ROOT/logs/$benchmark.seed42.pid"
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
