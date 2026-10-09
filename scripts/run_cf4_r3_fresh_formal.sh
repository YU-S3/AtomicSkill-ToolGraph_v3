#!/usr/bin/env bash
# One independent run_single_cell service per benchmark/seed. No automatic resume.
set -euo pipefail
umask 077
MODE=${1:---dry-run}
case "$MODE" in --dry-run|--check|--execute) ;; *) echo 'Use --dry-run, --check or --execute' >&2; exit 2 ;; esac
[[ $# -le 1 ]] || exit 2
CODE=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-/home/yangchengyu/asg_alfworld_venv/bin/python}
DATASETS=${DATASETS:-/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1}
CORPUS_ROOT=${CORPUS_ROOT:-/mnt/d/T3S_exp/SkillCompiler_resources_20261003/raw/officeqa/treasury_bulletins_parsed/transformed}
ENV_FILE=${ENV_FILE:-/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/yangchengyu/cf4_r3_fresh_formal_deepseek_4bench_3seed_20261009_m4}
BENCHMARKS=(searchqa livemath officeqa spreadsheetbench)
SEEDS=(42 43 44)
command_for() {
  COMMAND=("$PYTHON" -u -m atomic_skillgraph.experiments.run_single_cell
    --config configs/main_experiment_v1.yaml --benchmark "$1" --seed "$2"
    --datasets "$DATASETS" --output "$OUTPUT_ROOT/$1/seed$2"
    --model-key deepseek-v4-flash --env-file "$ENV_FILE")
  if [[ $1 == officeqa ]]; then COMMAND+=(--corpus-root "$CORPUS_ROOT"); fi
}
if [[ $MODE == --dry-run ]]; then
  for b in "${BENCHMARKS[@]}"; do for seed in "${SEEDS[@]}"; do
    command_for "$b" "$seed"
    printf '(cd %q && PYTHONPATH=%q ' "$CODE" "$CODE/src"
    printf '%q ' "${COMMAND[@]}"
    printf ')\n'
  done; done
  exit 0
fi
: "${EXPECTED_SHA:?Set EXPECTED_SHA to the approved final commit}"
[[ $(git -C "$CODE" rev-parse HEAD) == "$EXPECTED_SHA" ]] || { echo 'Source commit differs' >&2; exit 2; }
[[ -z $(git -C "$CODE" status --porcelain --untracked-files=no) ]] || { echo 'Tracked source is dirty' >&2; exit 2; }
[[ -x "$PYTHON" && -d "$DATASETS" && -d "$CORPUS_ROOT" && -f "$ENV_FILE" ]] || { echo 'Required path missing' >&2; exit 2; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo 'Use a new output root; refusing reuse' >&2; exit 2; }
cd "$CODE"
CODE="$CODE" DATASETS="$DATASETS" CORPUS_ROOT="$CORPUS_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" PYTHONPATH="$CODE/src" "$PYTHON" - <<'PY'
from pathlib import Path
import hashlib, json, os, subprocess
import yaml
from atomic_skillgraph import empirical
from atomic_skillgraph.experiments.canonical_manifest import ADAPTER_NAMES, ordered_train, verify
from atomic_skillgraph.experiments.run_formal import configured_models, model_settings, resolved_config
from skillcompiler_bench_contracts.livemath import CHOICE_SEED, CHOICE_PROJECTION_VERSION, NORMALIZATION_VERSION
code, datasets, output = (Path(os.environ[k]) for k in ('CODE', 'DATASETS', 'OUTPUT_ROOT'))
assert Path(empirical.__file__).resolve().is_relative_to(code/'src')
assert empirical.IMPLEMENTATION_REVISION == 'empirical-v3.1-CF4-R3'
untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z']).decode().split('\0')
assert not any(Path(n).suffix in {'.py', '.pyw', '.so', '.pyd'} for n in untracked if n)
spec = yaml.safe_load(Path('configs/main_experiment_v1.yaml').read_text())
authority = Path(spec['authority']).resolve()
verify(authority)
material = json.loads((datasets/'materialization.json').read_text())
assert material['authority_sha256'] == hashlib.sha256((authority/'manifest.json').read_bytes()).hexdigest()
assert material['livemath_normalization_version'] == NORMALIZATION_VERSION
assert material['livemath_choice_projection_version'] == CHOICE_PROJECTION_VERSION and material['livemath_choice_seed'] == CHOICE_SEED
models = configured_models(json.loads(Path(spec['model_lock']).read_text()))
model, = [m for m in models if m['model_id'] == 'deepseek-v4-flash']
assert model['input_modalities'] == ['text']
base = model_settings(yaml.safe_load(Path(spec['base_config']).read_text()), model)
profiles = json.loads(Path(spec['benchmark_profiles']).read_text())
assert profiles['scorer_sha256']['spreadsheet.py'] == hashlib.sha256(Path('src/atomic_skillgraph/harness/scorers/spreadsheet.py').read_bytes()).hexdigest()
banks, phases = set(), set()
for benchmark in ('searchqa', 'livemath', 'officeqa', 'spreadsheetbench'):
    train_ids = json.loads((authority/benchmark/'train.json').read_text())['tasks']
    orders = [ordered_train(train_ids, seed) for seed in (42, 43, 44)]
    assert all(sorted(t['task_id'] for t in rows) == sorted(t['task_id'] for t in train_ids) for rows in orders)
    assert len({tuple(t['task_id'] for t in rows) for rows in orders}) == 3
    for seed in (42, 43, 44):
        root = output/benchmark/f'seed{seed}'
        for split in ('train', 'val', 'test'):
            c = resolved_config(base, profiles['profiles'][ADAPTER_NAMES.get(benchmark, benchmark)],
                benchmark, seed, split, root, datasets, authority, os.environ['CORPUS_ROOT'])
            assert Path(c['manifest']) == authority/benchmark/(split+'.json')
            assert Path(c['harness']['evaluator_records']).is_file()
            assert c['experiment']['runtime_mode'] == ('online' if split == 'train' else 'frozen')
            assert Path(c['data_dir']) == root/'train'/('bank' if split == 'train' else 'frozen_bank')
            phases.add(c['experiment']['output_dir'])
            if split == 'train': banks.add(c['data_dir'])
assert len(phases) == 36 and len(banks) == 12
for process in Path('/proc').iterdir():
    if not process.name.isdigit(): continue
    try: arguments = (process/'cmdline').read_bytes().decode().split('\0')
    except (OSError, UnicodeError): continue
    if any(a in {'atomic_skillgraph.experiments.run_single_cell', 'atomic_skillgraph.experiments.run_formal'} for a in arguments):
        raise SystemExit('An experiment is already running; inspect before a fresh launch')
print('36 phase configs / 12 independent Banks / seed orders passed; zero model calls.')
PY
systemctl --user show-environment >/dev/null
docker image inspect sha256:4db520e20cd121f830731c2d0c0bcfbe9cacd254d1817262404cc18e1e757783 >/dev/null
for b in "${BENCHMARKS[@]}"; do for seed in "${SEEDS[@]}"; do
  if systemctl --user cat "asg-cf4r3-m4-$b-s$seed.service" >/dev/null 2>&1; then
    echo "Service name already exists: $b seed$seed; refusing duplicate launch" >&2; exit 2
  fi
done; done
if [[ $MODE == --check ]]; then exit 0; fi
mkdir "$OUTPUT_ROOT"
mkdir "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/tmp"
printf '%s\n' "$EXPECTED_SHA" > "$OUTPUT_ROOT/source_commit.txt"
printf 'benchmark\tseed\tunit\n' > "$OUTPUT_ROOT/units.tsv"
for b in "${BENCHMARKS[@]}"; do for seed in "${SEEDS[@]}"; do
  unit="asg-cf4r3-m4-$b-s$seed"
  mkdir -p "$OUTPUT_ROOT/tmp/$b/seed$seed"
  command_for "$b" "$seed"
  systemd-run --user --unit="$unit" --property=Type=exec --working-directory="$CODE" \
    --setenv="PYTHONPATH=$CODE/src" --setenv="TMPDIR=$OUTPUT_ROOT/tmp/$b/seed$seed" \
    --property="StandardOutput=append:$OUTPUT_ROOT/logs/$b.seed$seed.log" \
    --property="StandardError=append:$OUTPUT_ROOT/logs/$b.seed$seed.log" "${COMMAND[@]}"
  printf '%s\t%s\t%s\n' "$b" "$seed" "$unit" >> "$OUTPUT_ROOT/units.tsv"
done; done
printf 'Started 12 independent Train -> Frozen -> Val -> Test services: %s\n' "$OUTPUT_ROOT"
