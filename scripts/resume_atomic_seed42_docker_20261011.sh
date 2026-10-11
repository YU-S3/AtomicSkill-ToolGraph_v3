#!/usr/bin/env bash
# Resume the two audited Docker interruptions using their original pinned source.
set -euo pipefail
mode=${1:---print}
case "$mode" in --print|--execute) ;; *) echo 'Use --print or --execute' >&2; exit 2;; esac
CODE=/home/yangchengyu/asg_atomic_unified_r1_selector_20261010
OUTPUT_ROOT=/home/yangchengyu/atomic_unified_r1_selector_seed42_formal_20261010
PYTHON=/home/yangchengyu/asg_alfworld_venv/bin/python
DATASETS=/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1
ENV_FILE=/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
cd "$CODE"
export PYTHONPATH="$CODE/src"
"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import hashlib,json,sqlite3,subprocess,sys
from pathlib import Path
from atomic_skillgraph.experiments.run_empirical import code_identity

root=Path(sys.argv[1]);source=code_identity()
assert source['git_sha']=='80762b4c84fc71d56a42d9783f1fdd3e70ce43ab' and not source['tracked_dirty'], 'Pinned source changed'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tree=lambda p:{str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file()}
allocation=json.loads(Path('configs/atomic_unified_seed42_budget.json').read_text())
for name,split,task in [('livemath','test','livemath:202512:3'),('spreadsheetbench','train','spreadsheet:43589')]:
    cell=root/name/'seed42';phase=cell/split
    assert json.loads((cell/'run_manifest.json').read_text())['source_commit']==source['git_sha']
    assert json.loads((phase/'execution_manifest.json').read_text())['code']==source
    ledger=json.loads((cell/'budget.json').read_text())
    assert ledger['limits']=={k:allocation['cells'][name][k] for k in ('token_limit','finish_reserve','request_limit')}
    assert not ledger['unknown_billing'] and all(a['status']=='metered' for a in ledger['attempts'].values())
    backup=cell/'infrastructure_recovery'/(hashlib.sha256(task.encode()).hexdigest()[:16]+'_attempt1')
    receipt=json.loads((backup/'receipt.json').read_text())
    assert receipt['applied'] and receipt['budget_unchanged'] and receipt['banks_unchanged']
    assert tree(phase/'checkpoints'/task/'1')==receipt['checkpoint_sha256']
    assert sha(phase/'requests'/(task+'_attempt1.json'))==receipt['audit_sha256']
    original=sqlite3.connect((backup/'run.sqlite3').as_uri()+'?mode=ro',uri=True)
    old=original.execute("SELECT * FROM tasks WHERE status='completed' ORDER BY rowid").fetchall();original.close()
    current=sqlite3.connect((phase/'run.sqlite3').as_uri()+'?mode=ro',uri=True)
    assert all(current.execute('SELECT * FROM tasks WHERE id=?',(r[0],)).fetchone()==r for r in old), 'Completed prefix changed'
    attempt,status,result=current.execute('SELECT attempts,status,result FROM tasks WHERE id=?',(task,)).fetchone();current.close()
    if status=='interrupted_infrastructure':
        assert attempt==1 and result is None
        assert sha(cell/'budget.json')==receipt['budget_sha256']
        assert all(tree(Path(p))==expected for p,expected in receipt['bank_files_sha256'].items())
    else:
        assert attempt>=2 and status in {'running','completed'}, 'Unexpected recovery state'
    assert not any(cell.rglob('STOP_AFTER_TASK')), 'Explicit task-boundary stop is present'
for process in Path('/proc').glob('[0-9]*/cmdline'):
    try:args=process.read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    assert b'atomic_skillgraph.experiments.run_single_cell' not in args, 'An experiment is already active'
probe=subprocess.run(['/mnt/c/Windows/System32/wsl.exe','-d','docker-desktop','--','sh','-c',
    'wc -l < /proc/1/mountinfo; cat /proc/sys/fs/mount-max'],capture_output=True,check=True,timeout=20)
values=[int(v) for v in probe.stdout.replace(b'\0',b'').decode().splitlines() if v.strip().isdigit()]
assert len(values)==2 and values[0]<values[1]*0.8, 'Docker mount namespace lacks headroom; restart Desktop first'
subprocess.run(['docker','run','--rm','--network','none','--read-only','--entrypoint','/bin/true',
    'sha256:4db520e20cd121f830731c2d0c0bcfbe9cacd254d1817262404cc18e1e757783'],check=True,timeout=20)
print(json.dumps({'preflight':'passed','source':source['git_sha'],'docker_mount_count':values[0],
                  'docker_mount_limit':values[1],'paid_requests':0}),flush=True)
PY
benchmarks=(livemath spreadsheetbench)
commands() {
  command=("$PYTHON" -m atomic_skillgraph.experiments.run_single_cell
    --config configs/atomic_unified_seed42.yaml --benchmark "$1" --seed 42
    --model-key deepseek-v4-flash --datasets "$DATASETS" --output "$OUTPUT_ROOT/$1/seed42"
    --env-file "$ENV_FILE" --resume)
}
for benchmark in "${benchmarks[@]}"; do
  commands "$benchmark";printf '%q ' "${command[@]}";printf '\n'
done
[[ $mode != --print ]] || exit 0
exec 8>"$OUTPUT_ROOT/batch.lock"
flock -n 8 || { echo 'Batch is already running' >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks" "$OUTPUT_ROOT/tmp"
pids=()
for benchmark in "${benchmarks[@]}"; do
  commands "$benchmark";mkdir -p "$OUTPUT_ROOT/tmp/$benchmark"
  (
    exec 9>"$OUTPUT_ROOT/locks/$benchmark.lock"
    flock -n 9 || exit 2
    export TMPDIR="$OUTPUT_ROOT/tmp/$benchmark"
    exec "${command[@]}"
  ) >>"$OUTPUT_ROOT/logs/$benchmark.seed42.log" 2>&1 &
  pids+=("$!");printf '%s\n' "$!" > "$OUTPUT_ROOT/logs/$benchmark.seed42.pid"
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
