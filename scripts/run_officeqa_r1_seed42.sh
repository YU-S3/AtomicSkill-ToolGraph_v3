#!/usr/bin/env bash
# Fresh OfficeQA only. The six-task engineering gate is part of the formal Train.
set -euo pipefail
mode=${1:---print};revision=${2:-}
case "$mode" in --print|--execute|--resume) ;; *) echo 'Use --print, --execute or --resume, followed by the pinned commit' >&2;exit 2;;esac
CODE=/home/yangchengyu/asg_officeqa_r1_seed42_20261011
OUTPUT=/home/yangchengyu/officeqa_r1_fresh_seed42_formal_20261011/officeqa/seed42
OLD=/home/yangchengyu/atomic_unified_r1_selector_seed42_formal_20261010/officeqa/seed42
PYTHON=/home/yangchengyu/asg_alfworld_venv/bin/python
cd "$CODE";export PYTHONPATH="$CODE/src"
[[ $(git rev-parse HEAD) == "$revision" ]] && [[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'Pinned clean commit required' >&2;exit 2; }
command=("$PYTHON" -m atomic_skillgraph.experiments.run_single_cell
 --config configs/atomic_unified_seed42.yaml --benchmark officeqa --seed 42 --model-key deepseek-v4-flash
 --datasets /home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1 --output "$OUTPUT"
 --corpus-root /mnt/d/T3S_exp/SkillCompiler_resources_20261003/raw/officeqa/treasury_bulletins_parsed/transformed
 --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env)
if [[ $mode == --print ]];then printf '%q ' "${command[@]}";printf '\n';exit 0;fi
mkdir -p "$(dirname "$OUTPUT")"
exec 9>"$(dirname "$OUTPUT")/officeqa.lock";flock -n 9 || { echo 'OfficeQA is already running' >&2;exit 2; }
"$PYTHON" - "$OUTPUT" "$OLD" "$mode" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
from atomic_skillgraph.experiments.run_empirical import write_json
output,old=map(Path,sys.argv[1:3]);mode=sys.argv[3]
image=json.loads(Path('benchmark_profiles.json').read_text())['profiles']['officeqa']['image_digest']
subprocess.run(['docker','run','--rm','--network','none','--read-only','--entrypoint','/bin/true',image],check=True,timeout=30)
if mode=='--execute':
    assert not output.exists(), 'Fresh output already exists; use explicit same-version resume'
    original=old/'budget.json';raw=original.read_bytes();ledger=json.loads(raw)
    limits=json.loads(Path('configs/atomic_unified_seed42_budget.json').read_text())['cells']['officeqa']
    assert ledger['limits']=={k:limits[k] for k in ('token_limit','finish_reserve','request_limit')}
    assert not ledger['unknown_billing'] and all(a['status']=='metered' for a in ledger['attempts'].values())
    assert sum(a['accounted_tokens'] for a in ledger['attempts'].values())<limits['token_limit']
    output.mkdir(parents=True);(output/'budget.json').write_bytes(raw)
    write_json(output/'cost_carryforward_manifest.json',{'source_run':str(old),'source_budget_sha256':hashlib.sha256(raw).hexdigest(),
        'inherited_tokens':sum(a['accounted_tokens'] for a in ledger['attempts'].values()),'inherited_http':len(ledger['attempts']),
        'inherited_attempt_ids':list(ledger['attempts']),'limits_unchanged':True,'bank_inherited':False})
else:
    assert (output/'run_manifest.json').exists() and (output/'cost_carryforward_manifest.json').exists()
PY
export TMPDIR="$(dirname "$OUTPUT")/tmp";mkdir -p "$TMPDIR"
if [[ ! -f "$OUTPUT/prefix_engineering_audit.json" ]];then
  if [[ $mode == --execute ]];then
    "${command[@]}" --max-new-tasks 6 --stop-after-val
  else
    # A stopped prefix may need fewer than six new tasks to reach the same fixed gate.
    remaining=$("$PYTHON" - "$OUTPUT/train/run.sqlite3" <<'PY'
import sqlite3,sys
from pathlib import Path
p=Path(sys.argv[1]);db=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True)
print(6-db.execute("SELECT COUNT(*) FROM tasks WHERE status='completed'").fetchone()[0]);db.close()
PY
)
    if ((remaining > 0));then "${command[@]}" --resume --max-new-tasks "$remaining" --stop-after-val;fi
  fi
  "$PYTHON" -m atomic_skillgraph.experiments.audit_officeqa_prefix "$OUTPUT"
fi
exec "${command[@]}" --resume
