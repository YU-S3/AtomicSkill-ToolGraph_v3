#!/usr/bin/env bash
# Fresh OfficeQA only. The six-task engineering gate is part of the formal Train.
set -euo pipefail
mode=${1:---print};revision=${2:-}
case "$mode" in --print|--execute|--resume) ;; *) echo 'Use --print, --execute or --resume, followed by the pinned commit' >&2;exit 2;;esac
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${OFFICE_R1_OUTPUT:-/home/yangchengyu/officeqa_runtime_r2_fresh_seed42_formal_20261012/officeqa/seed42}
OLD=${OFFICE_R1_COST_SOURCE:-/home/yangchengyu/officeqa_finish_window_r1_fresh_seed42_formal_20261012/officeqa/seed42}
CHECK=${OFFICE_R1_VALIDATION_COST_SOURCE:-}
CONFIG=${OFFICE_R1_CONFIG:-configs/atomic_unified_seed42.yaml}
CORPUS=${OFFICE_R1_CORPUS:-/mnt/d/T3S_exp/SkillCompiler_resources_20261003/raw/officeqa/treasury_bulletins_parsed/transformed}
REVISION=${OFFICE_R1_BUDGET_REVISION:-}
PYTHON=/home/yangchengyu/asg_alfworld_venv/bin/python
cd "$CODE";export PYTHONPATH="$CODE/src"
[[ $(git rev-parse HEAD) == "$revision" ]] && [[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'Pinned clean commit required' >&2;exit 2; }
command=("$PYTHON" -m atomic_skillgraph.experiments.run_single_cell
 --config "$CONFIG" --benchmark officeqa --seed 42 --model-key deepseek-v4-flash
 --datasets /home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1 --output "$OUTPUT"
 --corpus-root "$CORPUS"
 --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env)
if [[ $mode == --print ]];then printf '%q ' "${command[@]}";printf '\n';exit 0;fi
mkdir -p "$(dirname "$OUTPUT")"
exec 9>"$(dirname "$OUTPUT")/officeqa.lock";flock -n 9 || { echo 'OfficeQA is already running' >&2;exit 2; }
"$PYTHON" - "$OUTPUT" "$OLD" "$mode" "$CHECK" "$CONFIG" "$REVISION" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
import yaml
from atomic_skillgraph.experiments.run_empirical import write_json
from atomic_skillgraph.experiments.seed42_budget import revised_limits
output,old=map(Path,sys.argv[1:3]);mode=sys.argv[3]
image=json.loads(Path('benchmark_profiles.json').read_text())['profiles']['officeqa']['image_digest']
subprocess.run(['docker','run','--rm','--network','none','--read-only','--entrypoint','/bin/true',image],check=True,timeout=30)
if mode=='--execute':
    assert not output.exists(), 'Fresh output already exists; use explicit same-version resume'
    assert output.resolve()!=old.resolve(), 'Cost source must remain separate from the fresh run'
    original=old/'budget.json';raw=original.read_bytes();ledger=json.loads(raw)
    original_limits=json.loads(Path('configs/atomic_unified_seed42_budget.json').read_text())['cells']['officeqa']
    assert ledger['limits']=={k:original_limits[k] for k in ('token_limit','finish_reserve','request_limit')}
    spec=yaml.safe_load(Path(sys.argv[5]).read_text())
    budget=(json.loads(Path(spec['budget_allocation']).read_text())['cells']['officeqa'] if spec.get('budget_allocation')
            else yaml.safe_load(Path(spec['base_config']).read_text())['budget'])
    revision=json.loads(Path(sys.argv[6]).read_text()) if sys.argv[6] else None
    limits=revised_limits(ledger['limits'],{k:budget[k] for k in ledger['limits']},revision)
    limits_unchanged=ledger['limits']==limits
    assert not ledger['unknown_billing'] and all(a['status']=='metered' for a in ledger['attempts'].values())
    assert sum(a['accounted_tokens'] for a in ledger['attempts'].values())<limits['token_limit']
    assert len(ledger['attempts'])<limits['request_limit']
    additional=None
    if sys.argv[4]:
        check=Path(sys.argv[4]);check_raw=(check/'budget.json').read_bytes();check_ledger=json.loads(check_raw)
        assert not check_ledger['unknown_billing'] and all(a['status']=='metered' for a in check_ledger['attempts'].values())
        for key,attempt in check_ledger['attempts'].items():
            assert key not in ledger['attempts'] or ledger['attempts'][key]==attempt, 'Conflicting HTTP identity'
            ledger['attempts'][key]=attempt
        additional={'source_run':str(check),'source_budget_sha256':hashlib.sha256(check_raw).hexdigest(),
                    'attempt_ids':list(check_ledger['attempts'])}
    assert sum(a['accounted_tokens'] for a in ledger['attempts'].values())<limits['token_limit']
    assert len(ledger['attempts'])<limits['request_limit']
    ledger['limits']=limits
    output.mkdir(parents=True);write_json(output/'budget.json',ledger)
    write_json(output/'cost_carryforward_manifest.json',{'source_run':str(old),'source_budget_sha256':hashlib.sha256(raw).hexdigest(),
        'inherited_tokens':sum(a['accounted_tokens'] for a in ledger['attempts'].values()),'inherited_http':len(ledger['attempts']),
        'inherited_attempt_ids':list(ledger['attempts']),'limits_unchanged':limits_unchanged,'bank_inherited':False,
        'budget_revision':revision,
        'validation_cost_source':additional,
        'prior_cost_carryforward':json.loads((old/'cost_carryforward_manifest.json').read_text()) if (old/'cost_carryforward_manifest.json').exists() else None})
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
"$PYTHON" -m atomic_skillgraph.experiments.audit_officeqa_prefix "$OUTPUT" --verify
exec "${command[@]}" --resume
