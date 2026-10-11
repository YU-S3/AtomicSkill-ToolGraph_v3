#!/usr/bin/env bash
# One fresh method observation: six formal Train tasks, same Bank through Train60/Val17.
set -euo pipefail
mode=${1:---print};revision=${2:-}
case "$mode" in --print|--execute|--resume) ;; *) echo 'Use --print, --execute or --resume with the pinned commit' >&2;exit 2;;esac
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${LIVEMATH_PROCESS_OUTPUT:-/home/yangchengyu/livemath_public_process_seed42_formal_20261012/livemath/seed42}
OLD=${LIVEMATH_PROCESS_COST_SOURCE:-/home/yangchengyu/atomic_unified_r1_selector_seed42_formal_20261010/livemath/seed42}
PYTHON=/home/yangchengyu/asg_alfworld_venv/bin/python
cd "$CODE";export PYTHONPATH="$CODE/src"
[[ $(git rev-parse HEAD) == "$revision" ]] && [[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'Pinned clean commit required' >&2;exit 2; }
command=("$PYTHON" -m atomic_skillgraph.experiments.run_single_cell
 --config configs/livemath_public_process_seed42.yaml --benchmark livemath --seed 42 --model-key deepseek-v4-flash
 --datasets /home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1 --output "$OUTPUT"
 --stop-after-val --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env)
if [[ $mode == --print ]];then printf '%q ' "${command[@]}";printf '\n';exit 0;fi
mkdir -p "$(dirname "$OUTPUT")";exec 9>"$(dirname "$OUTPUT")/livemath.lock";flock -n 9 || { echo 'LiveMath is already running' >&2;exit 2; }
"$PYTHON" - "$OUTPUT" "$OLD" "$mode" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
from atomic_skillgraph.experiments.run_empirical import write_json
output,old=map(Path,sys.argv[1:3]);mode=sys.argv[3]
image=json.loads(Path('benchmark_profiles.json').read_text())['profiles']['livemath']['image_digest']
subprocess.run(['docker','run','--rm','--network','none','--read-only','--entrypoint','/bin/true',image],check=True,timeout=30)
if mode=='--execute':
    assert not output.exists() and output.resolve()!=old.resolve(), 'Fresh method needs a separate empty output'
    raw=(old/'budget.json').read_bytes();ledger=json.loads(raw)
    limits=json.loads(Path('configs/atomic_unified_seed42_budget.json').read_text())['cells']['livemath']
    assert ledger['limits']=={k:limits[k] for k in ('token_limit','finish_reserve','request_limit')}
    assert not ledger['unknown_billing'] and all(a['status']=='metered' for a in ledger['attempts'].values())
    assert sum(a['accounted_tokens'] for a in ledger['attempts'].values())<limits['token_limit'] and len(ledger['attempts'])<limits['request_limit']
    output.mkdir(parents=True);(output/'budget.json').write_bytes(raw)
    write_json(output/'cost_carryforward_manifest.json',{'source_run':str(old),'source_budget_sha256':hashlib.sha256(raw).hexdigest(),
        'inherited_tokens':sum(a['accounted_tokens'] for a in ledger['attempts'].values()),'inherited_http':len(ledger['attempts']),
        'inherited_attempt_ids':list(ledger['attempts']),'limits_unchanged':True,'bank_inherited':False,
        'prior_cost_carryforward':json.loads((old/'cost_carryforward_manifest.json').read_text()) if (old/'cost_carryforward_manifest.json').exists() else None})
else:assert (output/'run_manifest.json').exists() and (output/'cost_carryforward_manifest.json').exists()
PY
export TMPDIR="$(dirname "$OUTPUT")/tmp";mkdir -p "$TMPDIR"
if [[ ! -f "$OUTPUT/prefix_engineering_audit.json" ]];then
  if [[ $mode == --execute ]];then "${command[@]}" --max-new-tasks 6
  else
    remaining=$("$PYTHON" - "$OUTPUT/train/run.sqlite3" <<'PY'
import sqlite3,sys
from pathlib import Path
db=sqlite3.connect(Path(sys.argv[1]).as_uri()+'?mode=ro',uri=True)
print(6-db.execute("SELECT COUNT(*) FROM tasks WHERE status='completed'").fetchone()[0]);db.close()
PY
)
    if ((remaining > 0));then "${command[@]}" --resume --max-new-tasks "$remaining";fi
  fi
  "$PYTHON" -m atomic_skillgraph.experiments.audit_choice_prefix "$OUTPUT"
fi
exec "${command[@]}" --resume
