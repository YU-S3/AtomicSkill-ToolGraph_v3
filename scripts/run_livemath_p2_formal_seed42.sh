#!/usr/bin/env bash
# Fresh Train60 -> one frozen-contract P2 stage -> Frozen -> Val17 -> Test100.
set -euo pipefail
mode=${1:---check};revision=${2:-}
case "$mode" in --check|--execute|--resume) ;; *) echo 'Use --check, --execute or --resume with the pinned commit' >&2;exit 2;;esac
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${LIVEMATH_P2_FORMAL_OUTPUT:-/home/yangchengyu/livemath_p2_formal_seed42_20261012/livemath/seed42}
P1=${LIVEMATH_P2_COST_SOURCE:-/home/yangchengyu/livemath_public_process_seed42_formal_20261012/livemath/seed42}
P2=${LIVEMATH_P2_BATCH_COST_SOURCE:-/home/yangchengyu/livemath_p2_r8_no_change_format_seed42_20261012}
DATASETS=/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1
PYTHON=/home/yangchengyu/asg_alfworld_venv/bin/python
cd "$CODE";export PYTHONPATH="$CODE/src"
[[ $(git rev-parse HEAD) == "$revision" ]] && [[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'Pinned clean commit required' >&2;exit 2; }
[[ -f /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env ]] || { echo 'Existing model environment file missing' >&2;exit 2; }
mkdir -p "$(dirname "$OUTPUT")";exec 9>"$(dirname "$OUTPUT")/livemath-p2.lock";flock -n 9 || { echo 'Formal P2 is already running' >&2;exit 2; }
"$PYTHON" - "$OUTPUT" "$P1" "$P2" "$DATASETS" "$mode" <<'PY'
import json,subprocess,sys
from pathlib import Path
import yaml
from atomic_skillgraph.experiments.formal_choice_batch import configure,carryforward
from atomic_skillgraph.experiments.run_formal import configured_models,model_settings,resolved_config
from atomic_skillgraph.experiments.canonical_manifest import verify,sha256
from atomic_skillgraph.experiments.run_choice_capability_batch import partition
from atomic_skillgraph.experiments.run_empirical import write_json
from atomic_skillgraph.empirical.contracts import digest
from skillcompiler_bench_contracts.livemath import NORMALIZATION_VERSION,CHOICE_PROJECTION_VERSION,CHOICE_SEED
output,p1,p2,data=map(Path,sys.argv[1:5]);mode=sys.argv[5]
spec=yaml.safe_load(Path('configs/livemath_p2_formal_seed42.yaml').read_text());authority=Path(spec['authority'])
verify(authority)
models=[m for m in configured_models(json.loads(Path(spec['model_lock']).read_text())) if m['model_id']=='deepseek-v4-flash']
assert len(models)==1,'Exactly one configured DeepSeek model required'
base=model_settings(yaml.safe_load(Path(spec['base_config']).read_text()),models[0])
base['budget']=json.loads(Path(spec['budget_allocation']).read_text())['cells']['livemath']
base=configure(base,spec,'livemath',42,authority)
profile=json.loads(Path(spec['benchmark_profiles']).read_text())['profiles']['livemath']
configs={split:resolved_config(base,profile,'livemath',42,split,output,data,authority,None) for split in ('train','val','test')}
material=json.loads((data/'materialization.json').read_text())
assert material['authority_sha256']==sha256(authority/'manifest.json')
assert material['livemath_normalization_version']==NORMALIZATION_VERSION
assert material['livemath_choice_projection_version']==CHOICE_PROJECTION_VERSION and material['livemath_choice_seed']==CHOICE_SEED
assert sha256(data/'livemath_integrity.json')==material['integrity_sha256']
for name,value in material['files_sha256'].items():assert sha256(data/name)==value,name
for name,value in material.get('external_files_sha256',{}).items():assert sha256(name)==value,name
for split,count in [('train',60),('val',17),('test',100)]:
    rows=json.loads((data/'livemath'/(split+'.json')).read_text())['tasks']
    assert len(rows)==count and [r['task_id'] for r in rows]==[r['task_id'] for r in json.loads((authority/'livemath'/(split+'.json')).read_text())['tasks']]
    if split=='train':partition(rows,base['experiment']['choice_batch_contract'])
image=json.loads(Path('benchmark_profiles.json').read_text())['profiles']['livemath']['image_digest']
subprocess.run(['docker','run','--rm','--network','none','--read-only','--entrypoint','/bin/true',image],check=True,timeout=30)
ledger,lineage=carryforward([p1,p2],base['budget'])
if mode=='--execute':
    assert not output.exists(),'Fresh formal P2 requires a new empty output directory'
    write_json(output/'budget.json',ledger);write_json(output/'cost_carryforward_manifest.json',lineage)
elif mode=='--resume':
    assert (output/'run_manifest.json').exists(),'No formal identity to resume'
    saved=json.loads((output/'cost_carryforward_manifest.json').read_text());assert saved==lineage,'Historical cost identity changed'
    current=json.loads((output/'budget.json').read_text())
    assert current['limits']==ledger['limits'] and not current['unknown_billing']
    assert all(current['attempts'].get(k)==v for k,v in ledger['attempts'].items())
elif output.exists():raise ValueError('Check expects a fresh output; use --resume for an existing run')
print(json.dumps({'status':'passed','new_model_http':0,'schedule':'Train60 -> one P2 -> Frozen -> Val17 -> Test100',
    'inherited_tokens':lineage['inherited_tokens'],'inherited_http':lineage['inherited_http'],
    'p2_limits':{'tokens':1800000,'http':80},'cumulative_limits':ledger['limits'],'bank_inherited':False,
    'resolved_config_hash':digest(configs),'choice_batch_version':base['experiment']['choice_batch_version'],
    'choice_contract_sha256':base['experiment']['choice_batch_contract_sha256']}))
PY
[[ $mode != --check ]] || exit 0
export TMPDIR="$(dirname "$OUTPUT")/tmp";mkdir -p "$TMPDIR"
command=("$PYTHON" -m atomic_skillgraph.experiments.run_single_cell
 --config configs/livemath_p2_formal_seed42.yaml --benchmark livemath --seed 42 --model-key deepseek-v4-flash
 --datasets "$DATASETS" --output "$OUTPUT" --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env)
[[ $mode != --resume ]] || command+=(--resume)
exec "${command[@]}"
