"""Explicit pre-Gate recovery of P7 arm accounting; never sends model requests."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import run_math_effect_p7 as run

REVISION = 'p7.arm-accounting.v2'


def inspect_parent(parent, checkout):
    code = 'import json,sys; from atomic_skillgraph.experiments.run_math_effect_p7 import checked; print(json.dumps(checked(sys.argv[1])))'
    result = subprocess.run([sys.executable, '-c', code, str(parent)], cwd=checkout,
        env={**os.environ, 'PYTHONPATH':str(Path(checkout).resolve()/'src')},
        check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def recover(output, parent, parent_checkout, human_instruction):
    root=Path(output).resolve();parent=Path(parent).resolve();code=run.code_identity()
    if root.exists() or root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError('Recovery requires a separate, nonexistent directory')
    if code['tracked_dirty'] or not code['git_sha'] or not human_instruction.strip():
        raise ValueError('Recovery requires committed code and explicit human instruction')
    with ExitStack() as stack:
        for name in ('controller.lock','execution.lock'):
            lock=stack.enter_context((parent/name).open('r'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=inspect_parent(parent,parent_checkout)
        if code==old['code']:raise ValueError('Register the corrected source version first')
        stop=run.read(parent/'review_required.json')
        if stop!={'status':'budget_stopped','code':'p7_budget_exhausted','stage':'edit','automatic_retry':False}:
            raise ValueError('This recovery only handles the pre-Gate P7 accounting stop')
        if any((parent/n).exists() for n in ('gate','frozen_bank','val','test','feedback/result.json')):
            raise ValueError('Cannot recover completed selection or cross a sealed Gate')
        run.validate_settings(old['settings'])
        authority=run.read(parent/'paid_authorization.json')
        phases={k:old['settings']['phase_limits'][k] for k in ('learning','feedback','gate','val')}
        if authority.get('plan_hash')!=run.digest(old) or authority.get('phases')!=phases or not authority.get('human_instruction'):
            raise ValueError('Recovery cannot change the paid phase authority')
        ledger=run.read(parent/'budget/ledger.json');prefix=run.read(parent/'budget/prefix.json')
        new={k:v for k,v in ledger['attempts'].items() if k not in prefix['attempts']}
        if run.billing_blocked(ledger) or any(v['status']!='metered' for v in new.values()):
            raise ValueError('Unresolved P7 billing cannot be replayed')
        bank=run.Bank(parent/'active_bank',readonly=True)
        try:
            if bank.digest()!=old['initial_bank_digest']:raise ValueError('Expected the original unpublished Bank')
        finally:bank.close()
        if not (parent/'learning/initial/learning.json').is_file():raise ValueError('Missing committed initial learning')
        excluded=[];recorded=set();traces=list((parent/'feedback').rglob('trace.json'))
        for path in traces:run.runtime.verify_trace(run.read(path))
        for stage in ('learning','feedback'):
            for path in (parent/stage).rglob('request.json'):
                folder=path.parent
                if not (folder/'response.json').exists():
                    if (folder/'started.json').exists() or (folder/'requests.json').exists():
                        raise ValueError('Interrupted physical request cannot be repeated')
                    excluded.append(path.relative_to(parent).as_posix());continue
                committed=folder.parent/('trace.json' if stage=='feedback' else 'learning.json')
                if not committed.exists():raise ValueError('Request has no committed task/learning result')
                response=run.read(folder/'response.json')
                if response['response_hash']!=run.digest(response['turn']):raise ValueError('Response hash changed')
                records=run.read(folder/'requests.json')
                if response['request_hash']!=run.digest(records):raise ValueError('Physical receipt changed')
                for row in records:
                    key=row['request_id'];entry=new.get(key)
                    if not entry or entry['actual_usage']!=row['raw_usage']:raise ValueError('Physical receipt differs from ledger')
                    if key in recorded:raise ValueError('Duplicate physical receipt')
                    recorded.add(key)
        if recorded!=set(new):raise ValueError('Incomplete P7 physical billing evidence')
        names=set(old['prepared_hashes'])|{'budget/ledger.json'}
        for folder in ('active_bank','learning','feedback','candidate_versions','memory'):
            names.update(p.relative_to(parent).as_posix() for p in (parent/folder).rglob('*') if p.is_file())
        bound=names|{'plan.json','plan_identity.json','paid_authorization.json','review_required.json'}
        hashes={str(parent/n):run.sha256(parent/n) for n in sorted(bound)}
        for name in sorted(names-set(excluded)):
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(parent/name,target)
        run.write_json(root/'budget/recovery_prefix.json',ledger)
        cfg=run.read(root/'config.json');cfg['data_dir']=str(root/'active_bank')
        cfg['experiment'].update(output_dir=str(root),implementation_revision=REVISION)
        run.write_json(root/'config.json',cfg)
        continuation={'schema':REVISION,'human_instruction':human_instruction,'parent_output':str(parent),
            'parent_plan_hash':run.digest(old),'parent_run_id':old['run_id'],'parent_code':old['code'],
            'new_code':code,'parent_files':hashes,'not_sent_requests_retained_only_in_parent':excluded,
            'inherited_p7_request_ids':sorted(new),'inherited_p7_tokens':sum(v['accounted_tokens'] for v in new.values()),
            'reused_completed_traces':len(traces),'phase_limits_unchanged':True,'resume_stage':'edit','test_authorized':False}
        run.write_json(root/'continuation_manifest.json',continuation)
        plan=deepcopy(old);plan.pop('run_id')
        plan.update(code=code,created_at=run.utc(),continuation=run.digest(continuation),implementation_revision=REVISION)
        plan['files'].update(hashes)
        plan['prepared_hashes']={n:run.sha256(root/n) for n in [*old['prepared_hashes'],'budget/recovery_prefix.json','continuation_manifest.json']}
        plan['run_id']='p7_'+run.digest(plan)[:24]
        run.write_json(root/'plan.json',plan);run.write_json(root/'plan_identity.json',{'sha256':run.digest(plan)})
        authority.update(plan_hash=run.digest(plan),human_instruction=human_instruction,
            inherited_authority_hash=run.digest(run.read(parent/'paid_authorization.json')),recorded_at=run.utc(),test_authorized=False)
        run.write_json(root/'paid_authorization.json',authority)
        run.checked(root)
        for phase in phases:run.authorize(root,phase,plan)
        return continuation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('output','parent','parent-checkout','human-instruction'):p.add_argument('--'+arg,required=True)
    args=p.parse_args()
    result=recover(args.output,args.parent,args.parent_checkout,args.human_instruction)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
