"""Explicit continuation after one incomplete response; retain original task/HTTP caps."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from filelock import FileLock
import run_office_humanstrong_v13_crossmodel as runner
from atomic_skillgraph.empirical.budget_governor import reserved_recovery
from atomic_skillgraph.empirical.checkpoint import native_event_restorable
from atomic_skillgraph.harness.benchmarks import OfficeAdapter

c = runner.common
BASE = Path('/home/yangchengyu/officeqa_humanstrong_v13_rules_crossmodel_test102_schema_r1_seed42_20261022')


def prepare(root, base=BASE, *, failure_kind='incomplete_stream', authorization=None, source_changes=None):
    if root.exists(): raise ValueError('Continuation needs a fresh directory')
    code = c.code_identity()
    assert code['git_sha'] and not code['tracked_dirty']
    original = c.read(base/'plan.json');ledger = c.read(base/'budget.json')
    assert c.sha(base/'plan.json') == c.read(base/'plan_identity.json')['sha256']
    assert c.read(base/'launch_status.json')['status'] == 'failed'
    assert original['models'] == list(runner.MODELS) and original['package_identity'] == c.package_identity()
    assert all(c.sha(p)==h for p,h in original['protected_files'].items())
    inherited = reserved_recovery(ledger)
    unknown = {k:v for k,v in ledger['attempts'].items() if v['status']!='metered' and k not in inherited}
    if len(unknown)!=1: raise ValueError('Review each unresolved HTTP separately')
    request_id, charge = next(iter(unknown.items()))
    assert charge['status']=='unknown_billing' and charge.get('actual_usage') is None
    owner = charge['context']['parent_scope']
    source = Path(owner.rsplit(':',1)[0])
    assert source in [Path(e['path']) for e in original['episodes']]
    assert not (source/'summary.json').exists()
    audit_paths = list((source/'requests').glob('*.json'))
    attempts = [a for p in audit_paths for q in c.read(p)['requests'] for a in q.get('http_attempts',[])]
    error = [a for a in attempts if a['request_id']==request_id]
    assert len(error)==1 and error[0].get('raw_usage') is None
    if failure_kind=='incomplete_stream':
        assert error[0]['error_code']=='provider_transport_error'
        stream = error[0].get('response_diagnostic',{}).get('stream',{})
        assert stream.get('events',0)>0 and stream.get('completed') is False and not stream.get('usage_received')
        assert stream.get('finish_reason') is None
    elif failure_kind=='gateway_timeout':
        assert authorization and error[0]['error_code']=='provider_timeout' and error[0]['http_status']==504
        diagnostic=error[0].get('response_diagnostic',{}).get('error',{})
        assert diagnostic.get('type')=='gateway_timeout' or diagnostic.get('code')=='gateway_timeout'
    else:raise ValueError('Unsupported reviewed transport failure')
    states = list((source/'checkpoints').glob('*/*/state.json'))
    assert len(states)==1
    saved = c.read(states[0]);assert not saved.get('program_started')
    cfg = c.read(source/'config.json');adapter = OfficeAdapter({},cfg)
    specs = {t['name']:t for t in adapter.available_tools()}
    events = c.read(states[0].parent/'native_events.json')
    assert adapter.capabilities.checkpoint_mode=='workspace_copy'
    assert all(native_event_restorable(e,specs) for e in events)
    assert saved.get('executor_state') and saved['executor_state']['pending_step'] is None
    used = sum(map(c.charge,ledger['attempts'].values()))
    assert ledger['limits']==original['ledger_limits'] and used<ledger['limits']['token_limit']<=runner.CEILING
    root.mkdir(parents=True)
    c.save_json(root/'budget_prefix.json',ledger)
    receipt=ledger.setdefault('inherited_reservation_recovery',
        {'schema':'billing-reservation-recovery.v1','authorization':'','attempts':{}})
    receipt['authorization'] += authorization or (' User requested repair/continuation on 2026-10-22. '
        'Retain this one incomplete-stream reservation at its full upper bound; all task and cumulative limits unchanged. '
        'Resume the checkpoint once; do not replay completed actions or tasks; no automatic unlimited retries.')
    receipt['attempts'][request_id]={'entry_hash':c.digest(charge),'reserved_tokens':charge['reserved_tokens']}
    assert not c.billing_blocked(ledger)
    c.save_json(root/'budget.json',ledger)
    c.save_json(root/'tasks.json',c.read(base/'tasks.json'))
    for name in ('base_config.json',*[m+'_config.json' for m in runner.MODELS]):
        value=c.read(base/name);value['experiment']['output_dir']=str(root)
        c.write_json(root/name,value)
    episodes=[];retained=[];protected=dict(original['protected_files'])
    for e in original['episodes']:
        old=Path(e['path'])
        assert c.sha(old/'config.json')==e['config_hash'] and c.tree_identity(old/'bank')==e['bank_identity']
        if (old/'summary.json').exists():
            assert c.read(old/'summary.json')['complete'] and c.read(old/'freeze_record.json')['unchanged']
            episodes.append({**e,'retained_from':str(base),'origin_code':original['code']});retained.append(e)
            for p in (old/'summary.json',old/'freeze_record.json'):protected[str(p)]=c.sha(p)
            continue
        target=root/old.relative_to(base);shutil.copytree(old/'bank',target/'bank')
        config=c.read(old/'config.json');config['data_dir']=str(target/'bank');config['experiment']['output_dir']=str(target)
        c.write_json(target/'config.json',config)
        new={**e,'path':str(target),'config_hash':c.sha(target/'config.json')}
        if old==source:
            for folder in ('requests','checkpoints'):shutil.copytree(old/folder,target/folder)
            for name in ('run.sqlite3','.retrieval_cursor_key'):
                if (old/name).exists():shutil.copyfile(old/name,target/name)
            checkpoint=target/states[0].relative_to(old);state=c.read(checkpoint)
            identity=deepcopy(state['method_identity'])
            state['recovery_identity']={'source':str(old),'source_code':original['code'],
                'source_method_identity':identity,'response_namespace':str(states[0].parent),
                'parent_budget_scope':owner,'unknown_request_id':request_id}
            identity['config_hash']=c.digest(config);identity['budget']['path']=str(root/'budget.json')
            state['method_identity']=identity;c.save_json(checkpoint,state)
            manifest=c.read(old/'execution_manifest.json')
            c.write_json(target/'execution_manifest.json',{**manifest,'config':config,'code':code})
            new.update(recovery_source=str(old),original_parent_scope=owner)
            for p in old.rglob('*'):
                if p.is_file():protected[str(p)]=c.sha(p)
        else:
            assert not (old/'execution_manifest.json').exists(), 'Another interrupted episode needs review'
        episodes.append(new)
    for p in [base/'budget.json',base/'plan.json',base/'launch_status.json',base/'last_error.json',
              root/'budget_prefix.json',root/'tasks.json',root/'base_config.json',Path(__file__).resolve(),
              *[root/(m+'_config.json') for m in runner.MODELS]]:protected[str(p)]=c.sha(p)
    recovery={'source_code':original['code'],'retained_completed_episodes':len(retained),
        'interrupted_episode':str(source),'retained_native_events':len(events),'unknown_request_id':request_id,
        'unknown_tokens_reserved':charge['reserved_tokens'],'parent_budget_scope':owner,
        'limits_unchanged':ledger['limits'],'public_inputs_bank_and_inference_unchanged':True,
        'source_changes':source_changes if source_changes is not None else [
            'Allow restoration of completed tool refusals with backend_invoked=false.',
            'Report known tokens separately when an interrupted request has unknown billing.']}
    plan={**original,'created_at':c.utc(),'code':code,'base_run':str(base),'episodes':episodes,
        'protected_files':protected,'inherited_used_or_reserved':used,'recovery':recovery,
        'new_token_cap':ledger['limits']['token_limit']-used,
        'new_http_cap':ledger['limits']['request_limit']-len(ledger['attempts']),
        'observation_prefix_path':str(base/'budget_prefix.json'),
        'locks':sorted(set([*original['locks'],str(base/'.run.lock')])),
        'authorization':receipt['authorization'],'controller_sha256':c.sha(Path(__file__).resolve())}
    c.save_json(root/'plan.json',plan);c.save_json(root/'plan_identity.json',{'sha256':c.sha(root/'plan.json')})
    c.save_json(root/'continuation_manifest.json',recovery)
    c.save_json(root/'cost_carryforward_manifest.json',{'source':str(base),'used_or_reserved':used,
        'unknown_request_id':request_id,'reserved_tokens':charge['reserved_tokens'],'limits':ledger['limits']})
    c.save_json(root/'launch_status.json',{'status':'prepared','created_at':c.utc()})
    c.verify(root)
    print({'root':str(root),**recovery})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('prepare','check','launch','run','status'))
    ap.add_argument('--root',required=True);ap.add_argument('--base',default=str(BASE))
    ap.add_argument('--env-file',default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args=ap.parse_args();root=Path(args.root).resolve();runner.fixed.bind_runner(True)
    if args.action=='prepare':prepare(root,Path(args.base).resolve())
    elif args.action=='check':c.verify(root);print('Frozen continuation identity and billing verified; zero HTTP.')
    elif args.action=='status':print(json.dumps(runner.status(root),ensure_ascii=False,indent=2))
    elif args.action=='run':sys.exit(runner.execute(root,args.env_file))
    else:
        c.verify(root)
        with FileLock(str(root/'.launch.lock'),timeout=0):
            assert c.read(root/'launch_status.json')['status']=='prepared'
            with (root/'run.log').open('x') as log:
                p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--root',str(root),
                    '--env-file',args.env_file],cwd=c.SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            c.save_json(root/'launch_status.json',{'status':'launching','pid':p.pid,'started_at':c.utc()})
            print({'pid':p.pid,'root':str(root)})
