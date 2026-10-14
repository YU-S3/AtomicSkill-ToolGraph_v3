"""P5: prepare/check, bounded natural generation, Docker trials, and audit. No Val/Test entry."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import yaml

from ..core.errors import BudgetExhausted
from ..empirical import math_program_induction as m
from ..empirical import math_program_probe as probe_module
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor, billing_blocked
from ..empirical.contracts import digest
from ..empirical.system import EmpiricalSystem, validate_config
from ..harness.benchmarks import AnswerAdapter
from .canonical_manifest import sha256
from .formal_choice_batch import BatchGovernor
from .formal_log import utc
from .run_empirical import code_identity, write_json, load_env
from .run_choice_capability_batch import learning_boundary


def read(p):return json.loads(Path(p).read_text())


def jsonl(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    temporary.replace(path)


def configure(original, root, settings):
    from ..empirical import POLICY_DEFAULTS
    c=deepcopy(original)
    for section,defaults in POLICY_DEFAULTS.items():c.setdefault(section,{}).update(defaults)
    e=c['experiment']
    for k in list(e):
        if k.startswith('choice_'):e.pop(k)
    e.update(math_program_method_version=m.VERSION,runtime_mode='online',output_dir=str(root/'learning'),seed=42)
    c['data_dir']=str(root/'bank');c['budget']=settings['cumulative_limits']
    c['program_worker']['record_execution_receipt']=True
    for section in ('learning','runtime'):
        c[section].pop('choice_guidance',None)
    for k in ('capability_material_version','capability_payload_max_bytes','source_program_payload_max_bytes'):
        c['learning'].pop(k,None)
    c['runtime'].pop('choice_bounded_finalization',None)
    c['llm']['max_retries']=0
    c['llm']['purpose_overrides']={name:{'protocol':{'thinking_type':'disabled'},'max_completion_tokens':cap,'max_retries':0}
        for name,cap in [('math_program_spec',4096),('math_program_build',8192)]}
    for stage in ('extractor','tool_builder'):
        c['llm'][stage]['max_total_tokens_per_task']=800000
        c['llm'][stage]['request_timeout_seconds']=600
    return validate_config(c)


def prepare(source, output, config):
    root=Path(output).resolve();settings=yaml.safe_load(Path(config).read_text());code=code_identity()
    if root.exists():raise ValueError('P5 requires a new directory; preserve all old runs')
    if not code['git_sha'] or code['tracked_dirty']:raise ValueError('Commit P5 before preparing the paid identity')
    expected=Path(__file__).parents[3]/'configs/livemath_p5_program_probe_seed42.yaml'
    if settings!=yaml.safe_load(expected.read_text()):raise ValueError('P5 policy differs from committed configuration')
    rows,atoms,inputs,original,parent=m.load_sources(source,settings['source_p4'])
    if original['llm']['model']!=settings['model']:raise ValueError('Source and Builder backbone must agree')
    opportunities,selected=m.catalog(rows,atoms,settings['seed'],settings['max_opportunities'])
    prefix=read(settings['ledger_source'])
    if billing_blocked(prefix):raise ValueError('Unresolved billing without inherited reservation authority')
    if prefix['limits']!=settings['cumulative_limits']:raise ValueError('Cumulative budget cannot be reset')
    charge=sum(a.get('accounted_tokens',a['reserved_tokens']) for a in prefix['attempts'].values())
    if charge+settings['learning_budget_proposed']['token_limit']>prefix['limits']['token_limit']:
        raise ValueError('Proposed learning exceeds remaining campaign authority')
    if len(prefix['attempts'])+settings['learning_budget_proposed']['request_limit']>prefix['limits']['request_limit']:
        raise ValueError('Proposed HTTP exceeds remaining campaign authority')
    inputs[str(Path(settings['ledger_source']).resolve())]=sha256(Path(settings['ledger_source']))
    frozen=Path(settings['source_p4'])/'frozen_bank'
    # Hash-only preservation guard; no P4 asset or utility content enters learning.
    for p in frozen.iterdir():
        if p.is_file():inputs[str(p.resolve())]=sha256(p)
    cfg=configure(original,root,settings)
    bank=Bank(root/'bank',environment=cfg['program_environment'])
    initial=bank.digest();bank.close()
    plan={'method_version':m.VERSION,'code':code,'settings':settings,'prepared_at':utc(),
        'source_train':str(Path(source).resolve()),'source_train_run_id':parent['run_id'],
        'source_hash':digest(inputs),'policy_sha':digest(settings),'inputs':inputs,
        'initial_bank_digest':initial,'inherited_tokens':charge,'inherited_http':len(prefix['attempts']),
        'initial_bank':'fresh empty; no P4 assets or selection imported',
        'selected_opportunity_ids':[o['opportunity_id'] for o in selected],
        'consumer_probe_enabled':False,'model_response_identity':'pending_real_requests',
        'atomic_catalog_hash':digest(atoms),'source_catalog_hash':digest(rows)}
    plan['run_id']='p5_'+digest([plan['source_hash'],m.VERSION,code,settings])[:24]
    for name,value in [('config.json',cfg),('source_catalog.json',rows),('atomic_catalog.json',atoms),
        ('selected_opportunities.json',selected),('campaign_budget_prefix.json',prefix),('budget.json',prefix),
        ('learning/campaign_budget_prefix.json',prefix)]:write_json(root/name,value)
    jsonl(root/'opportunity_catalog.jsonl',opportunities)
    plan['prepared_hashes']={n:sha256(root/n) for n in ('config.json','source_catalog.json','atomic_catalog.json',
        'selected_opportunities.json','opportunity_catalog.jsonl','campaign_budget_prefix.json','learning/campaign_budget_prefix.json')}
    write_json(root/'pilot_manifest.json',plan);write_json(root/'plan_identity.json',{'sha256':digest(plan)})
    for f in ('spec_proposals.jsonl','program_bundles.jsonl','worker_receipts.jsonl'):jsonl(root/f,[])
    return check(root)


def checked(root):
    root=Path(root);plan=read(root/'pilot_manifest.json')
    if digest(plan)!=read(root/'plan_identity.json')['sha256']:raise ValueError('P5 manifest changed')
    if plan['code']!=code_identity():raise ValueError('P5 source identity changed; create an explicit new version')
    for path,h in plan['inputs'].items():
        if sha256(Path(path))!=h:raise ValueError('Historical source/ledger changed: '+path)
    for name,h in plan['prepared_hashes'].items():
        if sha256(root/name)!=h:raise ValueError('Prepared input changed: '+name)
    prefix=read(root/'campaign_budget_prefix.json');ledger=read(root/'budget.json')
    if ledger['limits']!=prefix['limits'] or any(ledger['attempts'].get(k)!=v for k,v in prefix['attempts'].items()):
        raise ValueError('Historical charge identity changed')
    if ledger.get('inherited_reservation_recovery')!=prefix.get('inherited_reservation_recovery'):
        raise ValueError('Historical unresolved reservation changed')
    validate_config(read(root/'config.json'))
    return plan


def check(root):
    root=Path(root);plan=checked(root);cfg=read(root/'config.json')
    from ..agents.provider import OpenAICompatibleProvider, OpenAICompatibleConfig
    from ..agents.protocol import NativeToolSpec
    from ..empirical.model_view import learning_payload
    from ..empirical.budget_governor import input_token_bound
    rows=read(root/'source_catalog.json');atoms=read(root/'atomic_catalog.json');capacities=[]
    for op in read(root/'selected_opportunities.json'):
        settings={**cfg['llm'],**cfg['llm']['extractor'],**cfg['llm']['purpose_overrides']['math_program_spec']}
        provider=OpenAICompatibleProvider(OpenAICompatibleConfig(base_url=settings['base_url'],model=settings['model'],
            api_key_env=settings['api_key_env'],dialect=settings['dialect'],thinking_type='disabled',max_completion_tokens=4096,max_retries=0))
        material={**m.material(op,rows,atoms),'learning_material_version':m.MATERIAL}
        messages=[{'role':'system','content':m.PROMPT},{'role':'user','content':json.dumps(learning_payload(material),ensure_ascii=False,allow_nan=False)}]
        payload=provider._build_payload(messages,[NativeToolSpec('submit_math_program_spec','Submit the requested result',m.PROPOSAL)])
        size=input_token_bound(payload)
        if size>65536:raise ValueError('Unissued P5 source material exceeds frozen capacity')
        capacities.append({'opportunity_id':op['opportunity_id'],'wire_bytes':size,'reserved_tokens':size+4096})
    docker=docker_ready()
    result={'status':'passed' if docker['ready'] else 'blocked','plan_hash':digest(plan),'docker':docker,
        'new_model_http':0,'opportunities':len(capacities),'capacities':capacities,
        'billing_ready':not billing_blocked(read(root/'budget.json')),'paid_authorized':(root/'paid_authorization.json').exists(),
        'consumer_probe_enabled':False,'val_test_enabled':False}
    write_json(root/'zero_http_check.json',result);return result


def docker_ready():
    import sys
    if sys.platform!='linux':return {'ready':False,'reason':'P5 requires Linux Docker'}
    try:
        result=subprocess.run(['docker','image','inspect',m.IMAGE,'--format','{{.Id}}'],
            text=True,capture_output=True,timeout=20,check=True)
        return {'ready':result.stdout.strip()==m.IMAGE,'image_digest':result.stdout.strip()}
    except (OSError,subprocess.SubprocessError) as exc:return {'ready':False,'reason':str(exc)}


def system(root, approved=False, provider=None):
    root=Path(root);plan=checked(root)
    authority=root/'paid_authorization.json'
    if approved:
        receipt={'plan_hash':digest(plan),'scope':'learning_and_one_builder_repair_only',
            'limits':plan['settings']['learning_budget_proposed'],'consumer_probe':False}
        if authority.exists() and read(authority)!=receipt:raise ValueError('Paid authority changed')
        write_json(authority,receipt)
    if not authority.exists():raise ValueError('P5 learning budget is proposed, not authorized; review zero-HTTP results first')
    if read(authority)['plan_hash']!=digest(plan):raise ValueError('Authorization is for another plan')
    preflight=read(root/'zero_http_check.json')
    if preflight.get('status')!='passed' or preflight.get('plan_hash')!=digest(plan) or not docker_ready()['ready']:
        raise ValueError('P5 requires matching zero-HTTP preflight and locked Linux Docker before paid generation')
    owner=BudgetGovernor(root/'budget.json',**plan['settings']['cumulative_limits'])
    governor=BatchGovernor(owner,{'run_id':plan['run_id'],'new_limits':plan['settings']['learning_budget_proposed']},root/'learning')
    s=EmpiricalSystem(read(root/'config.json'),harness=AnswerAdapter('livemath',{}),budget_governor=governor,provider=provider)
    started=root/'learning/started.json'
    try:
        if not started.exists():
            if s.bank.digest()!=plan['initial_bank_digest'] or any(s.bank.all(k) for k in ('skill','program','implementation')):
                raise ValueError('P5 must begin with its declared empty Bank')
            write_json(started,{'plan_hash':digest(plan),'initial_bank_digest':s.bank.digest()})
        elif read(started)!={'plan_hash':digest(plan),'initial_bank_digest':plan['initial_bank_digest']}:
            raise ValueError('P5 initial Bank receipt changed')
        return s,plan
    except BaseException:
        s.close();raise


def receive(s,plan,op,folder,stage,prompt,material,name,schema,purpose,validator):
    learning_boundary(s,folder,plan['run_id']+':'+op['opportunity_id'],
        unit_limits={'token_limit':800000,'request_limit':3})
    identity=digest([plan['code'],plan['policy_sha'],s.config['llm'],prompt,material,schema])
    failure=folder/('rejected_'+identity+'.json')
    if failure.exists():
        old=read(failure);exc=ValueError(old['reason']);exc.model_authored=True
        exc.invalid_source=old.get('invalid_source');raise exc
    try:
        return s.learner._receive('p5_'+identity,stage,prompt,material,name,schema,
            material_version=m.MATERIAL,decision_purpose=purpose,validator=validator,repair_limit=0)
    except ValueError as exc:
        if model_failure(exc):write_json(failure,{'identity':identity,'reason':str(exc),'invalid_source':getattr(exc,'invalid_source',None)})
        raise


def model_failure(exc):return isinstance(exc,ValueError) and getattr(exc,'model_authored',False)


def build(s,plan,op,value,root,previous=None,feedback=None):
    directory=root/'learning'/op['opportunity_id'];asset=s.bank.put('skill',m.atomic(value,op))
    generation=1 if previous else 0
    material={'atomic':value['atomic'],'program_spec':value['program_spec'],'limitations':value['limitations'],
        'operation_semantics':probe_module.semantics(value['program_spec']['probe_kind']),
        'generation_basis':'hypothesis_from_train','source_opportunity_ids':value['source_opportunity_ids']}
    if previous:material.update(previous_source=previous.get('source'),previous_failure=feedback)
    def validate(answer):
        try:m.program(answer['source'],value['program_spec'],asset,s.config['program_environment'],previous.get('id') if previous else None)
        except ValueError as exc:exc.invalid_source=answer['source'];raise
    try:
        answer=receive(s,plan,op,directory,'tool_builder',m.BUILD_PROMPT,material,'submit_program',m.BUILD,
            'math_program_build',validate)
        program=s.bank.put('program',m.program(answer['source'],value['program_spec'],asset,s.config['program_environment'],previous.get('id') if previous else None))
        path=root/'programs'/(program['id']+'.py');path.parent.mkdir(exist_ok=True);path.write_text(program['source'])
        impl=s.bank.put('implementation',{'skill_id':asset['id'],'program_id':program['id'],'state':'candidate',
            'method_version':m.VERSION,'generation_basis':'hypothesis_from_train','formal_usable':False})
        return {'state':'code_generated','generation':generation,'atomic':asset,'program':program,'implementation':impl,
            'autonomously_generated':True,'consumed':False,'usable':False}
    except ValueError as exc:
        if not model_failure(exc):raise
        return {'state':'model_invalid','generation':generation,'reason':str(exc),'source':getattr(exc,'invalid_source',None),
            'atomic':asset,'autonomously_generated':False}


def learn(root, approved=False, provider=None):
    root=Path(root);s,plan=system(root,approved,provider);rows=read(root/'source_catalog.json');atoms=read(root/'atomic_catalog.json')
    try:
        for op in read(root/'selected_opportunities.json'):
            folder=root/'learning'/op['opportunity_id'];path=folder/'learning.json'
            if path.exists():continue
            try:
                value=receive(s,plan,op,folder,'extractor',m.PROMPT,m.material(op,rows,atoms),
                    'submit_math_program_spec',m.PROPOSAL,'math_program_spec',lambda v:m.validate_spec(v,op))
                value=m.validate_spec(value,op);row={'opportunity_id':op['opportunity_id'],'proposal':value,'state':value['decision']}
                if value['decision']=='propose_program':
                    bundle=build(s,plan,op,value,root)
                    if bundle['state']=='model_invalid':
                        previous={'source':bundle.get('source')};bundle2=build(s,plan,op,value,root,previous,bundle['reason'])
                        bundle2['previous_attempt']=bundle;bundle=bundle2
                    row['bundle']=bundle;row['state']=bundle['state']
            except ValueError as exc:
                if not model_failure(exc):raise
                row={'opportunity_id':op['opportunity_id'],'state':'model_invalid','reason':str(exc)}
            write_json(path,row)
    except BudgetExhausted as exc:write_json(root/'stop.json',{'status':'budget_stopped','reason':str(exc)})
    finally:s.close()
    return report(root)


def probe(root, worker=None, approved=False, provider=None):
    from ..empirical.program_worker import ProgramWorker
    root=Path(root);plan=checked(root);worker=worker or ProgramWorker(read(root/'config.json')['program_worker'])
    for op in read(root/'selected_opportunities.json'):
        folder=root/'learning'/op['opportunity_id'];path=folder/'learning.json'
        if not path.exists():continue
        row=read(path);bundle=row.get('bundle',{})
        if bundle.get('state')!='code_generated':continue
        result_path=folder/'probe.json'
        if result_path.exists():continue
        result=probe_module.probe(worker,bundle['program'],folder/'worker')
        # One shared Builder repair allowance covers declaration and real Worker failure.
        if result['state']=='probe_failed' and bundle['generation']==0 and approved:
            s,_=system(root,approved,provider)
            try:
                revised=build(s,plan,op,row['proposal'],root,bundle['program'],result)
                revised['previous_attempt']=bundle;revised['previous_probe']=result
                row['bundle']=revised;row['state']=revised['state'];write_json(path,row)
                if revised['state']=='code_generated':result=probe_module.probe(worker,revised['program'],folder/'worker')
            except BudgetExhausted as exc:
                write_json(root/'stop.json',{'status':'budget_stopped','reason':str(exc)})
                return report(root)
            finally:s.close()
        write_json(result_path,result)
    return report(root)


def report(root):
    root=Path(root);plan=checked(root);rows=[];bundles=[];receipts=[]
    for op in read(root/'selected_opportunities.json'):
        folder=root/'learning'/op['opportunity_id'];p=folder/'learning.json'
        row={'opportunity_id':op['opportunity_id'],'origin':op['origin'],'parent_outcome':op['parent_outcome'],
            'state':'pending','autonomously_generated':False,'probe_passed':False,'consumed':False,'usable':False}
        if p.exists():
            d=read(p);row['state']=d['state'];bundle=d.get('bundle')
            if bundle:bundles.append(bundle);row['autonomously_generated']=bundle.get('autonomously_generated',False)
            if (folder/'probe.json').exists():
                probe=read(folder/'probe.json');row.update(state=probe['state'],probe_passed=probe['probe_passed'] if 'probe_passed' in probe else False)
            receipts.extend(read(p) for p in (folder/'worker').glob('*.json') if not p.name.endswith('.started.json'))
        rows.append(row)
    ledger=read(root/'budget.json');prefix=read(root/'campaign_budget_prefix.json')
    attempts={k:v for k,v in ledger['attempts'].items() if k not in prefix['attempts']}
    value={'method_version':m.VERSION,'plan_hash':digest(plan),'source_commit':plan['code']['git_sha'],
        'status':'prepared' if all(r['state']=='pending' for r in rows) else 'awaiting_probe' if any(r['state']=='code_generated' for r in rows)
            else 'completed_no_program' if not any(r['probe_passed'] for r in rows) else 'completed_train_trial',
        'rows':rows,'generated_programs':len([r for r in rows if r['autonomously_generated']]),
        'probe_passed':sum(r['probe_passed'] for r in rows),'usable_programs':0,'consumer_probe_executed':False,
        'new_http':len(attempts),'new_tokens':sum(v.get('accounted_tokens',v['reserved_tokens']) for v in attempts.values()),
        'new_metered_tokens':sum(v.get('accounted_tokens',0) for v in attempts.values()),
        'new_unresolved_reserved_tokens':sum(v['reserved_tokens'] for v in attempts.values() if 'accounted_tokens' not in v),
        'billing_blocked':billing_blocked(ledger),'val_test_executed':False}
    if (root/'stop.json').exists():value['stop']=read(root/'stop.json');value['status']=value['stop']['status']
    write_json(root/'usage_ledger.json',ledger);write_json(root/'pilot_review.json',value)
    jsonl(root/'spec_proposals.jsonl',[read(p) for p in sorted((root/'learning').glob('*/learning.json'))])
    jsonl(root/'program_bundles.jsonl',bundles);jsonl(root/'worker_receipts.jsonl',receipts)
    (root/'pilot_review.md').write_text('# LiveMath P5\n\n'+json.dumps(value,ensure_ascii=False,indent=2)+
        '\n\nLocal probe success is train_trial_only, never formal usable or general theorem proof.\n')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('prepare')
    for name in ('source','output','config'):p.add_argument('--'+name,required=True)
    for stage in ('check','learn','probe','report'):
        p=sub.add_parser(stage);p.add_argument('--output',required=True)
        if stage=='check':p.add_argument('--zero-http',action='store_true',required=True)
        if stage in {'learn','probe'}:
            p.add_argument('--env-file');p.add_argument('--paid-learning-approved',action='store_true')
    a=parser.parse_args()
    if getattr(a,'env_file',None):load_env(a.env_file)
    if a.stage=='prepare':result=prepare(a.source,a.output,a.config)
    else:
        import fcntl
        with (Path(a.output)/'execution.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=check(a.output) if a.stage=='check' else report(a.output) if a.stage=='report' else (
                learn(a.output,a.paid_learning_approved) if a.stage=='learn' else probe(a.output,approved=a.paid_learning_approved))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
