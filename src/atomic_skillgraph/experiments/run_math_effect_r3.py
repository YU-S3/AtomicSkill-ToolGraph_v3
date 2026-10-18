"""Finite P7-R3 Train-trace learning, Docker qualification and paired development."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from ..agents.protocol import NativeToolSpec
from ..empirical import math_effect_program_r3 as p, math_effect_runtime_r3 as r
from ..empirical import math_effect_runtime as base, math_effect_source as source
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor,billing_blocked,input_token_bound
from ..empirical.contracts import PublicTask,digest,program_digest
from ..empirical.math_effect_learner import request_material
from ..harness.benchmarks import AnswerAdapter
from .canonical_manifest import sha256
from .formal_log import utc,tree_identity
from .run_empirical import code_identity,write_json,load_env
from .run_math_effect_p7 import merge_ledgers,configure,phase_governor,StageContext

read=source.read
RUNTIME={'completion':65536,'finish_completion':16384,'timeout':600,'parent_tokens':128000,'tail_bytes':16384}


def validate_settings(s):
    if (s['method_version']!=p.VERSION or s['seed']!=42 or s['runtime']!=RUNTIME or
        s['initial_candidates']!=3 or s['edit_rounds']!=2 or s['structural_repair_limit']!=1 or
        s['cumulative_limits']!={'token_limit':45000000,'request_limit':9165,'finish_reserve':0} or
        set(s['phase_limits'])!={'learning','feedback','val','test'} or
        sum(v['token_limit'] for v in s['phase_limits'].values())>7800000):raise ValueError('R3 finite contract changed')
    if sha256(Path(s['tokenizer_path']))!=s['tokenizer_sha256']:raise ValueError('Tokenizer changed')


def prepare(root,config,source_run,datasets):
    root=Path(root).resolve();s=yaml.safe_load(Path(config).read_text());validate_settings(s);code=code_identity()
    if root.exists() or code['tracked_dirty'] or not code['git_sha']:raise ValueError('Need a new directory and committed source')
    evidence=source.load(source_run,datasets);prefix,origins=merge_ledgers(s['ledger_sources'],s['cumulative_limits'])
    if billing_blocked(prefix):raise ValueError('Unresolved new billing')
    occupied=sum(x.get('accounted_tokens',x['reserved_tokens']) for x in prefix['attempts'].values())
    if occupied+sum(x['token_limit'] for x in s['phase_limits'].values())>45000000:raise ValueError('Allocation exceeds remaining cumulative tokens')
    if len(prefix['attempts'])+sum(x['request_limit'] for x in s['phase_limits'].values())>9165:raise ValueError('Allocation exceeds cumulative HTTP')
    cfg=configure(evidence['config'],s,root);cfg['program_worker']={'record_execution_receipt':True,'wall_timeout_seconds':15}
    keys=[t['physical_key'] for t in evidence['tasks']['train']];materials=source.proposal_material_r2(evidence)
    files=evidence['files']
    for path in [Path(config),Path(s['tokenizer_path']),*[Path(x) for x in s['ledger_sources']]]:files[str(path.resolve())]=sha256(path)
    baseline={}
    parent=Path(s['baseline_run'])
    for phase,split,selected in [('feedback','train',evidence['partition']['members']['edit_feedback']),
                                  ('val','val',[t['physical_key'] for t in evidence['tasks']['val']])]:
        baseline[phase]={};tasks={t['physical_key']:t for t in evidence['tasks'][split]}
        for key in selected:
            path=parent/phase/'off'/key/'trace.json';t=read(path);r.verify(t)
            if t['task']!=tasks[key] or t['runtime_contract_hash']!=r.contract(cfg) or t['selection']['selected']:
                raise ValueError('Historical NoSkill is not an exact same-input/same-Runtime baseline')
            for request_id,usage in zip(t['request_ids'],t['raw_usage']):
                charged=prefix['attempts'].get(request_id,{})
                if charged.get('status')!='metered' or charged.get('actual_usage')!=usage:
                    raise ValueError('Reused baseline physical usage is absent from the cumulative ledger')
            adapter=AnswerAdapter('livemath',read(Path(datasets)/'livemath'/('evaluator_records_'+split+'.json')))
            adapter.reset(PublicTask(**tasks[key]))
            try:
                if adapter.evaluate(adapter.submit(t['prediction']))!=t['score']:raise ValueError('Baseline native score differs')
            finally:adapter.close()
            for f in [path,*map(Path,t['request_files'])]:files[str(f.resolve())]=sha256(f)
            baseline[phase][key]={'path':str(path),'trace_hash':t['trace_hash'],'request_ids':t['request_ids'],'new_http':0}
    root.mkdir(parents=True)
    values={'config.json':cfg,'source_materials.json':materials,'source_catalog.json':evidence['catalog'],
        'tasks.json':evidence['tasks'],'partition.json':evidence['partition'],'baseline.json':baseline,
        'budget/prefix.json':prefix,'budget/ledger.json':prefix,'budget/origins.json':origins}
    for name,v in values.items():write_json(root/name,v)
    bank=Bank(root/'initial_bank',environment=cfg['program_environment']);bank.bind_choice_method(p.VERSION);initial=bank.digest();bank.close()
    plan={'method_version':p.VERSION,'code':code,'created_at':utc(),'settings':s,'datasets':str(Path(datasets).resolve()),
        'source_run':str(source_run),'source_run_id':evidence['manifest']['run_id'],
        'source_commit':evidence['manifest']['source_commit'],'learning_source_keys':keys,'initial_bank_digest':initial,
        'files':files,'prepared_hashes':{name:sha256(root/name) for name in values if name!='budget/ledger.json'},
        'historical_tokens_or_reserved':occupied,'historical_http':len(prefix['attempts']),
        'test_development_history_disclosed':True,'target_correct_test100':[70,80],
        'admission_policy':{'train_dispatches':4,'train_accuracy':.8,'train_historical_losses':0},
        'val_gate':{'min_correct':12,'min_net':2,'max_losses':1,'empty_increase':0},
        'iteration_bound':{'initial_rosters':1,'substantive_edits':2,'structural_repairs_per_request':1}}
    plan['run_id']='p7_r3_'+digest(plan)[:24]
    write_json(root/'plan.json',plan);write_json(root/'plan_identity.json',{'hash':digest(plan)})
    return check(root)


def checked(root):
    root=Path(root);plan=read(root/'plan.json');validate_settings(plan['settings'])
    if code_identity()!=plan['code'] or digest(plan)!=read(root/'plan_identity.json')['hash']:raise ValueError('Frozen source/plan changed')
    for path,h in plan['files'].items():
        if sha256(Path(path))!=h:raise ValueError('Source evidence changed: '+path)
    for path,h in plan['prepared_hashes'].items():
        if sha256(root/path)!=h:raise ValueError('Prepared material changed: '+path)
    prefix=read(root/'budget/prefix.json');ledger=read(root/'budget/ledger.json')
    if ledger['limits']!=prefix['limits'] or any(ledger['attempts'].get(k)!=v for k,v in prefix['attempts'].items()):raise ValueError('Historical billing changed')
    return plan


def authorize(root,plan,phase):
    value=read(Path(root)/'paid_authorization.json')
    if (value.get('plan_hash')!=digest(plan) or not value.get('human_instruction') or
        value.get('phases',{}).get(phase)!=plan['settings']['phase_limits'][phase]):raise ValueError('Missing matching finite paid authority')
    if billing_blocked(read(Path(root)/'budget/ledger.json')):raise ValueError('Unresolved billing; no new HTTP')


def material(root,parents=(),feedback=None,rule_feedback=None):
    rows=read(Path(root)/'source_materials.json')
    from ..empirical.math_effect_selection import stronger_meta
    groups={}
    for row in rows:
        task=PublicTask(row['case_id'],row['case_id'],row['public_task']['goal'],{'choices':row['public_task']['choices']})
        key='public_meta_present' if stronger_meta(task)['has_stronger_meta'] else 'public_meta_absent'
        g=groups.setdefault(key,{'tasks':0,'historically_correct':0})
        g['tasks']+=1;g['historically_correct']+=bool(row['historical_correct'])
    value={'source_materials':rows,'public_failure_groups':groups,'parents':list(parents),
        'evidence_boundary':'Only historical predictions and correctness, no gold labels or human Bank assets.'}
    if feedback:
        value['train_feedback']={'groups':feedback['groups'],'rows':[{'public_task':x['off']['task'],
            'off':{k:x['off'][k] for k in ('prediction','score','valid')},
            'on':{k:x['on'][k] for k in ('prediction','score','valid','answer_origin')},
            'intervened':x['intervened']} for x in feedback['rows']]}
    if rule_feedback:
        value['host_train_rule_feedback']=[{'candidate_id':q['candidate_id'],'dispatched':q['dispatched'],
            'correct':q['correct'],'losses':q['losses_vs_historical'],'invariance':q['label_order_ambiguity_checks'],
            'qualified_for_trial':q['qualified_for_trial'],'rows':q['rows']} for q in rule_feedback.values()]
    return value


def check(root):
    root=Path(root);plan=checked(root);cfg=read(root/'config.json');s=plan['settings'];caps=[]
    provider=base.provider(cfg['llm'],'learning',s)
    # Capacity includes all original Train public inputs and an upper-sized malformed response.
    baseline=[read(v['path']) for v in read(root/'baseline.json')['feedback'].values()]
    feedback={'groups':{'fixture':'capacity_only'},'rows':[{'off':v,'on':{**v,'answer_origin':'dynamic'},'intervened':False} for v in baseline]}
    editor=material(root,[{'capacity_fixture':'x'*6000}]*3,feedback)
    editor['host_train_rule_feedback']=[{'physical_key':str(i),'capacity_fixture':'x'*256} for i in range(180)]
    materials=[('induction',material(root)),('editor_capacity',editor),
        ('repair',{'original_material':editor,'invalid_proposal':'x'*32768})]
    for name,data in materials:
        wire=provider._build_payload(request_material(p.PROMPT,data),[NativeToolSpec('submit_effect','Submit learned Skills',p.BATCH)],
            {'type':'function','function':{'name':'submit_effect'}})
        if len(json.dumps(wire).encode())>s['learning_wire_bytes']:raise ValueError('Full learning material too large')
        with TemporaryDirectory() as temp:
            g=BudgetGovernor(Path(temp)/'ledger.json',**s['phase_limits']['learning'],finish_reserve=0)
            g.admit('offline_capacity',wire,{'parent_scope':'offline','parent_token_limit':s['phase_limits']['learning']['token_limit'],
                'episode_scope':'offline','episode_request_limit':2})
        caps.append({'stage':name,'input_bound':input_token_bound(wire),'completion':8192})
    tasks=read(root/'tasks.json')
    for split in ('train','val','test'):
        for row in tasks[split]:
            task=PublicTask(**row);main=base.provider(cfg['llm'],'runtime',s);finish=base.provider(cfg['llm'],'finish',s)
            wire=main._build_payload(base.messages(task,[{'guidance':'x'*2400}]),[])
            end=finish._build_payload(base.messages(task,[{'guidance':'x'*2400}],{'text':'x'*16384,'qualification':'unverified'}),[])
            if input_token_bound(wire)+65536+input_token_bound(end)+16384>128000:raise ValueError('Public input/finish cannot fit parent')
            with TemporaryDirectory() as temp:
                g=BudgetGovernor(Path(temp)/'ledger.json',token_limit=128000,request_limit=2,finish_reserve=0)
                context={'parent_scope':'offline','parent_token_limit':128000,'episode_scope':'offline','episode_request_limit':2}
                g.admit('offline_main',wire,context)
                g.admit('offline_finish',end,{**context,'purpose':'finish_only'})
    result={'status':'passed','plan_hash':digest(plan),'new_model_http':0,'source_tasks':60,'independent_test':False,
        'capacity':caps,'canonical_counts':{k:len(v) for k,v in tasks.items()},'baseline_reuse_tasks':29,
        'historical_tokens_or_reserved':plan['historical_tokens_or_reserved'],'phase_limits':s['phase_limits'],
        'worst_case_full_completion_guaranteed':False}
    write_json(root/'zero_http_check.json',result);return result


def progress(root,stage,**values):
    root=Path(root);prefix=read(root/'budget/prefix.json')['attempts'];ledger=read(root/'budget/ledger.json')
    attempts=[v for k,v in ledger['attempts'].items() if k not in prefix]
    value={'stage':stage,'updated_at':utc(),'new_tokens_or_reserved':sum(a.get('accounted_tokens',a['reserved_tokens']) for a in attempts),
        'new_http':len(attempts),'billing_blocked':billing_blocked(ledger),**values}
    write_json(root/'progress.json',value);print(json.dumps(value,ensure_ascii=False),flush=True);return value


def qualify(root,plan,candidates):
    cfg=read(root/'config.json');tasks=[PublicTask(**t) for t in read(root/'tasks.json')['train']]
    historical={x['case_id']:x['historical_correct'] for x in read(root/'source_materials.json')}
    adapter=AnswerAdapter('livemath',read(Path(plan['datasets'])/'livemath/evaluator_records_train.json'));out={}
    try:
        for c in candidates:
            if c['candidate']['public_choice_rule'] is None:continue
            folder=root/'qualification'/c['id']
            out[c['id']]=p.qualify(c,tasks,historical,adapter,cfg['program_environment'],folder)
    finally:adapter.close()
    return out


def evaluate(root,plan,phase,policy,qualifications):
    authorize(root,plan,phase);cfg=read(root/'config.json');split='train' if phase=='feedback' else 'val'
    tasks={t['physical_key']:PublicTask(**t) for t in read(root/'tasks.json')[split]}
    baselines=read(root/'baseline.json')[phase];adapter=AnswerAdapter('livemath',read(Path(plan['datasets'])/'livemath'/('evaluator_records_'+split+'.json')))
    governor=StageContext(phase_governor(root,phase,plan),phase);pid=digest(policy);off=[];on=[]
    try:
        for key,b in baselines.items():
            off.append(read(b['path']))
            on.append(r.solve(tasks[key],adapter,policy,qualifications,cfg,root/phase/'on'/pid/key,governor))
            progress(root,phase,policy_id=pid,completed=len(on),total=len(baselines),correct=sum(x['score']['hard'] for x in on),
                program_answers=sum(x['answer_origin']=='learned_program' for x in on))
    finally:adapter.close()
    receipt=r.pair_report(off,on,policy,split);write_json(root/phase/'receipts'/(pid+'.json'),receipt);return receipt


def rank(receipt):
    g=receipt['groups']['all'];return (-g['on_correct'],g['losses'],g['empty_on'],g['tokens_on'],receipt['candidate_ids'])


def execute(root):
    root=Path(root);plan=checked(root);cfg=read(root/'config.json');sources=plan['learning_source_keys']
    if (root/'result.json').exists():return read(root/'result.json')
    if (root/'freeze_identity.json').exists():
        value=read(root/'selection.json');value['frozen']=frozen(root)
        write_json(root/'result.json',value);return value
    best=None;policy=[];qualifications={};history=[];parents=[];rule_feedback={}
    for round_ in range(3):
        authorize(root,plan,'learning');progress(root,'learning',round=round_)
        g=StageContext(phase_governor(root,'learning',plan),'learning')
        learned=p.learn(root/'learning'/str(round_),cfg,g,material(root,parents,best,rule_feedback),sources,parents=parents)
        proposed=learned['candidates']
        if not proposed:
            if round_==0:break
            continue
        qs=qualify(root,plan,proposed)
        parents=proposed;rule_feedback=qs
        # Failed executable rules are retained in diagnostics, never silently used as text.
        proposed=[x for x in proposed if x['candidate']['public_choice_rule'] is None or qs[x['id']]['qualified_for_trial']]
        qs={k:v for k,v in qs.items() if v['qualified_for_trial']}
        if not proposed:continue
        receipt=evaluate(root,plan,'feedback',proposed,qs);history.append(receipt)
        if best is None or rank(receipt)<rank(best):best=receipt;policy=proposed;qualifications=qs
        write_json(root/'feedback/best.json',{'policy':policy,'qualifications':qualifications,'receipt':best})
    if best is None:
        result={'status':'completed_no_candidate','method_version':p.VERSION,'history':[],'policy':[]}
    else:
        val=evaluate(root,plan,'val',policy,qualifications)
        consumed={pid for x in val['rows'] for pid in x['on'].get('consumed_program_ids',[])}
        coverage=all(q['program']['id'] in consumed for q in qualifications.values())
        passed=val['qualified'] and coverage
        result={'status':'val_passed' if passed else 'completed_negative_val','method_version':p.VERSION,
            'policy':policy,'qualifications':qualifications,'train_receipt':best,'val_receipt':val,
            'all_executable_candidates_consumed_in_val':coverage,'history':[x['receipt_hash'] for x in history]}
        write_json(root/'selection.json',result)
        if passed:result['frozen']=freeze(root,plan,result)
    write_json(root/'result.json',result);progress(root,result['status']);return result


def freeze(root,plan,result):
    if not r.verify_utility(result['val_receipt']) or not result['val_receipt']['qualified']:raise ValueError('Val gate not passed')
    cfg=read(root/'config.json');bank=Bank(root/'deployment_bank',environment=cfg['program_environment']);bank.bind_choice_method(p.VERSION)
    with bank.db:bank.db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)',('effect_source_keys',json.dumps(plan['learning_source_keys'])))
    try:
        for c in result['policy']:
            skill={'id':'skill_'+c['id'],'method_version':p.VERSION,'effect_candidate':c,'goal':c['candidate']['goal'],
                'guidance':json.dumps(p.old.card(c['candidate']),ensure_ascii=False),'state':'active_usable',
                'execution_intent':'program_candidate' if c['candidate']['public_choice_rule'] else 'guidance_only',
                'val_receipt':result['val_receipt']}
            bank.put('skill',skill)
            q=result['qualifications'].get(c['id'])
            if not q or not any(q['program']['id'] in x['on'].get('consumed_program_ids',[]) for x in result['val_receipt']['rows']):continue
            program=bank.put('program',q['program'])
            validation={'policy':p.VERSION+'.qualification','passed':True,'candidate':c,'train':q,'val':result['val_receipt'],
                'program_digest':program_digest(program),'environment_hash':digest(program['environment'])}
            attempt={'id':'attempt_'+digest(validation),'program_id':program['id'],'task_key':'val17','origin':'readonly_val',
                'outcome':'positive','basis':'task_outcome','validation':validation}
            bank.record(attempt);bank.record_validation(program['id'],validation)
            bank.put('implementation',{'method_version':p.VERSION,'skill_id':skill['id'],'program_id':program['id']})
        if (root/'frozen_bank/freeze.json').exists():manifest=read(root/'frozen_bank/freeze.json')
        else:manifest=bank.freeze(root/'frozen_bank')
        frozen_bank=Bank(root/'frozen_bank',readonly=True,environment=cfg['program_environment'])
        try:
            if (manifest['source_digest']!=bank.digest() or manifest['digest']!=frozen_bank.digest() or
                {x['id'] for x in frozen_bank.all('program')}!={q['program']['id'] for q in result['qualifications'].values()}):
                raise ValueError('Frozen snapshot did not preserve all qualified learned Programs')
        finally:frozen_bank.close()
    finally:bank.close()
    value={'method_version':p.VERSION,'policy':result['policy'],'qualifications':result['qualifications'],
        'manifest':manifest,'frozen_tree':tree_identity(root/'frozen_bank'),'selection_hash':digest(read(root/'selection.json')),
        'plan_hash':digest(plan),'freeze_time':utc()}
    write_json(root/'freeze_identity.json',value);return value


def frozen(root):
    root=Path(root);v=read(root/'freeze_identity.json')
    if v['frozen_tree']!=tree_identity(root/'frozen_bank') or v['selection_hash']!=digest(read(root/'selection.json')):raise ValueError('Frozen assets/evidence changed')
    return v


def test(root):
    root=Path(root);plan=checked(root);authorize(root,plan,'test');v=frozen(root)
    if v['plan_hash']!=digest(plan):raise ValueError('Frozen experiment identity changed')
    cfg=read(root/'config.json');tasks=[PublicTask(**t) for t in read(root/'tasks.json')['test']]
    adapter=AnswerAdapter('livemath',read(Path(plan['datasets'])/'livemath/evaluator_records_test.json'))
    g=StageContext(phase_governor(root,'test',plan),'test');traces=[]
    try:
        for t in tasks:
            traces.append(r.solve(t,adapter,v['policy'],v['qualifications'],cfg,root/'test'/t.physical_key,g))
            frozen(root);progress(root,'test',completed=len(traces),total=100,correct=sum(x['score']['hard'] for x in traces))
    finally:adapter.close()
    result={'tasks':100,'correct':sum(x['score']['hard'] for x in traces),'tokens':sum(x['tokens'] for x in traces),
        'http':sum(x['http'] for x in traces),'program_answers':sum(x['answer_origin']=='learned_program' for x in traces),
        'target_reached':sum(x['score']['hard'] for x in traces)>=70,'frozen_unchanged':frozen(root)==v,
        'development_exposed_test':True,'trace_hashes':[x['trace_hash'] for x in traces]}
    write_json(root/'test/summary.json',result);progress(root,'completed_test',**result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','execute','test','monitor'])
    parser.add_argument('--output',required=True);parser.add_argument('--config');parser.add_argument('--source-run');parser.add_argument('--datasets');parser.add_argument('--env-file')
    args=parser.parse_args();root=Path(args.output)
    if args.env_file:load_env(args.env_file)
    if args.command=='prepare':result=prepare(root,args.config,args.source_run,args.datasets)
    elif args.command=='check':result=check(root)
    elif args.command=='monitor':result=read(root/'progress.json') if (root/'progress.json').exists() else read(root/'zero_http_check.json')
    else:
        import fcntl
        with (root/'controller.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:result=execute(root) if args.command=='execute' else test(root)
            except Exception as exc:
                progress(root,'stopped',error_type=type(exc).__name__,error=str(exc));raise
    if args.command in {'execute','test'}:result={k:v for k,v in result.items() if k not in {'policy','qualifications','train_receipt','val_receipt','history','frozen'}}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
