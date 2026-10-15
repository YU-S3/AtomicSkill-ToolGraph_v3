"""Explicit P7 stages. No automatic Test, no history import, no unknown-bill replay."""
import argparse
from copy import deepcopy
import json
import math
import random
import statistics
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from ..agents.protocol import NativeToolSpec
from ..core.errors import BudgetExhausted
from ..empirical import math_effect_contract as c, math_effect_learner as learner
from ..empirical import math_effect_source as source, math_effect_optimizer as opt
from ..empirical import math_effect_runtime as runtime, math_effect_memory as memory
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor, billing_blocked, input_token_bound
from ..empirical.contracts import PublicTask, digest
from ..harness.benchmarks import AnswerAdapter
from .formal_choice_batch import BatchGovernor
from .formal_log import tree_identity, utc
from .canonical_manifest import sha256
from .run_empirical import write_json, code_identity, load_env

read=source.read
CONTRACT=Path(__file__).parents[3]/'configs/livemath_effect_p7_seed42.yaml'


def merge_ledgers(paths, limits):
    attempts={};recoveries={};origins={}
    for path in paths:
        state=read(path)
        for key,row in state['attempts'].items():
            if key in attempts and attempts[key]!=row:raise ValueError('Conflicting physical billing: '+key)
            attempts[key]=row;origins.setdefault(key,[]).append(str(path))
        rec=state.get('inherited_reservation_recovery',{})
        for key,value in rec.get('attempts',{}).items():
            if key in recoveries and recoveries[key]!=value:raise ValueError('Reservation identity differs')
            recoveries[key]=value
    result={'limits':limits,'attempts':attempts,'unknown_billing':any(a['status']!='metered' for a in attempts.values())}
    if recoveries:
        result['inherited_reservation_recovery']={'schema':'billing-reservation-recovery.v1',
            'authorization':'Inherited unchanged reservation authorities; see source ledger hashes', 'attempts':recoveries}
    return result,origins


def configure(original, settings, root):
    cfg=deepcopy(original)
    cfg['experiment']={**cfg['experiment'],'choice_method_version':c.VERSION,'runtime_mode':'frozen',
        'implementation_revision':c.VERSION,'output_dir':str(root),'seed':42,'benchmark':'livemath'}
    for k in list(cfg['experiment']):
        if k.startswith(('choice_batch','choice_selection','math_program','manual_reference')):cfg['experiment'].pop(k)
    cfg['effect_p7']=deepcopy(settings);cfg['data_dir']=str(root/'active_bank')
    cfg['budget']=settings['cumulative_limits'];cfg['runtime'].pop('choice_guidance',None)
    return cfg


def prepare(output, config, source_run, datasets):
    root=Path(output).resolve();settings=yaml.safe_load(Path(config).read_text());code=code_identity()
    if root.exists():raise ValueError('P7 needs an independent fresh output directory')
    if code['tracked_dirty'] or not code['git_sha']:raise ValueError('Commit the P7 code before preparing')
    validate_settings(settings)
    evidence=source.load(source_run,datasets)
    prefix,origins=merge_ledgers(settings['ledger_sources'],settings['cumulative_limits'])
    files=evidence['files'];files[str(Path(config).resolve())]=sha256(Path(config))
    files[str(Path(settings['tokenizer_path']).resolve())]=sha256(Path(settings['tokenizer_path']))
    for p in settings['ledger_sources']:files[str(Path(p).resolve())]=sha256(Path(p))
    root.mkdir(parents=True)
    cfg=configure(evidence['config'],settings,root)
    sources=evidence['partition']['members']['proposal_train']
    initial=Bank(root/'active_bank');initial.bind_choice_method(c.VERSION)
    with initial.db:
        initial.db.execute('INSERT INTO metadata VALUES(?,?)',('effect_source_keys',json.dumps(sources)))
    initial_digest=initial.digest();initial.close()
    material=source.proposal_material(evidence,settings['trace_window_bytes'])
    for name,value in [('config.json',cfg),('partition.json',evidence['partition']),('tasks.json',evidence['tasks']),
        ('proposal_material.json',material),('source_catalog.json',evidence['catalog']),
        ('budget/ledger.json',prefix),('budget/prefix.json',prefix),('budget/origins.json',origins)]:write_json(root/name,value)
    plan={'method_version':c.VERSION,'code':code,'created_at':utc(),'settings':settings,'source_run':str(source_run),
        'source_run_id':evidence['manifest']['run_id'],'source_commit':evidence['manifest']['source_commit'],
        'datasets':str(Path(datasets).resolve()),'files':files,'initial_bank_digest':initial_digest,
        'source_manifest_hash':digest(sources),'test_development_history_disclosed':True,
        'initial_bank':'fresh empty P7; no historical published assets','generation_seed_supported':cfg['llm'].get('generation_seed_supported',False),
        'prepared_hashes':{n:sha256(root/n) for n in ('config.json','partition.json','tasks.json',
            'proposal_material.json','source_catalog.json','budget/prefix.json','budget/origins.json')}}
    plan['run_id']='p7_'+digest(plan)[:24]
    write_json(root/'plan.json',plan);write_json(root/'plan_identity.json',{'sha256':digest(plan)})
    return check(root)


def validate_settings(s):
    if (s['method_version']!=c.VERSION or s['partition']!={'proposal_train':28,'edit_feedback':12,'protected_gate':20}
        or s['initial_candidates']!=3 or s['edit_rounds']!=2 or s['edits_per_round']!=2 or s['program_enabled']):
        raise ValueError('P7 method contract differs from confirmed text-first scope')
    if s['gate']!={'min_net':2,'max_losses':1,'no_new_empty':True}:raise ValueError('P7 qualification changed')
    if s['runtime']!={'completion':65536,'finish_completion':16384,'timeout':600,'parent_tokens':128000,'tail_bytes':16384}:
        raise ValueError('P7 Runtime contract changed')
    if sha256(Path(s['tokenizer_path']))!=s['tokenizer_sha256']:raise ValueError('Tokenizer identity differs')


def checked(root):
    root=Path(root);p=read(root/'plan.json')
    if p['method_version']!=c.VERSION or p['code']!=code_identity() or digest(p)!=read(root/'plan_identity.json')['sha256']:
        raise ValueError('P7 source or plan changed; register a new version')
    for path,h in p['files'].items():
        if sha256(Path(path))!=h:raise ValueError('Locked source changed: '+path)
    for path,h in p['prepared_hashes'].items():
        if sha256(root/path)!=h:raise ValueError('Prepared input changed: '+path)
    ledger=read(root/'budget/ledger.json');prefix=read(root/'budget/prefix.json')
    if ledger['limits']!=prefix['limits'] or any(ledger['attempts'].get(k)!=v for k,v in prefix['attempts'].items()):
        raise ValueError('Historical charge changed')
    validate_settings(p['settings']);return p


def phase_governor(root,phase,plan):
    folder=Path(root)/'budget'/phase
    if not (folder/'campaign_budget_prefix.json').exists():write_json(folder/'campaign_budget_prefix.json',read(Path(root)/'budget/prefix.json'))
    # Each stage scope is a real run folder prefix, so resume never adopts another stage's requests.
    owner=BudgetGovernor(Path(root)/'budget/ledger.json',**plan['settings']['cumulative_limits'])
    class PhaseGovernor(BatchGovernor):
        def owns(self,context):return context.get('p7_phase')==phase
    return PhaseGovernor(owner,{'new_limits':plan['settings']['phase_limits'][phase]},folder)


def authorize(root,phase,plan):
    path=Path(root)/('test_authorization.json' if phase=='test' else 'paid_authorization.json')
    if not path.exists():raise ValueError('P7 paid stage requires explicit user authorization: '+phase)
    value=read(path)
    if (value.get('plan_hash')!=digest(plan) or not value.get('human_instruction') or
        value.get('limits')!=plan['settings']['phase_limits'][phase] and value.get('phases',{}).get(phase)!=plan['settings']['phase_limits'][phase]):
        raise ValueError('Paid authority does not match frozen phase/plan')
    if phase=='test':
        identity=test_identity(root,plan)
        if value.get('review_identity')!=identity or not read(Path(root)/'review.json')['recommend_release']:
            raise ValueError('Test needs approval bound to the completed Val review and exact Frozen')
    if billing_blocked(read(Path(root)/'budget/ledger.json')):raise ValueError('Unresolved billing blocks P7')


def check(root):
    root=Path(root);plan=checked(root);cfg=read(root/'config.json');settings=plan['settings'];caps=[]
    p=runtime.provider(cfg['llm'],'learning',settings)
    members=read(root/'partition.json')['members'];tasks=read(root/'tasks.json')
    feedback_tasks=[t for t in tasks['train'] if t['physical_key'] in members['edit_feedback']]
    # Fixed worst-size Editor windows, preserving every complete public question/choice.
    editor={'parents':[],'paired_feedback':[{'case_id':t['physical_key'],'public_task':{'goal':t['goal'],'choices':t['inputs']['choices']},
        'off':{'observed_prediction':source.window('x'*8192,settings['trace_window_bytes']),'correct':False},
        'on':{'observed_prediction':source.window('x'*8192,settings['trace_window_bytes']),'correct':True}} for t in feedback_tasks]}
    materials=[('inducer',learner.PROMPT,read(root/'proposal_material.json'),c.BATCH),('editor_windows',learner.EDITOR,editor,c.EDIT)]
    for name,prompt,material,schema in materials:
        tools=[NativeToolSpec('submit_effect','Submit a compact Train-only policy proposal',schema)]
        wire=p._build_payload(learner.request_material(prompt,material),tools,{'type':'function','function':{'name':'submit_effect'}})
        size=input_token_bound(wire)
        if size>settings['learning_wire_bytes']:raise ValueError('Complete P7 learning wire exceeds capacity')
        caps.append({'stage':name,'wire_bytes':size,'reserve':size+p.config.max_completion_tokens})
    for split,rows in tasks.items():
        for row in rows:
            task=PublicTask(**row);main=runtime.provider(cfg['llm'],'runtime',settings);finish=runtime.provider(cfg['llm'],'finish',settings)
            wire=main._build_payload(runtime.messages(task),[])
            tail={'text':'x'*16384,'source_request_id':'x'*64,'response_sha256':'x'*64,'reasoning_sha256':'x'*64,
                'full_bytes':65536,'start_byte':49152,'end_byte':65536,'truncated':True,'qualification':'unverified'}
            end=finish._build_payload(runtime.messages(task,tail=tail),[])
            reserve=input_token_bound(wire)+65536+input_token_bound(end)+16384
            if reserve>128000:raise ValueError('Complete question plus bounded finish does not fit parent')
            caps.append({'stage':split,'physical_key':task.physical_key,'main_wire_bytes':input_token_bound(wire),
                         'finish_wire_bytes':input_token_bound(end),'parent_reserve':reserve})
    ledger=read(root/'budget/ledger.json');used=sum(r.get('accounted_tokens',r['reserved_tokens']) for r in ledger['attempts'].values())
    stages=settings['phase_limits'];proposed=sum(v['token_limit'] for v in stages.values())
    admissions=[]
    for phase,cap in settings['phase_limits'].items():
        with TemporaryDirectory(prefix='p7_admission_') as scratch:
            scratch=Path(scratch);write_json(scratch/'budget/ledger.json',ledger);write_json(scratch/'budget/prefix.json',ledger)
            g=phase_governor(scratch,phase,plan)
            prov=runtime.provider(cfg['llm'],'learning' if phase=='learning' else 'runtime',settings)
            task=PublicTask(**tasks['train'][0]);wire=prov._build_payload(runtime.messages(task),[])
            try:
                g.admit('zero_http_admission',wire,{'p7_phase':phase,'parent_scope':str(scratch/phase),
                    'parent_token_limit':700000 if phase=='learning' else 128000})
                admissions.append({'phase':phase,'admitted':True,'new_http':0})
            except BudgetExhausted as exc:admissions.append({'phase':phase,'admitted':False,'code':exc.code,'new_http':0})
    result={'status':'passed','method_version':c.VERSION,'new_model_http':0,'plan_hash':digest(plan),'capacity':caps,'admissions':admissions,
        'partition':{k:len(v) for k,v in read(root/'partition.json')['members'].items()},
        'billing_ready':not billing_blocked(ledger),'historical_tokens_or_reservations':used,'historical_http':len(ledger['attempts']),
        'proposed_new_tokens':proposed,'cumulative_remaining_tokens':ledger['limits']['token_limit']-used,
        'full_proposal_fits':used+proposed<=ledger['limits']['token_limit'],
        'test_worst_case_tokens':25600000,'test_proposed_tokens':stages['test']['token_limit'],
        'paid_authorized':False,'optional_program_workflow':'disabled; no claims of P7 utility',
        'recommend_release':False,'reason':'Engineering check only; real Train pairs/Gate/Val not executed'}
    write_json(root/'zero_http_check.json',result);return result


class StageContext:
    """Attribute every physical attempt to exactly one P7 phase."""
    def __init__(self,governor,phase):self.governor=governor;self.phase=phase
    def admit(self,key,wire,context):return self.governor.admit(key,wire,{**context,'p7_phase':self.phase})
    def __getattr__(self,name):return getattr(self.governor,name)


def run_learn(root,plan):
    root=Path(root);authorize(root,'learning',plan)
    if (root/'gate/lock.json').exists():raise ValueError('No learning after Gate lock')
    cfg=read(root/'config.json');sources=read(root/'partition.json')['members']['proposal_train']
    g=StageContext(phase_governor(root,'learning',plan),'learning')
    result=learner.learn(root/'learning/initial',cfg,g,read(root/'proposal_material.json'),sources)
    for candidate in result['candidates']:memory.append(root/'candidate_versions',candidate)
    memory.append(root/'memory',{'stage':'initial','result':result,'evidence_status':'unproven_hypothesis'})
    return result


def evaluate(root,plan,phase,policy,keys):
    root=Path(root);cfg=read(root/'config.json');authorize(root,phase,plan)
    tasks=read(root/'tasks.json');split=phase if phase in ('val','test') else 'train'
    by_key={t['physical_key']:t for t in tasks[split]}
    records=read(Path(plan['datasets'])/'livemath'/('evaluator_records_'+split+'.json'))
    adapter=AnswerAdapter('livemath',records);g=StageContext(phase_governor(root,phase,plan),phase)
    off=[];on=[];policy_id=digest(policy)
    for key in keys:
        task=PublicTask(**by_key[key])
        off.append(runtime.solve(task,adapter,[],cfg,root/phase/'off'/key,g))
        on.append(runtime.solve(task,adapter,policy,cfg,root/phase/'on'/policy_id/key,g))
    name='edit_feedback' if phase=='feedback' else 'protected_gate' if phase=='gate' else phase
    r=opt.receipt(off,on,policy,plan['source_manifest_hash'],name,
        {'tasks':len(keys),'min_net':2,'max_losses':1})
    write_json(root/phase/'receipts'/(policy_id+'.json'),r);return r


def policies(candidates):
    global_=[c for c in candidates if c['candidate']['scope']=='global']
    local=[c for c in candidates if c['candidate']['scope']=='local']
    return ([global_] if global_ else [])+[[*global_,c] for c in local]


def run_edit(root,plan):
    root=Path(root)
    if (root/'gate/lock.json').exists():raise ValueError('Protected Gate is sealed; no more edits')
    initial=read(root/'learning/initial/learning.json');candidates=initial['candidates']
    if not candidates:
        result={'status':'completed_no_candidate','best':None};write_json(root/'feedback/result.json',result);return result
    cfg=read(root/'config.json');members=read(root/'partition.json')['members'];history=[];known={c['id']:c for c in candidates}
    queue=policies(candidates);best=None;best_policy=[]
    for round_ in range(plan['settings']['edit_rounds']+1):
        for policy in queue:
            r=evaluate(root,plan,'feedback',policy,members['edit_feedback']);history.append(r)
            if best is None or opt.rank(r)<opt.rank(best):best=r;best_policy=policy
        write_json(root/'feedback/best.json',{'policy':best_policy,'receipt':best})
        if round_==plan['settings']['edit_rounds']:break
        authorize(root,'learning',plan)
        # Full public paired inputs and model responses, but neither Gate nor heldout historical answers.
        material=learner.editor_material(best_policy,best,history,plan['settings']['trace_window_bytes'])
        g=StageContext(phase_governor(root,'learning',plan),'learning')
        edited=learner.learn(root/'learning'/('edit_'+str(round_+1)),cfg,g,material,members['proposal_train'],
            parents={c['id']:c for c in best_policy})
        memory.append(root/'memory',{'stage':'edit','round':round_+1,'result':edited})
        queue=[]
        for child in edited['candidates']:
            known[child['id']]=child;memory.append(root/'candidate_versions',child)
            policy=[child if c['id']==child['parent_id'] else c for c in best_policy]
            queue.append(policy)
        if not queue:break
    result={'status':'feedback_qualified' if best['qualified'] else 'stop_negative_result',
        'best':best,'policy':best_policy,'versions_evaluated':len(history)}
    write_json(root/'feedback/result.json',result);return result


def run_gate(root,plan):
    root=Path(root);selection=read(root/'feedback/result.json')
    if selection['status']!='feedback_qualified':raise ValueError('No positive Train policy for Gate')
    lock={'policy':selection['policy'],'feedback_hash':digest(selection),'plan_hash':digest(plan)}
    path=root/'gate/lock.json'
    if path.exists() and read(path)!=lock:raise ValueError('Gate already locked to a different policy')
    write_json(path,lock)
    if (root/'gate/result.json').exists():return read(root/'gate/result.json')
    r=evaluate(root,plan,'gate',lock['policy'],read(root/'partition.json')['members']['protected_gate'])
    result={'status':'gate_qualified' if r['qualified'] else 'method_not_released','receipt':r,'lock_hash':digest(lock)}
    write_json(root/'gate/result.json',result);return result


def freeze(root,plan):
    root=Path(root);gate=read(root/'gate/result.json');feedback=read(root/'feedback/result.json')
    lock=read(root/'gate/lock.json')
    if (gate['status']!='gate_qualified' or not opt.verify_receipt(gate['receipt']) or not opt.verify_receipt(feedback['best'])
        or gate['lock_hash']!=digest(lock) or lock['feedback_hash']!=digest(feedback) or lock['policy']!=feedback['policy']):
        raise ValueError('Failed/invalid Gate cannot publish')
    for receipt in (feedback['best'],gate['receipt']):
        if receipt['candidate_ids']!=[c['id'] for c in feedback['policy']]:raise ValueError('Final policy differs from paired evidence')
        for row in receipt['rows']:
            for arm in ('off','on'):runtime.verify_trace(row[arm])
    if (root/'freeze_identity.json').exists():return verify_frozen(root)
    bank=Bank(root/'active_bank');bank.bind_choice_method(c.VERSION)
    try:
        for candidate in feedback['policy']:bank.put('skill',c.asset(candidate,[feedback['best'],gate['receipt']]))
        manifest=bank.freeze(root/'frozen_bank')
    finally:bank.close()
    value={'method_version':c.VERSION,'policy':feedback['policy'],'gate_hash':digest(gate),
        'plan_hash':digest(plan),'manifest':manifest,'frozen_tree':tree_identity(root/'frozen_bank')}
    write_json(root/'freeze_identity.json',value);return value


def verify_frozen(root):
    root=Path(root);value=read(root/'freeze_identity.json')
    if value['frozen_tree']!=tree_identity(root/'frozen_bank'):raise ValueError('Frozen bytes changed')
    bank=Bank(root/'frozen_bank',readonly=True)
    try:
        if any(not c.checked_effect_asset(a) for a in bank.all('skill')):raise ValueError('Invalid Frozen P7 publication')
    finally:bank.close()
    return value


def run_eval(root,plan,phase):
    root=Path(root);frozen=verify_frozen(root)
    if phase=='test':authorize(root,'test',plan)
    keys=[t['physical_key'] for t in read(root/'tasks.json')[phase]]
    receipt=evaluate(root,plan,phase,frozen['policy'],keys)
    verify_frozen(root);write_json(root/phase/'paired_report.json',receipt)
    return review(root,plan) if phase=='val' else receipt


def test_identity(root,plan):
    root=Path(root)
    return {'val_report_hash':sha256(root/'val/paired_report.json'),'method':c.VERSION,
        'frozen_hash':verify_frozen(root)['frozen_tree']['sha256'],'model_hash':digest(read(root/'config.json')['llm']),
        'budget':plan['settings']['phase_limits']['test'],'tasks_hash':digest(read(root/'tasks.json')['test']),
        'plan_hash':digest(plan)}


def review(root,plan):
    root=Path(root);r=read(root/'val/paired_report.json')
    if not opt.verify_receipt(r):raise ValueError('Val evidence invalid')
    tokens=[row[arm]['tokens'] for row in r['rows'] for arm in ('off','on')]
    estimate=max(tokens)*200  # conservative observed maximum; not a theoretical guarantee
    allowance=plan['settings']['phase_limits']['test']['token_limit']
    value={'recommend_release':r['qualified'] and estimate<=allowance,'metrics':metrics(r),
        'val_net_correct':r['net_correct'],'val_losses':r['losses'],'val_qualified':r['qualified'],
        'test_observed_max_projection_tokens':estimate,'test_theoretical_max_tokens':25600000,
        'test_budget_sufficiency':estimate<=allowance,'test_identity':test_identity(root,plan),
        'reason':'Review actual paired effects and cost before separately authorizing Test',
        'test_development_history_disclosed':True}
    write_json(root/'review.json',value);return value


def metrics(receipt):
    rows=receipt['rows'];n=len(rows)
    def interval(k,n):
        if not n:return None
        z=1.95996398454;p=k/n;denom=1+z*z/n
        center=(p+z*z/(2*n))/denom;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
        return [center-half,center+half]
    def group(rows):
        return {arm:{'tasks':len(rows),'correct':sum(r[arm]['score']['hard'] for r in rows),
            'accuracy':sum(r[arm]['score']['hard'] for r in rows)/len(rows) if rows else None,
            'accuracy_wilson95':interval(sum(r[arm]['score']['hard'] for r in rows),len(rows)),
            'empty':sum(not r[arm]['valid'] for r in rows),'tokens':sum(r[arm]['tokens'] for r in rows)} for arm in ('off','on')}
    value=group(rows);rng=random.Random(42)
    diffs=[int(r['on']['score']['hard'])-int(r['off']['score']['hard']) for r in rows]
    bootstrap=sorted(sum(rng.choice(diffs) for _ in diffs)/n for _ in range(2000)) if n else []
    value['paired_gain_bootstrap95']=[bootstrap[49],bootstrap[1949]] if bootstrap else None
    value['wins']=receipt['wins'];value['losses']=receipt['losses']
    for arm in ('off','on'):
        tokens=sorted(r[arm]['tokens'] for r in rows)
        value[arm].update(mean_tokens=statistics.mean(tokens) if tokens else None,
            median_tokens=statistics.median(tokens) if tokens else None,p95_tokens=tokens[math.ceil(.95*n)-1] if n else None,
            http=sum(r[arm]['http'] for r in rows),latency_ms=sum(r[arm].get('latency_ms',0) for r in rows),
            usd=None,usd_reason='No frozen provider price table; raw usage retained')
    value['meta_option_present']=group([r for r in rows if r['off']['selection']['features']['meta_option_present']])
    value['meta_option_absent']=group([r for r in rows if not r['off']['selection']['features']['meta_option_present']])
    value['on_off_token_ratio']=receipt['tokens_on']/receipt['tokens_off'] if receipt['tokens_off'] else None
    return value


def report(root):
    root=Path(root);plan=checked(root);ledger=read(root/'budget/ledger.json');prefix=read(root/'budget/prefix.json')
    new={k:v for k,v in ledger['attempts'].items() if k not in prefix['attempts']}
    result={'method_version':c.VERSION,'run_id':plan['run_id'],'code':plan['code'],'new_http':len(new),
        'new_tokens':sum(r.get('accounted_tokens',r['reserved_tokens']) for r in new.values()),
        'billing_ready':not billing_blocked(ledger),'recommend_release':False,'test_development_history_disclosed':True}
    for stage,path in [('learning','learning/initial/learning.json'),('feedback','feedback/result.json'),('gate','gate/result.json'),('review','review.json')]:
        if (root/path).exists():result[stage]=read(root/path)
    result['recommend_release']=result.get('review',{}).get('recommend_release',False)
    usage={phase:{'tokens':sum(r.get('accounted_tokens',r['reserved_tokens']) for r in new.values() if r['context'].get('p7_phase')==phase),
        'http':sum(r['context'].get('p7_phase')==phase for r in new.values())} for phase in plan['settings']['phase_limits']}
    write_json(root/'usage_by_phase.json',usage)
    for folder,name in [('candidate_versions','candidate_versions.jsonl'),('memory','experience_memory.jsonl')]:
        values=memory.events(root/folder)
        (root/name).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values),encoding='utf-8')
    write_json(root/'report.json',result)
    (root/'report.md').write_text('# P7 review\n\n'+json.dumps({k:v for k,v in result.items() if k not in {'learning','feedback','gate'}},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {k:v for k,v in result.items() if k not in {'learning','feedback','gate'}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','check','learn','edit','gate','freeze','val','review','test','report'])
    parser.add_argument('--output',required=True);parser.add_argument('--config',default=str(CONTRACT))
    parser.add_argument('--source-run');parser.add_argument('--datasets');parser.add_argument('--env-file')
    parser.add_argument('--zero-http',action='store_true');args=parser.parse_args()
    if args.zero_http and args.stage not in ('prepare','check','report'):
        parser.error('--zero-http cannot execute a paid stage')
    root=Path(args.output)
    if args.env_file:load_env(args.env_file)
    try:
        if args.stage=='prepare':result=prepare(root,args.config,args.source_run,args.datasets)
        elif args.stage=='check':result=check(root)
        elif args.stage=='report':result=report(root)
        else:
            plan=checked(root)
            functions={'learn':run_learn,'edit':run_edit,'gate':run_gate,'freeze':freeze,'review':review}
            result=functions[args.stage](root,plan) if args.stage in functions else run_eval(root,plan,args.stage)
    except BudgetExhausted as exc:
        result={'status':'budget_stopped','code':exc.code,'stage':args.stage,'automatic_retry':False}
        write_json(root/'review_required.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
