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
    version=settings['method_version']
    cfg['experiment']={**cfg['experiment'],'choice_method_version':version,'runtime_mode':'frozen',
        'implementation_revision':version,'output_dir':str(root),'seed':42,'benchmark':'livemath'}
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
    if settings['method_version']==c.R2:return prepare_r2(root,config,source_run,datasets,settings,code)
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
    if s.get('method_version')==c.R2:return validate_settings_r2(s)
    if (s['method_version']!=c.VERSION or s['partition']!={'proposal_train':28,'edit_feedback':12,'protected_gate':20}
        or s['initial_candidates']!=3 or s['edit_rounds']!=2 or s['edits_per_round']!=2 or s['program_enabled']):
        raise ValueError('P7 method contract differs from confirmed text-first scope')
    if s['gate']!={'min_net':2,'max_losses':1,'no_new_empty':True}:raise ValueError('P7 qualification changed')
    if s['runtime']!={'completion':65536,'finish_completion':16384,'timeout':600,'parent_tokens':128000,'tail_bytes':16384}:
        raise ValueError('P7 Runtime contract changed')
    if sha256(Path(s['tokenizer_path']))!=s['tokenizer_sha256']:raise ValueError('Tokenizer identity differs')


def checked(root):
    root=Path(root);p=read(root/'plan.json')
    if p['method_version'] not in {c.VERSION,c.R2} or p['code']!=code_identity() or digest(p)!=read(root/'plan_identity.json')['sha256']:
        raise ValueError('P7 source or plan changed; register a new version')
    for path,h in p['files'].items():
        if sha256(Path(path))!=h:raise ValueError('Locked source changed: '+path)
    for path,h in p['prepared_hashes'].items():
        if sha256(root/path)!=h:raise ValueError('Prepared input changed: '+path)
    ledger=read(root/'budget/ledger.json')
    anchors=['budget/prefix.json']+(['budget/recovery_prefix.json'] if p.get('continuation') else [])
    for name in anchors:
        prefix=read(root/name)
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
        review=read(Path(root)/'review.json')
        if value.get('review_identity')!=identity or not review['review_completed'] or not review['test_budget_sufficiency']:
            raise ValueError('Test needs approval bound to the completed Val review and exact Frozen')
    if billing_blocked(read(Path(root)/'budget/ledger.json')):raise ValueError('Unresolved billing blocks P7')


def check(root):
    root=Path(root);plan=checked(root);cfg=read(root/'config.json');settings=plan['settings'];caps=[]
    if plan['method_version']==c.R2:return check_r2(root,plan)
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
    if plan['method_version']==c.R2:return run_learn_r2(root,plan)
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
    if plan['method_version']==c.R2:return run_edit_r2(root,plan)
    root=Path(root)
    if (root/'gate/lock.json').exists():raise ValueError('Protected Gate is sealed; no more edits')
    initial=read(root/'learning/initial/learning.json');candidates=initial['candidates']
    if not candidates:
        result={'status':'completed_no_candidate','best':None};write_json(root/'feedback/result.json',result);return result
    cfg=read(root/'config.json');members=read(root/'partition.json')['members'];history=[]
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
            memory.append(root/'candidate_versions',child)
            policy=[child if c['id']==child['parent_id'] else c for c in best_policy]
            queue.append(policy)
        if not queue:break
    result={'status':'feedback_qualified' if best['qualified'] else 'stop_negative_result',
        'best':best,'policy':best_policy,'versions_evaluated':len(history)}
    write_json(root/'feedback/result.json',result);return result


def run_gate(root,plan):
    if plan['method_version']==c.R2:raise ValueError('R2 has no new protected Gate stage')
    root=Path(root);selection=read(root/'feedback/result.json')
    if selection['status']!='feedback_qualified':raise ValueError('No positive Train policy for Gate')
    authorize(root,'gate',plan)
    lock={'policy':selection['policy'],'feedback_hash':digest(selection),'plan_hash':digest(plan)}
    path=root/'gate/lock.json'
    if path.exists() and read(path)!=lock:raise ValueError('Gate already locked to a different policy')
    write_json(path,lock)
    if (root/'gate/result.json').exists():return read(root/'gate/result.json')
    r=evaluate(root,plan,'gate',lock['policy'],read(root/'partition.json')['members']['protected_gate'])
    result={'status':'gate_qualified' if r['qualified'] else 'method_not_released','receipt':r,'lock_hash':digest(lock)}
    write_json(root/'gate/result.json',result);return result


def freeze(root,plan):
    if plan['method_version']==c.R2:return freeze_r2(root,plan)
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
    if value['method_version']==c.R2:return verify_frozen_r2(root)
    if value['frozen_tree']!=tree_identity(root/'frozen_bank'):raise ValueError('Frozen bytes changed')
    bank=Bank(root/'frozen_bank',readonly=True)
    try:
        if any(not c.checked_effect_asset(a) for a in bank.all('skill')):raise ValueError('Invalid Frozen P7 publication')
    finally:bank.close()
    return value


def run_eval(root,plan,phase):
    if plan['method_version']==c.R2:
        if phase!='val':raise ValueError('R2 authorizes Val observation only; Test is not authorized')
        value=verify_frozen_r2(root)
        result=evaluate_r2(root,plan,'val',value['policy'],[t['physical_key'] for t in read(Path(root)/'tasks.json')['val']])
        verify_frozen_r2(root);write_json(Path(root)/'val/paired_report.json',result)
        return result
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
    value={'review_completed':True,'recommend_release':r['net_correct']>0 and estimate<=allowance,'metrics':metrics(r),
        'val_net_correct':r['net_correct'],'val_losses':r['losses'],'val_qualified':r['qualified'],
        'test_observed_max_projection_tokens':estimate,'test_theoretical_max_tokens':25600000,
        'test_budget_sufficiency':estimate<=allowance,'test_identity':test_identity(root,plan),
        'reason':'Val is descriptive, not an additional trained selection threshold. Explicit human Test review is required.',
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
    if plan['method_version']==c.R2:return report_r2(root,plan)
    new={k:v for k,v in ledger['attempts'].items() if k not in prefix['attempts']}
    result={'method_version':c.VERSION,'run_id':plan['run_id'],'code':plan['code'],'new_http':len(new),
        'new_metered_tokens':sum(r.get('accounted_tokens',0) for r in new.values()),
        'new_reserved_unresolved_tokens':sum(r['reserved_tokens'] for r in new.values() if 'accounted_tokens' not in r),
        'billing_ready':not billing_blocked(ledger),'recommend_release':False,'test_development_history_disclosed':True}
    result['progress']={phase:{'off_completed':len(list((root/phase/'off').glob('*/trace.json'))),
        'on_completed':len(list((root/phase/'on').glob('*/*/trace.json')))} for phase in ('feedback','gate','val','test')}
    result['candidate_versions']=len(memory.events(root/'candidate_versions'))
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


def validate_settings_r2(s):
    if (s['seed']!=42 or s['initial_candidates']!=3 or s['edit_rounds']!=2 or s['edits_per_round']!=2 or
        s['program_enabled'] or s['max_injection_tokens']!=600 or s['structural_repair_limit']!=1 or
        s['learning_scope']!='all_train60' or set(s['phase_limits'])!={'learning','feedback','val'}):
        raise ValueError('R2 differs from the authorized finite Train60/Train12/Val17 experiment')
    if s['runtime']!={'completion':65536,'finish_completion':16384,'timeout':600,'parent_tokens':128000,'tail_bytes':16384}:
        raise ValueError('R2 preserves the P7 Runtime budget')
    if s['phase_limits']['learning']['request_limit']!=16 or sum(x['token_limit'] for x in s['phase_limits'].values())!=4376292:
        raise ValueError('R2 must retain the unspent original P7 6.2M envelope')
    if s['cumulative_limits']!={'token_limit':45000000,'request_limit':9165,'finish_reserve':0}:
        raise ValueError('Cumulative authority changed')
    if sha256(Path(s['tokenizer_path']))!=s['tokenizer_sha256']:raise ValueError('Tokenizer identity differs')


def prepare_r2(root,config,source_run,datasets,settings,code):
    evidence=source.load(source_run,datasets);parent=Path(settings['parent_run'])
    previous=read(parent/'plan.json');partition=read(parent/'partition.json')
    if previous['source_run_id']!=evidence['manifest']['run_id'] or partition!=evidence['partition']:
        raise ValueError('R2 source or old 28/12/20 identity differs')
    if read(parent/'controller_status.json')['status']!='completed_negative_gate':raise ValueError('P7 parent must be stopped and preserved')
    prefix,origins=merge_ledgers(settings['ledger_sources'],settings['cumulative_limits'])
    if billing_blocked(prefix):raise ValueError('New unresolved billing blocks preparation')
    occupied=sum(x.get('accounted_tokens',x['reserved_tokens']) for x in prefix['attempts'].values())
    new_tokens=sum(x['token_limit'] for x in settings['phase_limits'].values())
    if occupied+new_tokens>settings['cumulative_limits']['token_limit']:raise ValueError('R2 allocation exceeds cumulative balance')
    usage=read(parent/'usage_by_phase.json')
    if sum(x['tokens'] for x in usage.values())+new_tokens!=6200000:raise ValueError('Old P7 spent + R2 remainder differs from original authority')
    files=evidence['files']
    for path in [Path(config),Path(settings['tokenizer_path']),*[Path(x) for x in settings['ledger_sources']],
                 parent/'plan.json',parent/'partition.json',parent/'learning/initial/learning.json',parent/'usage_by_phase.json']:
        files[str(path.resolve())]=sha256(path)
    root.mkdir(parents=True);cfg=configure(evidence['config'],settings,root)
    keys=[t['physical_key'] for t in evidence['tasks']['train']]
    material=source.proposal_material_r2(evidence,settings['process_excerpt_bytes'])
    batches=source.material_batches_r2(material,settings['source_batch_bytes'])
    # One induction per byte batch, one optional consolidation, two editors, each with at most one repair.
    upper=2*len(batches)+(2 if len(batches)>1 else 0)+4+2
    if upper>16:raise ValueError('All-source learning cannot fit the frozen physical request allowance')
    with_bank=Bank(root/'active_bank');with_bank.bind_choice_method(c.R2)
    with with_bank.db:with_bank.db.execute('INSERT INTO metadata VALUES(?,?)',('effect_source_keys',json.dumps(keys)))
    initial=with_bank.digest();with_bank.close()
    for name,value in [('config.json',cfg),('partition.json',partition),('tasks.json',evidence['tasks']),
        ('source_catalog.json',evidence['catalog']),('source_materials.json',material),('learning_batches.json',batches),
        ('legacy_learning.json',read(parent/'learning/initial/learning.json')),('budget/ledger.json',prefix),
        ('budget/prefix.json',prefix),('budget/origins.json',origins)]:write_json(root/name,value)
    plan={'method_version':c.R2,'code':code,'created_at':utc(),'settings':settings,'source_run':str(source_run),
        'source_run_id':evidence['manifest']['run_id'],'parent_run':str(parent),'parent_plan_hash':digest(previous),
        'datasets':str(Path(datasets).resolve()),'files':files,'initial_bank_digest':initial,'source_manifest_hash':digest(keys),
        'learning_source_keys':keys,'historical_partition_retained':True,'feedback_is_independent_holdout':False,
        'test_development_history_disclosed':True,'val_development_history_disclosed':True,'test_authorized':False,
        'budget_allocation':{'prior_p7_spent_tokens':6200000-new_tokens,'new_tokens':new_tokens,
            'historical_cumulative_occupied':occupied,'combined_max_tokens':occupied+new_tokens,
            'learning_max_http_including_repairs':16,'all_source_call_upper_bound':upper},
        'prepared_hashes':{n:sha256(root/n) for n in ['config.json','partition.json','tasks.json','source_catalog.json',
            'source_materials.json','learning_batches.json','legacy_learning.json','budget/prefix.json','budget/origins.json']}}
    plan['run_id']='p7_r2_'+digest(plan)[:24]
    write_json(root/'plan.json',plan);write_json(root/'plan_identity.json',{'sha256':digest(plan)})
    write_json(root/'progress.json',{'status':'prepared','stage':'offline_check','delivery_ready':False})
    return check_r2(root,plan)


def check_r2(root,plan):
    root=Path(root);cfg=read(root/'config.json');settings=plan['settings'];caps=[]
    p=runtime.provider(cfg['llm'],'learning',settings)
    def admit_offline(wire,phase,parent_limit,purpose='runtime'):
        # Real production admission against an isolated ledger; never touches paid usage.
        with TemporaryDirectory() as directory:
            g=BudgetGovernor(Path(directory)/'ledger.json',**settings['phase_limits'][phase],finish_reserve=0)
            g.admit('offline_only',wire,{'parent_scope':'capacity','parent_token_limit':parent_limit,
                'episode_scope':'capacity','episode_request_limit':2,'purpose':purpose})
    migrated=[]
    for raw in read(root/'legacy_learning.json').get('proposal',{}).get('candidates',[]):
        try:
            v=c.migrate_candidate_r2(raw,plan['learning_source_keys'],all_have_choices=True)
            migrated.append(c.candidate(v,plan['learning_source_keys'],{'response_hash':'offline_migration'},version=c.R2))
        except c.ContractError:pass  # The execution path retains explicit targeted repair diagnostics.
    materials=[('induction_'+str(i),learner.PROMPT_R2,{'source_materials':b,'historical_candidates':migrated},c.BATCH_R2)
        for i,b in enumerate(read(root/'learning_batches.json'))]
    old=read(Path(plan['parent_run'])/'feedback/result.json')['best']
    feedback=opt.receipt_r2([r['off'] for r in old['rows']],[r['on'] for r in old['rows']],migrated,
        plan['source_manifest_hash'],'historical_capacity_replay_only')
    editor=learner.editor_material_r2(migrated,feedback,[feedback]*3,read(root/'source_materials.json'))
    materials.append(('editor_historical_replay',learner.EDITOR_R2,editor,c.EDIT_R2))
    # A maximum-completion-sized malformed output is retained alongside full source inputs.
    repair={'errors':[{'index':0,'original':'x'*settings['learning_completion']*4,'diagnostics':[]}],
        'sources_shown':plan['learning_source_keys'],'original_material':editor,
        'preserve_legal_siblings':[c.card(x['candidate']) for x in migrated]}
    materials.append(('editor_repair_capacity',learner.PROMPT_R2,repair,c.REPAIR_EDIT_R2))
    for name,prompt,material,schema in materials:
        wire=p._build_payload(learner.request_material(prompt,material),
            [NativeToolSpec('submit_effect','Submit Train-only candidate or edit',schema)],
            {'type':'function','function':{'name':'submit_effect'}})
        size=input_token_bound(wire)
        if size>settings['learning_wire_bytes']:raise ValueError('Complete source batch exceeds wire capacity')
        admit_offline(wire,'learning',settings['phase_limits']['learning']['token_limit'])
        caps.append({'stage':name,'wire_bytes':size,'reserve':size+p.config.max_completion_tokens,'production_admission':'passed'})
    tasks=read(root/'tasks.json');rows=[*tasks['train'],*tasks['val']]
    for row in rows:
        task=PublicTask(**row);main=runtime.provider(cfg['llm'],'runtime',settings);finish=runtime.provider(cfg['llm'],'finish',settings)
        cards=[c.card(x['candidate']) for x in runtime.select(migrated,task,settings['tokenizer_path'],600,version=c.R2)['selected']]
        wire=main._build_payload(runtime.messages(task,cards),[])
        tail={'text':'x'*16384,'source_request_id':'x'*64,'response_sha256':'x'*64,'reasoning_sha256':'x'*64,
            'full_bytes':65536,'start_byte':49152,'end_byte':65536,'truncated':True,'qualification':'unverified'}
        end=finish._build_payload(runtime.messages(task,cards,tail=tail),[])
        if input_token_bound(wire)+65536+input_token_bound(end)+16384>128000:raise ValueError('Public inputs do not fit unchanged parent budget')
        phase='feedback' if row['split']=='train' else 'val'
        admit_offline(wire,phase,128000)
        admit_offline(end,phase,128000-input_token_bound(wire)-65536,'finish_only')
        caps.append({'stage':row['split'],'task_id':row['task_id'],'main_max_tokens':wire['max_tokens'],
            'finish_max_tokens':end['max_tokens'],'timeout':main.config.request_timeout_seconds,'parent_tokens':128000,
            'production_admission':'passed','candidate_fixture':'historical_migration'})
    result={'status':'passed','method_version':c.R2,'new_model_http':0,'plan_hash':digest(plan),'capacity':caps,
        'source_tasks':len(plan['learning_source_keys']),'feedback_tasks':12,'val_tasks':17,'test_authorized':False,
        'source_request_upper_bound':plan['budget_allocation']['all_source_call_upper_bound'],
        'budget_allocation':plan['budget_allocation'],'first_feedback_worst_case_tokens':3072000,
        'feedback_phase_token_limit':settings['phase_limits']['feedback']['token_limit'],
        'worst_case_completion_guaranteed':False,'budget_basis':'observed original P7 distributions, hard caps still apply'}
    write_json(root/'zero_http_check.json',result);return result


def store_r2(root,candidates):
    bank=Bank(Path(root)/'active_bank')
    try:
        for value in candidates:
            memory.append(Path(root)/'candidate_versions',value);bank.put('skill',c.asset(value))
    finally:bank.close()


def run_learn_r2(root,plan):
    root=Path(root);authorize(root,'learning',plan)
    if (root/'freeze_identity.json').exists():raise ValueError('R2 candidates are frozen; no further learning')
    final=root/'learning/result.json'
    if final.exists():return read(final)
    cfg=read(root/'config.json');sources=plan['learning_source_keys'];g=StageContext(phase_governor(root,'learning',plan),'learning')
    old=read(root/'legacy_learning.json');seeds=[];repair=[]
    all_choices=all(len(t['inputs']['choices'])>1 for t in read(root/'tasks.json')['train'])
    for index,raw in enumerate(old.get('proposal',{}).get('candidates',[])):
        try:
            migrated=c.migrate_candidate_r2(raw,sources,all_have_choices=all_choices)
            c.validate_candidate_r2(migrated,sources,tokenizer_path=cfg['effect_p7']['tokenizer_path'])
            evidence={**old['model_evidence'],'migration':'contract_equivalent_migration',
                'original_candidate_hash':digest(raw),'original_candidate':raw,'new_natural_generation':False}
            seeds.append(c.candidate(migrated,sources,evidence,version=c.R2))
        except c.ContractError as exc:repair.append({'index':index,'candidate':raw,'diagnostics':exc.diagnostics})
    write_json(root/'learning/migration.json',{'candidates':seeds,'targeted_repair_needed':repair,'original_response_hash':old['model_evidence']['response_hash']})
    store_r2(root,seeds);all_material=read(root/'source_materials.json');pool=[]
    if repair:
        repaired=learner.learn(root/'learning/migration_repair',cfg,g,{'source_materials':all_material,'migration_targets':repair,
            'instruction':'Repair only indicated candidate contract fields. Choose when if the condition is real; never silently erase it.'},sources)
        store_r2(root,repaired['candidates']);seeds.extend(repaired['candidates'])
    for i,batch in enumerate(read(root/'learning_batches.json')):
        shown={r['case_id'] for r in batch}
        existing=[x for x in seeds if set(x['candidate']['source_case_ids'])<=shown]
        material={'source_materials':batch,'historical_candidates':[{k:v for k,v in x.items() if k!='model_evidence'} for x in existing],
            'instruction':'Use the complete Train batch to form up to three complementary Atomic candidates. Preserve useful historical behavior; any new diagnosis is a hypothesis.'}
        result=learner.learn(root/'learning'/('induction_'+str(i)),cfg,g,material,sources)
        pool.extend(result['candidates']);store_r2(root,result['candidates']);progress_r2(root,stage='learning')
    if len(read(root/'learning_batches.json'))>1 and pool:
        result=learner.learn(root/'learning/consolidation',cfg,g,{'source_materials':all_material,
            'batch_candidates':[{k:v for k,v in x.items() if k!='model_evidence'} for x in pool],
            'instruction':'One Train-only consolidation into at most three complementary candidates; preserve useful process differences.'},sources)
        store_r2(root,result['candidates']);pool=result['candidates'] or pool
    # Frozen generation order, not heldout scores. Historical candidates remain in the candidate store.
    selected=[];counts={'all_tasks':0,'when':0};seen=set()
    for value in [*pool,*seeds]:
        mode=value['candidate']['applicability']['mode'];content=digest(c.card(value['candidate']))
        if content in seen or counts[mode]>=(1 if mode=='all_tasks' else 2):continue
        selected.append(value);counts[mode]+=1;seen.add(content)
        if len(selected)==3:break
    result={'status':'candidates' if selected else 'no_candidate','candidates':selected,
        'all_source_tasks':len(sources),'historical_migrations':len(seeds),'natural_new_candidates':len(pool)}
    write_json(final,result);return result


def evaluate_r2(root,plan,phase,policy,keys,version_index=0):
    root=Path(root);authorize(root,phase,plan);cfg=read(root/'config.json');split='train' if phase=='feedback' else 'val'
    tasks={t['physical_key']:t for t in read(root/'tasks.json')[split]}
    records=read(Path(plan['datasets'])/'livemath'/('evaluator_records_'+split+'.json'))
    adapter=AnswerAdapter('livemath',records);g=StageContext(phase_governor(root,phase,plan),phase)
    policy_id=digest(policy);off=[];on=[];reuse=[]
    progress_r2(root,stage=phase,policy_id=policy_id,version_index=version_index)
    for key in keys:
        task=PublicTask(**tasks[key]);before=(root/phase/'off'/key/'trace.json').exists()
        a=runtime.solve(task,adapter,[],cfg,root/phase/'off'/key,g);off.append(a)
        if before and phase=='feedback' and version_index>0:reuse.append({'physical_key':key,'trace_hash':a['trace_hash'],'physical_request_ids':a['request_ids'],'new_http':0})
        progress_r2(root,stage=phase,policy_id=policy_id,version_index=version_index,last_task=task.task_id)
        on.append(runtime.solve(task,adapter,policy,cfg,root/phase/'on'/policy_id/key,g))
        progress=progress_r2(root,stage=phase,policy_id=policy_id,version_index=version_index,last_task=task.task_id)
        print(json.dumps({k:progress[k] for k in ['stage','version_index','paired_completed','per_arm_normal_completed','measured_tokens','new_http','last_completed_task']},ensure_ascii=False),flush=True)
    r=opt.receipt_r2(off,on,policy,plan['source_manifest_hash'],'train_feedback12' if phase=='feedback' else 'val17_observation')
    write_json(root/phase/'receipts'/(policy_id+'.json'),r)
    write_json(root/phase/'baseline_reuse'/(policy_id+'.json'),{'reused':reuse,'baseline_batches_paid':1,'new_baseline_http':0 if reuse else sum(x['http'] for x in off)})
    return r


def policy_r2(candidates):
    seen=set();result=[]
    for value in candidates:
        if value['id'] not in seen:result.append(value);seen.add(value['id'])
    # The full ordered roster retains complementary locals. select_r2 injects at most
    # one all_tasks and one applicable when candidate, under the combined 600-token cap.
    if len(result)>3:raise ValueError('R2 policy roster exceeds three candidates')
    return result


def run_edit_r2(root,plan):
    root=Path(root);out=root/'feedback/result.json'
    if out.exists():return read(out)
    if (root/'freeze_identity.json').exists():raise ValueError('R2 policy is frozen')
    initial=read(root/'learning/result.json');policy=policy_r2(initial['candidates'])
    if not policy:
        value={'status':'no_candidate','policy':[],'versions_evaluated':0};write_json(out,value);return value
    cfg=read(root/'config.json');keys=read(root/'partition.json')['members']['edit_feedback'];history=[]
    r=evaluate_r2(root,plan,'feedback',policy,keys);history.append(r);best=r;best_policy=policy
    write_json(root/'feedback/first_policy.json',{'policy_id':digest(policy),'policy':policy})
    for round_ in range(1,3):
        progress_r2(root,stage='learning')
        material=learner.editor_material_r2(best_policy,best,history,read(root/'source_materials.json'))
        g=StageContext(phase_governor(root,'learning',plan),'learning');authorize(root,'learning',plan)
        edited=learner.learn(root/'learning'/('edit_'+str(round_)),cfg,g,material,plan['learning_source_keys'],
            parents={x['id']:x for x in best_policy})
        store_r2(root,edited['candidates']);memory.append(root/'memory',{'round':round_,'result':edited})
        if not edited['candidates']:continue  # One no_change does not discard the remaining substantive round.
        children={x['parent_id']:x for x in edited['candidates']}
        updated=policy_r2([children.get(x['id'],x) for x in best_policy])
        if digest(updated) in {digest(x) for x in [best_policy]}:continue
        r=evaluate_r2(root,plan,'feedback',updated,keys,len(history));history.append(r)
        incumbent=opt.receipt_r2([x['on'] for x in best['rows']],[x['on'] for x in r['rows']],updated,
            plan['source_manifest_hash'],'train_feedback_vs_incumbent')
        write_json(root/'feedback/incumbent_comparisons'/(digest(updated)+'.json'),incumbent)
        if opt.rank(r)<opt.rank(best):best=r;best_policy=updated
        write_json(root/'feedback/best.json',{'policy':best_policy,'receipt':best})
    value={'status':'observed_gain' if best['observed_gain'] else 'no_observed_gain','policy':best_policy,
        'best':best,'versions_evaluated':len(history),'receipts':[x['receipt_hash'] for x in history],
        'qualified':False,'evaluation_status':'experimental_not_claimed_effective'}
    write_json(out,value);return value


def freeze_r2(root,plan):
    root=Path(root)
    if (root/'freeze_identity.json').exists():return verify_frozen_r2(root)
    selected=read(root/'feedback/result.json');policy=selected['policy']
    if not policy:raise ValueError('No candidate to freeze')
    if not opt.verify_receipt_r2(selected['best']):raise ValueError('R2 feedback evidence changed')
    for row in selected['best']['rows']:
        for arm in ('off','on'):runtime.verify_trace(row[arm])
    bank=Bank(root/'deployment_bank');bank.bind_choice_method(c.R2)
    try:
        with bank.db:
            bank.db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)',('effect_source_keys',json.dumps(plan['learning_source_keys'])))
            bank.db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)',('effect_deployed_ids',json.dumps([x['id'] for x in policy])))
        for candidate in policy:bank.put('skill',c.asset(candidate,[selected['best']]))
        if (root/'frozen_bank/freeze.json').exists():
            manifest=read(root/'frozen_bank/freeze.json')
            existing=Bank(root/'frozen_bank',readonly=True)
            try:
                if manifest['source_digest']!=bank.digest() or existing.digest()!=manifest['digest']:
                    raise ValueError('Interrupted Frozen snapshot differs from deployment')
            finally:existing.close()
        else:manifest=bank.freeze(root/'frozen_bank')
    finally:bank.close()
    value={'method_version':c.R2,'policy':policy,'evaluation_status':'experimental_not_claimed_effective',
        'feedback_hash':digest(selected),'plan_hash':digest(plan),'manifest':manifest,'frozen_tree':tree_identity(root/'frozen_bank')}
    write_json(root/'freeze_identity.json',value);return verify_frozen_r2(root)


def verify_frozen_r2(root):
    root=Path(root);value=read(root/'freeze_identity.json')
    if value['frozen_tree']!=tree_identity(root/'frozen_bank') or value['feedback_hash']!=digest(read(root/'feedback/result.json')):
        raise ValueError('R2 frozen policy or evidence changed')
    bank=Bank(root/'frozen_bank',readonly=True)
    try:
        assets=bank.all('skill')
        if any(not c.checked_experimental_asset(a) for a in assets) or {a['effect_candidate']['id'] for a in assets}!={x['id'] for x in value['policy']}:
            raise ValueError('Experimental Bank differs from locked policy')
    finally:bank.close()
    return value


def progress_r2(root,*,stage=None,policy_id=None,version_index=None,last_task=None,status='running'):
    import os
    root=Path(root);old=read(root/'progress.json') if (root/'progress.json').exists() else {}
    stage=stage or old.get('stage','learning');policy_id=policy_id or old.get('policy_id');version_index=version_index if version_index is not None else old.get('version_index',0)
    arms={a:[read(p) for p in (root/stage/a/(policy_id if a=='on' and policy_id else '')).rglob('trace.json')] for a in ('off','on')}
    normal={a:sum(t['valid'] and bool(t['raw_usage']) and all(u is not None for u in t['raw_usage']) for t in rows) for a,rows in arms.items()}
    reused=stage=='feedback' and version_index>0
    counts={a:len(v) for a,v in arms.items()};n=12 if stage=='feedback' else 17 if stage=='val' else 0
    ledger=read(root/'budget/ledger.json');prefix=read(root/'budget/prefix.json');new={k:v for k,v in ledger['attempts'].items() if k not in prefix['attempts']}
    first_path=root/'feedback/first_policy.json';first_normal=False
    if first_path.exists():
        pid=read(first_path)['policy_id']
        first_normal=all(sum(read(p)['valid'] for p in (root/'feedback'/arm/(pid if arm=='on' else '')).rglob('trace.json'))==12 for arm in ('off','on'))
    started_val=(root/'val').exists();val_normal=not started_val
    if started_val:
        val_normal=all(sum(read(p)['valid'] for p in (root/'val'/arm).rglob('trace.json'))==17 for arm in ('off','on'))
    started_arms_complete=all(len(ts)==12 and all(t['valid'] for t in ts) for folder in (root/'feedback/on').glob('*')
        if folder.is_dir() for ts in [[read(p) for p in folder.rglob('trace.json')]])
    value={'status':status,'stage':stage,'updated_at':utc(),'pid':os.getpid(),'policy_id':policy_id,'version_index':version_index,
        'phase_completed':counts['on']+(0 if reused else counts['off']),'phase_total':n if reused else n*2,
        'paired_completed':min(counts.values()),'per_arm_available':counts,
        'per_arm_completed':{**counts,'off':0 if reused else counts['off']},
        'per_arm_normal_completed':{**normal,'off':0 if reused else normal['off']},'reused_baseline_episodes':counts['off'] if reused else 0,
        'invalid_answer_count':sum(len(arms[a])-normal[a] for a in arms),'last_completed_task':last_task or old.get('last_completed_task'),
        'unresolved_new_requests':sum(x['status']!='metered' for x in new.values()),
        'measured_tokens':sum(x.get('accounted_tokens',0) for x in new.values()),'new_http':len(new),
        'paid_startup_observed':first_normal,'delivery_ready':status=='completed' and first_normal and val_normal and started_arms_complete and all(x['status']=='metered' for x in new.values())}
    write_json(root/'progress.json',value);return value


def report_r2(root,plan):
    root=Path(root);ledger=read(root/'budget/ledger.json');prefix=read(root/'budget/prefix.json')
    new={k:v for k,v in ledger['attempts'].items() if k not in prefix['attempts']}
    value={'method_version':c.R2,'run_id':plan['run_id'],'code':plan['code'],'implementation_ready':True,
        'usage':{phase:{'tokens':sum(x.get('accounted_tokens',x['reserved_tokens']) for x in new.values() if x['context'].get('p7_phase')==phase),
            'http':sum(x['context'].get('p7_phase')==phase for x in new.values())} for phase in plan['settings']['phase_limits']},
        'new_metered_tokens':sum(x.get('accounted_tokens',0) for x in new.values()),'new_http':len(new),
        'unresolved_new_requests':sum(x['status']!='metered' for x in new.values()),'test_authorized':False,
        'evaluation_status':'experimental_not_claimed_effective','progress':read(root/'progress.json')}
    for name,path in [('learning','learning/result.json'),('feedback','feedback/result.json'),('val','val/paired_report.json')]:
        if (root/path).exists():value[name]=read(root/path)
    value['accuracy_improved']=value.get('val',{}).get('net_correct',0)>0
    value['paid_startup_observed']=value['progress'].get('paid_startup_observed',False)
    value['phase_complete']={phase:all(len(list((root/phase/a).rglob('trace.json')))>=n for a in ('off','on')) for phase,n in [('feedback',12),('val',17)]}
    write_json(root/'report.json',value);return {k:v for k,v in value.items() if k not in {'learning','feedback','val'}}


def execute_r2(root,plan):
    if plan['method_version']!=c.R2:raise ValueError('Automatic finite execution is R2 only')
    root=Path(root);progress_r2(root,stage='learning')
    if not (root/'learning/result.json').exists():run_learn_r2(root,plan)
    selected=run_edit_r2(root,plan)
    if selected['status']=='no_candidate':
        progress_r2(root,stage='learning',status='completed_no_candidate');return report_r2(root,plan)
    freeze_r2(root,plan);run_eval(root,plan,'val')
    progress_r2(root,stage='val',status='completed');return report_r2(root,plan)


def monitor_r2(root):
    import os
    root=Path(root);value=read(root/'progress.json');ledger=read(root/'budget/ledger.json');prefix=read(root/'budget/prefix.json')
    new=[v for k,v in ledger['attempts'].items() if k not in prefix['attempts']]
    try:os.kill(value.get('pid'),0);alive=True
    except (OSError,TypeError):alive=False
    return {**value,'observed_at':utc(),'process_alive':alive,
        'unresolved_new_requests':sum(x['status']!='metered' for x in new),
        'measured_tokens':sum(x.get('accounted_tokens',0) for x in new),'new_http':len(new)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','check','learn','edit','gate','freeze','val','review','test','report','execute','monitor'])
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
        elif args.stage=='monitor':result=monitor_r2(root)
        else:
            import fcntl
            with (root/'execution.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:raise RuntimeError('Another P7 stage is already executing') from None
                plan=checked(root)
                functions={'learn':run_learn,'edit':run_edit,'gate':run_gate,'freeze':freeze,'review':review,'execute':execute_r2}
                result=functions[args.stage](root,plan) if args.stage in functions else run_eval(root,plan,args.stage)
    except BudgetExhausted as exc:
        result={'status':'budget_stopped','code':exc.code,'stage':args.stage,'automatic_retry':False}
        write_json(root/'review_required.json',result)
        if (root/'progress.json').exists():
            value=read(root/'progress.json');write_json(root/'progress.json',{**value,**result,'delivery_ready':False})
    except Exception as exc:
        if root.exists():write_json(root/'review_required.json',{'status':'stopped','stage':args.stage,
            'error_type':type(exc).__name__,'reason':str(exc),'automatic_retry':False})
        if (root/'progress.json').exists():
            value=read(root/'progress.json');write_json(root/'progress.json',{**value,'status':'stopped','error_type':type(exc).__name__,'reason':str(exc),'delivery_ready':False})
        raise
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
