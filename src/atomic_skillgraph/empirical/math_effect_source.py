"""Read-only raw Train provenance and public-only stratified partition."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .bank import Bank
from .contracts import PublicTask, digest
from .math_effect_selection import features
from ..experiments.canonical_manifest import sha256


def read(path):return json.loads(Path(path).read_text())


def partition(tasks, metadata, seed=42):
    sizes={'proposal_train':28,'edit_feedback':12,'protected_gate':20}
    strata=defaultdict(list)
    for t in tasks:
        m=metadata[t['task_id']]
        key=json.dumps([m['month'],sorted(m['theorem_type']),features(PublicTask(**t))['meta_option_present']])
        strata[key].append(t['physical_key'])
    result={k:[] for k in sizes};counts=Counter();distribution={}
    # Rare strata first; normalized quota deficits, hashed physical-key tie breaks.
    for key,keys in sorted(strata.items(),key=lambda p:(len(p[1]),p[0])):
        local=Counter()
        for physical in sorted(keys,key=lambda k:hashlib.sha256(f'p7|{seed}|{k}'.encode()).hexdigest()):
            available=[k for k in sizes if counts[k]<sizes[k]]
            target=min(available,key=lambda k:(local[k]/sizes[k],counts[k]/sizes[k],digest([seed,physical,k])))
            result[target].append(physical);counts[target]+=1;local[target]+=1
        distribution[key]=dict(local)
    if counts!=Counter(sizes):raise ValueError('P7 requires exactly 60 unique Train tasks')
    return {'algorithm':'p7.public-strata-deficit.sha256.v1','seed':seed,'members':result,
            'strata':distribution,'rare_strata':[k for k,v in strata.items() if len(v)<3]}


def load(source, datasets):
    root=Path(source);data=Path(datasets)/'livemath';files={}
    def locked(path):
        files[str(path.resolve())]=sha256(path);return read(path)
    config=locked(root/'train/config.json');summary=locked(root/'train/summary.json')
    execution=locked(root/'train/execution_manifest.json');manifest=locked(root/'run_manifest.json')
    if not summary['complete'] or summary['tasks']!=60 or manifest['benchmark']!='livemath' or manifest['run_seed']!=42:
        raise ValueError('P7 requires the completed original LiveMath seed42 Train60')
    if execution['config']!=config or [r['task'] for r in summary['cases']]!=execution['tasks']:
        raise ValueError('Source execution/config/summary identity mismatch')
    authority=Path(config['manifest']).parent
    tasks={};metadata={}
    for split,n in [('train',60),('val',17),('test',100)]:
        canonical=locked(authority/(split+'.json'));material=locked(data/(split+'.json'))
        if material['split_sha256']!=sha256(authority/(split+'.json')):raise ValueError('Materialized split identity differs')
        tasks[split]=material['tasks']
        if len(tasks[split])!=n or {t['task_id'] for t in tasks[split]}!={t['task_id'] for t in canonical['tasks']}:
            raise ValueError('Canonical split members differ')
        metadata.update({t['task_id']:t['task_metadata'] for t in canonical['tasks']})
        # Hash evaluator bytes, never expose its contents to Learner/Runtime.
        files[str((data/('evaluator_records_'+split+'.json')).resolve())]=sha256(data/('evaluator_records_'+split+'.json'))
    all_keys=[t['physical_key'] for ts in tasks.values() for t in ts]
    if len(set(all_keys))!=177:raise ValueError('Train/Val/Test physical overlap')
    if {digest(t) for t in tasks['train']}!={digest(t) for t in execution['tasks']}:
        raise ValueError('Historical Train differs from canonical public inputs')
    path=root/'train/bank/bank.sqlite3';files[str(path.resolve())]=sha256(path)
    bank=Bank(path.parent,readonly=True,environment=config['program_environment'])
    try:
        if bank.digest()!=summary['knowledge_digest']:raise ValueError('Historical Train Bank digest changed')
        cases=bank.train_cases()
    finally:bank.close()
    if [c['task'] for c in cases]!=execution['tasks']:raise ValueError('Missing actual Train cases')
    traces={};catalog=[]
    for case,scored in zip(cases,summary['cases']):
        t=case['task'];p=root/'train/traces'/(t['task_id']+'.json');trace=locked(p)
        if trace['task']!=t or trace['score']!=scored['score'] or trace['score']!=case['experience']['score']:
            raise ValueError('Historical score/trace/Bank mismatch')
        traces[t['physical_key']]=trace
        catalog.append({'physical_key':t['physical_key'],'task_id':t['task_id'],'trace':str(p),
            'trace_sha256':sha256(p),'score_sha256':digest(trace['score']),
            'response_sha256':digest(trace['requests']),'source_run_id':manifest['run_id']})
    for p in sorted((root/'train/requests').glob('*.json')):files[str(p.resolve())]=sha256(p)
    if (root/'budget.json').exists():locked(root/'budget.json')
    return {'tasks':tasks,'partition':partition(tasks['train'],metadata),'catalog':catalog,
        'files':files,'config':config,'manifest':manifest,'traces':traces}


def window(text, limit):
    raw=text.encode();end=min(len(raw),limit)
    while end<len(raw) and raw[end]&0xc0==0x80:end-=1
    return {'text':raw[:end].decode(),'full_sha256':hashlib.sha256(raw).hexdigest(),
            'start_byte':0,'end_byte':end,'full_bytes':len(raw),'truncated':end<len(raw)}


def proposal_material(source, trace_bytes=4096):
    keys=source['partition']['members']['proposal_train'];rows=[]
    for key in keys:
        t=source['traces'][key]
        process={'public_process':t['execution'].get('public_process_evidence'),
            'runtime_responses':[{k:r.get('response',{}).get(k) for k in ('content','reasoning_content','tool_calls')}
                for r in t['requests'] if r.get('stage')=='runtime']}
        rows.append({'case_id':key,'public_task':{'goal':t['task']['goal'],'choices':t['task']['inputs']['choices']},
            'observed_prediction':t['execution'].get('prediction'),'train_correctness':t['score']['hard'],
            'historical_trace':window(json.dumps(process,ensure_ascii=False),trace_bytes),
            'evidence_status':'observed_task_feedback_not_mathematical_proof','trace_hash':digest(t)})
    return rows


def proposal_material_r2(source,process_bytes=4096):
    """Complete public questions plus typed, honestly bounded observed process fields."""
    rows=[];catalog={x['physical_key']:x for x in source['catalog']}
    for task in source['tasks']['train']:
        key=task['physical_key'];t=source['traces'][key];observed=[]
        public=t['execution'].get('public_process_evidence')
        if public:observed.append({'kind':'public_process_evidence','value':deepcopy(public),'qualification':'observed_not_proven'})
        for i,r in enumerate(t['requests']):
            if r.get('stage')!='runtime':continue
            response=r.get('response',{})
            if response.get('tool_calls'):
                observed.append({'kind':'recorded_tool_calls','request_index':i,'value':deepcopy(response['tool_calls'])})
            reasoning=response.get('reasoning_content')
            if reasoning:
                observed.append({'kind':'model_reported_reasoning','request_index':i,
                    'excerpt':window(reasoning,process_bytes),'qualification':'unverified_model_process'})
        # A label-only response is task feedback, not an invented mathematical derivation.
        rows.append({'case_id':key,'public_task':{'goal':task['goal'],'choices':deepcopy(task['inputs']['choices'])},
            'historical_prediction':t['execution'].get('prediction',''),'historical_correct':t['score']['hard'],
            'observed_process':observed,'process_status':'recorded' if observed else 'not_recorded',
            'source_trace_hash':catalog[key]['trace_sha256'],'evidence_status':'task_feedback_not_proof'})
    return rows


def material_batches_r2(rows,max_bytes):
    batches=[];current=[]
    for row in rows:
        if len(json.dumps([row],allow_nan=False).encode())>max_bytes:
            raise ValueError('Complete source question exceeds one learning batch; no truncation')
        if current and len(json.dumps([*current,row],allow_nan=False).encode())>max_bytes:
            batches.append(current);current=[]
        current.append(row)
    if current:batches.append(current)
    if len({r['case_id'] for b in batches for r in b})!=len(rows):raise ValueError('Duplicate learning source')
    return batches
