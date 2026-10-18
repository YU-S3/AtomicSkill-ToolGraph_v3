"""Audited learned Program takeover with the unchanged P7 model fallback."""
from copy import deepcopy
from pathlib import Path

from . import math_effect_program_r3 as p, math_effect_runtime as base
from .checkpoint import save_json
from .contracts import digest, program_digest
from .math_effect_source import read


def contract(cfg):
    return digest([base.ANSWER,base.PROMPT,base.FINISH,cfg['llm'],cfg['effect_p7']['runtime']])


def verify(trace):
    base.verify_trace(trace)
    if trace.get('reused_model_response'):
        proof=trace['reused_model_response'];original=read(proof['source_trace'])
        base.verify_trace(original)
        if (original['trace_hash']!=proof['source_trace_hash'] or original['runtime_contract_hash']!=trace['runtime_contract_hash'] or
            any(original[k]!=trace[k] for k in ('task','prediction','score','request_ids','raw_usage','request_files','tokens','http')) or
            original['public_input_hash']!=trace['public_input_hash'] or proof['actual_wire_hash']!=trace['main_wire_hash']):
            raise ValueError('Reused physical response identity changed')
    for path,h in trace.get('program_receipts',{}).items():
        r=read(path)
        if digest(r)!=h or r['receipt_hash']!=digest({k:v for k,v in r.items() if k!='receipt_hash'}):
            raise ValueError('Program receipt changed')
    if trace.get('answer_origin')=='learned_program':
        executions=[read(path) for path in trace['program_receipts']]
        votes=[r['decisions'][0]['label'] for r in executions if r['valid'] and r['decisions'][0]['label']]
        if (not votes or len(set(votes))!=1 or trace['prediction']!='<answer>'+votes[0]+'</answer>' or
            trace['http'] or trace['tokens'] or not trace['consumed_program_ids']):raise ValueError('Invalid zero-HTTP takeover')


def solve(task,adapter,candidates,qualifications,cfg,root,governor,*,providers=None,worker=None,reuse_trace=None,require_reuse=False):
    root=Path(root);identity=digest([task.__dict__,candidates,qualifications,cfg]);path=root/'trace.json'
    if path.exists():
        saved=read(path)
        if saved['identity']!=identity:raise ValueError('R3 task inputs changed')
        verify(saved);adapter.reset(task)
        if adapter.evaluate(adapter.submit(saved['prediction']))!=saved['score']:raise ValueError('Official scorer changed')
        return saved
    votes=[];receipts={};used=[];valid_programs=True
    for c in candidates:
        q=qualifications.get(c['id'])
        if not q or not q['qualified_for_trial'] or not p.applicable(c,task):continue
        if not verify_train(q,c,cfg['program_environment']):raise ValueError('Program admission evidence changed')
        folder=root/'programs'/q['program']['id']
        r=p.execute(q['program'],[task],folder,worker)
        receipts[str((folder/'execution.json').resolve())]=digest(r)
        valid_programs=valid_programs and r['valid']
        if r['valid'] and r['decisions'][0]['label']:
            votes.append(r['decisions'][0]['label']);used.append((c['id'],q['program']['id']))
    agreed=bool(votes) and len(set(votes))==1 and valid_programs
    settings=cfg['effect_p7'];selection=p.routing(candidates,task,settings['tokenizer_path'],settings['max_injection_tokens'])
    if agreed:
        prediction='<answer>'+votes[0]+'</answer>';adapter.reset(task)
        score=adapter.evaluate(adapter.submit(prediction));audit=deepcopy(adapter.score_audit)
        trace={'schema':p.VERSION+'.answer','task':task.__dict__,'prediction':prediction,
            'valid':adapter.valid_answer_format(prediction),'score':score,'scorer_audit':audit,
            'score_hash':digest([score,audit]),'first_truncated':False,'finish_state':'not_needed',
            'budget_censored':False,'response_hashes':[],'response_models':[],'request_ids':[],
            'raw_usage':[],'tokens':0,'http':0,'latency_ms':0,'request_files':{},
            'selection':{**selection,'selected':[x[0] for x in used]},
            'public_input_hash':digest({'goal':task.goal,'choices':task.inputs['choices']}),
            'runtime_contract_hash':contract(cfg),'answer_origin':'learned_program'}
    else:
        expected=base.provider(cfg['llm'],'runtime',settings)._build_payload(
            base.messages(task,[p.old.card(c['candidate']) for c in selection['selected']]),[])
        reusable=compatible_response(reuse_trace,task,cfg,expected) if reuse_trace else None
        if reusable:
            trace=deepcopy(reusable);trace.pop('trace_hash')
            trace['selection']={**selection,'selected':[c['id'] for c in selection['selected']]}
            trace['answer_origin']='dynamic_reused'
            trace['reused_model_response']={'source_trace':str(Path(reuse_trace).resolve()),
                'source_trace_hash':reusable['trace_hash'],'actual_wire_hash':digest(expected),'new_tokens':0,'new_http':0,
                'qualification':'exact_public_input_provider_payload_and_runtime_contract; original physical response preserved'}
            adapter.reset(task)
            if adapter.evaluate(adapter.submit(trace['prediction']))!=trace['score']:raise ValueError('Reused native score changed')
        else:
            if require_reuse:raise ValueError('This Train stage cannot issue a new Runtime solve; exact response reuse is required')
            trace=deepcopy(base.solve(task,adapter,candidates,cfg,root/'dynamic',governor,
                providers=providers,selection_override=selection))
            trace.pop('trace_hash');trace['answer_origin']='dynamic'
    trace.update(identity=identity,method_version=p.VERSION,
        experiment_method_version=cfg['effect_p7'].get('derivation',{}).get('version',p.VERSION),
        program_calls=len(receipts),program_receipts=receipts,
        program_votes=votes,program_conflict=len(set(votes))>1,program_execution_failure=not valid_programs,
        consumed_program_ids=[x[1] for x in used] if agreed else [],
        program_admission_hash=digest(qualifications))
    trace['trace_hash']=digest(trace);save_json(path,trace);return trace


def compatible_response(path,task,cfg,payload):
    """Optional exact-wire reuse; no nearest prompt, score substitution or Test imports."""
    from .math_effect_optimizer import main_wire_hash
    path=Path(path)
    if not path.exists():return None
    saved=read(path);verify(saved)
    if (task.split=='test' or saved['task']!=task.__dict__ or not saved['http'] or
        saved['runtime_contract_hash']!=contract(cfg) or main_wire_hash(saved)!=digest(payload)):
        return None
    return saved


def verify_train(q,c,environment):
    try:
        ex=q['execution'];rows=q['rows'];selected=[r for r in rows if r['dispatched']]
        return (q['receipt_hash']==digest({k:v for k,v in q.items() if k!='receipt_hash'}) and
            q['candidate_id']==c['id'] and q['candidate_hash']==digest(c) and
            q['program']==p.compile_program(c,environment) and
            ex['receipt_hash']==digest({k:v for k,v in ex.items() if k!='receipt_hash'}) and ex['valid'] and
            ex['program_hash']==program_digest(q['program']) and len(ex['decisions'])==3*len(rows) and
            q['label_order_ambiguity_checks'] and len(selected)==q['dispatched'] and len(selected)>=4 and
            sum(bool(r['correct']) for r in selected)==q['correct'] and q['correct']/len(selected)>=.8 and
            not any(r['historical_correct'] and not r['correct'] for r in selected) and q['qualified_for_trial'])
    except (KeyError,ValueError,TypeError,ZeroDivisionError):return False


def pair_report(off,on,candidates,split):
    from .contracts import PublicTask
    from .math_effect_selection import stronger_meta
    if len(off)!=len(on) or len({t['task']['physical_key'] for t in off})!=len(off):raise ValueError('Missing or duplicate pairs')
    rows=[]
    for a,b in zip(off,on):
        verify(a);verify(b)
        if not set(b['selection']['selected'])<={c['id'] for c in candidates}:raise ValueError('Injected unlisted Skill')
        if any(a[k]!=b[k] for k in ('task','public_input_hash','runtime_contract_hash')):raise ValueError('Off/on protocol mismatch')
        if b['http'] and set(a['response_models'])!=set(b['response_models']):raise ValueError('Response model mismatch')
        if not a['http'] or any(u is None for t in (a,b) for u in t['raw_usage']):raise ValueError('Unqualified usage')
        if b['http']:
            from .math_effect_optimizer import main_wire_hash
            intervened=main_wire_hash(a)!=main_wire_hash(b)
        else:intervened=b['answer_origin']=='learned_program'
        rows.append({'physical_key':a['task']['physical_key'],'off':a,'on':b,
            'intervened':intervened,'meta':stronger_meta(PublicTask(**a['task']))['has_stronger_meta'],
            'win':not a['score']['hard'] and b['score']['hard'],'loss':a['score']['hard'] and not b['score']['hard']})
    def stats(rs):
        return {'tasks':len(rs),'off_correct':sum(x['off']['score']['hard'] for x in rs),
            'on_correct':sum(x['on']['score']['hard'] for x in rs),'wins':sum(x['win'] for x in rs),
            'losses':sum(x['loss'] for x in rs),'empty_off':sum(not x['off']['valid'] for x in rs),
            'empty_on':sum(not x['on']['valid'] for x in rs),'program_answers':sum(x['on'].get('answer_origin')=='learned_program' for x in rs),
            'tokens_off':sum(x['off']['tokens'] for x in rs),'tokens_on':sum(x['on']['tokens'] for x in rs)}
    total=stats(rows);net=total['on_correct']-total['off_correct'];intervened=stats([x for x in rows if x['intervened']])
    result={'method_version':p.VERSION,'split':split,'candidate_ids':[x['id'] for x in candidates],
        'candidate_hashes':[digest(x) for x in candidates],'rows':rows,'groups':{'all':total,
        'meta':stats([r for r in rows if r['meta']]),'non_meta':stats([r for r in rows if not r['meta']]),
        'intervened':intervened},'net_correct':net,
        'qualified':split=='val' and len(rows)==17 and net>=2 and total['losses']<=1 and
            intervened['on_correct']-intervened['off_correct']>=2 and
            total['empty_on']<=total['empty_off'] and total['on_correct']>=12,
        'baseline_cost_is_reused':True,'evaluation_scope':'development_observation'}
    result['receipt_hash']=digest(result);return result


def verify_utility(r):
    try:
        if r['receipt_hash']!=digest({k:v for k,v in r.items() if k!='receipt_hash'}):return False
        # Recompute all scores and intervention statistics from immutable traces.
        placeholders=[{'id':i} for i in r['candidate_ids']]
        expected=pair_report([x['off'] for x in r['rows']],[x['on'] for x in r['rows']],placeholders,r['split'])
        expected['candidate_hashes']=r['candidate_hashes'];expected.pop('receipt_hash');expected['receipt_hash']=digest(expected)
        return expected==r
    except (KeyError,ValueError,TypeError,OSError):return False


def qualification(program,validation):
    try:
        c=validation['candidate'];q=validation['train'];v=validation['val']
        return (validation['policy']==p.VERSION+'.qualification' and validation['passed'] is True and
            validation['program_digest']==program_digest(program) and validation['environment_hash']==digest(program['environment']) and
            program_digest(program)==program_digest(p.compile_program(c,program['environment'])) and
            verify_train(q,c,program['environment']) and verify_utility(v) and v['qualified'] and
            c['id'] in v['candidate_ids'] and digest(c)==v['candidate_hashes'][v['candidate_ids'].index(c['id'])] and
            any(program['id'] in r['on'].get('consumed_program_ids',[]) for r in v['rows']))
    except (KeyError,ValueError,TypeError,OSError):return False
