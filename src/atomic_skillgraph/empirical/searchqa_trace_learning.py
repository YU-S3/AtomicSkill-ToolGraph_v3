"""Train-only original records, honest post-hoc projections and bounded R2 induction."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess

from .contracts import PublicTask, digest, object_schema, validate_schema_instance
from .searchqa_guidance import (VERSION, MATERIAL, ATOMIC, FAMILIES, POLICY, applicability,
    checked_asset, documents, features, quote_hash, render_procedure)
from ..harness.scorers.searchqa import extract_answer


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def partition(tasks):
    if len(tasks) != 300 or len({t.physical_key for t in tasks}) != 300 or any(t.split != 'train' for t in tasks):
        raise ValueError('R2 requires 300 distinct canonical Train cases')
    ordered = sorted(tasks, key=lambda t: hashlib.sha256((VERSION+'|42|'+t.physical_key).encode()).hexdigest())
    return {'schema': VERSION, 'rule':'SHA256(method_version + |42| + canonical physical_key)',
            'induction':[t.physical_key for t in ordered[:240]],
            'worker_probe':[t.physical_key for t in ordered[240:270]],
            'utility_probe':[t.physical_key for t in ordered[270:]]}


def verify_source(source_run, datasets):
    """Only Train payloads; authority hashes and cumulative ledger are Host metadata."""
    source, datasets = Path(source_run).resolve(), Path(datasets).resolve()
    if datasets.name != 'searchqa': datasets = datasets/'searchqa'
    touched = {}
    def load(path):
        path = Path(path).resolve()
        if path.name.startswith('evaluator_records_') or path.name in {'val.json','test.json'} or any(
                x in {'val','test'} for x in (path.relative_to(source).parts if path.is_relative_to(source) else ())):
            raise PermissionError('Learning cannot read held-out/evaluator materials')
        value = read(path); touched[str(path)] = sha(path); return value
    config = load(source/'train/config.json')
    execution = load(source/'train/execution_manifest.json')
    summary = load(source/'train/summary.json')
    ledger = load(source/'budget.json')
    material = load(datasets/'train.json')
    authority_train = Path(config['manifest']).resolve()
    authority = load(authority_train.parent.parent/'manifest.json')
    canonical = load(authority_train)
    if (config['experiment']['benchmark'] != 'searchqa' or config['experiment']['seed'] != 42 or
            not summary['complete'] or summary['tasks'] != 300 or summary['unknown_billing_attempts'] or
            ledger['unknown_billing'] or any(a['status'] != 'metered' for a in ledger['attempts'].values())):
        raise ValueError('Incomplete, wrong-benchmark or unmetered parent')
    if (sha(authority_train) != authority['benchmarks']['searchqa']['splits']['train']['sha256'] or
            material['authority_sha256'] != sha(authority_train.parent.parent/'manifest.json') or
            material['split_sha256'] != sha(authority_train)):
        raise ValueError('Canonical materialization authority mismatch')
    task_map = {t['task_id']:t for t in material['tasks']}
    if len(task_map) != 300 or set(task_map) != {t['task_id'] for t in canonical['tasks']}:
        raise ValueError('Canonical Train300 identity mismatch')
    if len(execution['tasks']) != 300 or set(task_map) != {t['task_id'] for t in execution['tasks']}:
        raise ValueError('Parent must contain all canonical Train300')
    source_repo = next((p for p in authority_train.parents if (p/'.git').exists()), None)
    if source_repo is None: raise ValueError('Original source checkout is required')
    commit = subprocess.check_output(['git','-C',str(source_repo),'rev-parse','HEAD'],text=True).strip()
    files = sorted([*(source_repo/'src').rglob('*.py'), *(source_repo/'experiments').rglob('*.py')])
    source_hash = digest({p.relative_to(source_repo).as_posix():sha(p) for p in files})
    if execution['code'] != {'git_sha':commit,'tracked_dirty':False,'source_sha256':source_hash}:
        raise ValueError('Original source code identity changed')
    scorer = source_repo/'src/atomic_skillgraph/harness/scorers/searchqa.py'
    touched[str(scorer)] = sha(scorer)
    bank = source/'train/bank/bank.sqlite3'
    frozen = source/'train/frozen_bank/bank.sqlite3'
    for path in (bank,frozen,source/'train/run.sqlite3'):
        touched[str(path)] = sha(path)
    freeze = load(source/'train/frozen_bank/freeze.json')
    db = sqlite3.connect((source/'train/run.sqlite3').as_uri()+'?mode=ro',uri=True)
    db.execute('PRAGMA query_only=ON')
    committed = {r[0]:json.loads(r[1]) for r in db.execute("SELECT id,result FROM tasks WHERE status='completed'")}
    db.close()
    if set(committed) != set(task_map): raise ValueError('Parent SQLite lacks canonical completed records')
    cases, counts, tokens, request_ids = [], Counter(), 0, set()
    for original in execution['tasks']:
        task = PublicTask(**original)
        if original != task_map[task.task_id] or task.split != 'train': raise ValueError('Public Train source changed')
        path = source/'train/traces'/(task.task_id+'.json')
        trace = load(path)
        if trace != committed[task.task_id] or trace['task'] != original: raise ValueError('Trace/SQLite/public input mismatch')
        attempts = list((source/'train/requests').glob(task.task_id+'_attempt*.json'))
        matching = [p for p in attempts if load(p) == {'requests':trace['requests'],'usage':trace['usage']}]
        if len(matching) != 1: raise ValueError('Missing or ambiguous original request/usage receipt')
        e = trace['execution']
        op_count = len(e.get('attempts',[]))+len(e.get('temporary_executions',[]))+sum(bool(t.get('backend_invoked')) for t in trace.get('tools',[]))
        if op_count: raise ValueError('R2 parent contract expects direct-answer original traces, not executed evidence')
        if type(trace['score'].get('hard')) is not bool: raise ValueError('Missing original Host qualification')
        for req in trace['requests']:
            for attempt in req.get('http_attempts',[]):
                key = attempt['request_id']
                if key in request_ids or key not in ledger['attempts']: raise ValueError('Duplicate or missing paid receipt')
                if not attempt.get('usage'): raise ValueError('Unknown original HTTP usage')
                request_ids.add(key)
        tokens += sum(u['total_tokens'] for u in trace['usage'])
        counts['correct' if trace['score']['hard'] else 'incorrect'] += 1
        cases.append({'physical_key':task.physical_key,'task':original,'prediction':e.get('prediction',''),
            'correct':trace['score']['hard'],'answer_status':e.get('answer_status'),
            'empty_answer':e.get('empty_answer'), 'completion_truncated':e.get('completion_truncated'),
            'historical_operation_count':0,'local_evidence':[],
            'historical_guidance_ids':trace.get('injected_guidance_ids',[]),
            'trace_path':str(path),'trace_sha256':sha(path),'requests_path':str(matching[0]),
            'requests_sha256':sha(matching[0]),'score_sha256':digest(trace['score']),
            'responses_sha256':digest([r.get('response') for r in trace['requests']]),
            'public_fields_sha256':digest({'goal':task.goal,'inputs':task.inputs})})
    if tokens != summary['known_total_tokens'] or counts['correct'] != summary['successes']:
        raise ValueError('Parent usage/score summary mismatch')
    identity = {'schema':VERSION,'source_run':str(source),'source_code':execution['code'],
        'host_scorer_sha256':sha(scorer),'canonical_manifest':str(authority_train),
        'canonical_manifest_sha256':sha(authority_train),'authority_sha256':material['authority_sha256'],
        'materialized_train_sha256':sha(datasets/'train.json'),'source_files':touched,'counts':dict(counts),
        'old_train_tokens':tokens,'old_train_http':len(request_ids), 'old_total_tokens':sum(a['accounted_tokens'] for a in ledger['attempts'].values()),
        'old_total_http':len(ledger['attempts']),'old_cumulative_limits':ledger['limits'],
        'source_frozen':freeze,'model_request':config['llm']['model'],
        'response_models':dict(Counter(a.get('public_response',{}).get('model') for t in committed.values()
            for r in t['requests'] for a in r.get('http_attempts',[]))),
        'provenance':'Historical Runtime saw evolving Guidance; no historical Python/Program execution. New Bank starts empty.'}
    return identity, cases, config


def public_case(row):
    """Allowlist only; Host score, evaluator records and answer labels never enter wire."""
    task = PublicTask(**row['task'])
    context = task.inputs['context']
    answer = extract_answer(row['prediction'] or '')
    matches = []
    if answer:
        for hit in re.finditer(re.escape(answer),context,re.I):
            doc = next((d for d in documents(context) if d['start'] <= hit.start() < hit.end() <= d['end']),None)
            if doc: matches.append({'doc_index':doc['doc_index'],'start':hit.start(),'end':hit.end(),
                'quote':context[hit.start():hit.end()],'quote_sha':quote_hash(context[hit.start():hit.end()]),
                'basis':'posthoc_located_answer_surface'})
    return {'case_id':row['physical_key'],'question':task.goal,'context':context,
        'original_public_prediction':row['prediction'],'documents':documents(context),
        'public_features':features(task),'posthoc_located_answer_surface':matches,
        'historical_operation_count':0,'historical_executed_evidence':[]}


def learning_units(rows, groups):
    eligible = {r['physical_key']:r for r in rows if r['physical_key'] in groups['induction'] and r['correct']
                and not r['empty_answer'] and not r['completion_truncated'] and r['prediction']}
    pools = {family:[k for k in groups['induction'] if k in eligible and family in features(PublicTask(**eligible[k]['task']))['families']]
             for family in FAMILIES}
    units=[]
    # Fixed round-robin; no Val, gold, or probe outcomes rank sources or families.
    for turn in range(3):
        for family in FAMILIES:
            if len(units) >= 12: break
            keys=pools[family][turn*4:(turn+1)*4]
            if len(keys)<2: continue
            units.append({'id':'unit_'+str(len(units)+1).zfill(2),'family':family,'case_ids':keys})
    return units


def validate_candidate(candidate, sources, known_parents=()):
    validate_schema_instance(candidate,ATOMIC)
    ids=candidate['source_case_ids']
    if len(ids)!=len(set(ids)) or set(ids)-set(sources) or len(set(ids))<2:
        raise ValueError('Candidate needs two distinct supplied induction sources')
    if set(candidate['parent_ids'])-set(known_parents): raise ValueError('Unknown parent candidate ID')
    guidance=render_procedure(candidate)
    if len(guidance)>600: raise ValueError('Procedure plus fallback exceeds 600 characters')
    text=candidate['goal']+' '+guidance
    if re.search(r'searchqa:|[0-9a-f]{24,}|\b(?:gold|correct_answer|task_id|physical_key)\b',text,re.I):
        raise ValueError('Identity/answer lookup content is forbidden')
    for key in ids:
        source=sources[key]
        answer=extract_answer(source['original_public_prediction'])
        if len(answer.strip())>=3 and re.search(r'(?<!\w)'+re.escape(answer.strip())+r'(?!\w)',text,re.I):
            raise ValueError('Candidate copies a source answer instead of a procedure')
        for doc in source['documents']:
            if any(len(part.strip())>=40 and part.strip() in text for part in re.split(r'[.!?\n]',doc['quote'])):
                raise ValueError('Candidate copies source prose')
    seen=set()
    for ref in candidate['evidence_refs']:
        if ref['case_id'] not in ids: raise ValueError('Reference outside declared sources')
        source=sources[ref['case_id']]
        doc=next((d for d in source['documents'] if d['doc_index']==ref['doc_index']),None)
        a,b=ref['start'],ref['end']
        if not doc or not doc['start']<=a<b<=doc['end'] or quote_hash(source['context'][a:b])!=ref['quote_sha']:
            raise ValueError('SourceRef offset/quote/hash mismatch')
        seen.add(ref['case_id'])
    if seen!=set(ids): raise ValueError('Each source requires an exact public reference')
    for key in ids:
        source=sources[key]
        task=PublicTask('source',key,source['question'],{'context':source['context']})
        if not applicability(task,candidate['applicability'])[0]: raise ValueError('Candidate prerequisites absent on source')
    asset={'method_version':VERSION,'goal':candidate['goal'],'guidance':guidance,
        'input_schema':object_schema(),'output_schema':object_schema(),'execution_intent':'guidance_only',
        'result_role':'intermediate','entry_constraints':candidate['applicability'],
        'searchqa_metadata':{'schema':MATERIAL,'candidate':deepcopy(candidate),'candidate_hash':digest(candidate),
                            'source_consistent':True,'utility':'not_tested','historical_execution_claim':False}}
    asset['id']='skill_'+digest(asset)
    if not checked_asset(asset): raise ValueError('Invalid mapped R2 asset/card capacity')
    return asset


INDUCTION = object_schema({'decision':{'enum':['candidate','no_change']},
    'candidate':{'type':'object'}, 'reason':{'type':'string','maxLength':400}},['decision','reason'])


def induce(learner, unit, sources):
    from .prompts import SEARCHQA_INDUCTION_PROMPT
    proposal=learner._receive('searchqa_'+unit['id'],'extractor',SEARCHQA_INDUCTION_PROMPT,
        {'unit':unit,'sources':[sources[k] for k in unit['case_ids']], 'candidate_contract':ATOMIC},
        'submit_searchqa_candidate',INDUCTION,material_version=MATERIAL,repair_limit=1,
        decision_purpose='searchqa_induction')
    if proposal['decision']=='no_change':
        if 'candidate' in proposal: raise ValueError('no_change cannot carry a candidate')
        return {'decision':'no_change','reason':proposal['reason']}
    candidate=proposal.get('candidate')
    asset=validate_candidate(candidate,{k:sources[k] for k in unit['case_ids']})
    saved=learner.system.bank.put('skill',asset)
    return {'decision':'candidate','asset_id':saved['id'],'candidate':candidate,'source_consistent':True}


def propose_program_specs(learner, asset, sources):
    from .searchqa_program_probe import SPEC, PROPOSAL, validate_spec
    from .prompts import SEARCHQA_PROGRAM_SPEC_PROMPT
    ids=asset['searchqa_metadata']['candidate']['source_case_ids']
    proposal=learner._receive('spec_'+asset['id'],'extractor',SEARCHQA_PROGRAM_SPEC_PROMPT,
        {'atomic':asset['searchqa_metadata']['candidate'],'sources':[sources[k] for k in ids],
         'spec_contract':SPEC,'proposal_origin':'candidate_derived_from_text'},
        'submit_searchqa_program_spec',PROPOSAL,material_version=MATERIAL,repair_limit=1,
        decision_purpose='searchqa_program_spec')
    if proposal['decision']=='no_change':
        if 'spec' in proposal: raise ValueError('no_change cannot carry a spec')
        return None
    validate_spec(proposal['spec'],asset)
    return proposal['spec']
