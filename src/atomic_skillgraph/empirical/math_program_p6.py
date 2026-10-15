"""Train demand aggregation and independently checked local/transfer evidence for P6."""
from copy import deepcopy
import json
from pathlib import Path
import re
import time

from . import math_operation_contract_p6 as c
from .contracts import PublicTask, digest, program_digest, validate_program
from .math_program_probe import PureBroker
from .task_context import TaskContext
from ..harness.benchmarks import AnswerAdapter
from .checkpoint import save_json


def catalog(rows, maximum=8):
    demands=[];buckets={k:[] for k in c.KINDS}
    hints={'quantifier_dependency_audit':r'\\forall|\\exists|for every|there exists|unique',
           'exact_rational_comparison':r'fraction|rational|\\[dt]?frac',
           'finite_relation_quantifiers':r'finite relation|relation matrix'}
    for row in rows:
        task=row['task'];available=[]
        for kind in c.KINDS:
            grounded=c.ready_bindings(task,kind)
            refs=[s['span_id'] for s in c.public_spans(task) if re.search(hints[kind],s['quote'],re.I)]
            if grounded or refs:
                demand={'kind':'PublicDemand','physical_key':task['physical_key'],'task_id':task['task_id'],
                    'operation_kind':kind,'public_source_refs':refs,'parent_outcome':row['parent_outcome'],
                    'source_trace_sha':row['trace_sha256'],'input_ready':bool(grounded),
                    'status':'ready' if grounded else 'requires_grounding','reason':None if grounded else 'unbound_semantics',
                    'grounding':grounded,'proof_obligation':c.interface(kind)['semantics'],
                    'evidence_boundary':'public syntax and historical feedback, not a theorem proof'}
                demands.append(demand);available.append(kind)
                if grounded:buckets[kind].append(demand)
        if not available:demands.append({'kind':'PublicDemand','physical_key':task['physical_key'],'task_id':task['task_id'],
            'source_trace_sha':row['trace_sha256'],'parent_outcome':row['parent_outcome'],'status':'not_applicable',
            'input_ready':False,'reason':'no_registered_computable_public_obligation'})
    # Reserve one physical task per operation for independent transfer BEFORE generation.
    units=[];transfer=[]
    for kind,items in buckets.items():
        if len(items)>2:transfer.append(items[-1]['physical_key']);items=items[:-1]
        for i in range(0,len(items),2):
            selected=items[i:i+2]
            if not selected:continue
            u={'operation_kind':kind,'source_case_ids':[v['physical_key'] for v in selected],
               'public_demands':selected,'source_count_status':'multi_source_hypothesis' if len(selected)>1 else 'single_source_hypothesis',
               'generation_basis':'HypothesizedOperation_not_observed_execution'}
            u['unit_id']='p6_unit_'+digest(u);units.append(u)
    # Round-robin operation diversity, then fixed public source order. Never score ranking.
    groups={k:[u for u in units if u['operation_kind']==k] for k in c.KINDS};selected=[]
    for i in range(max([len(v) for v in groups.values()]+[0])):
        for k in c.KINDS:
            if i<len(groups[k]):selected.append(groups[k][i])
    selected=selected[:maximum]
    learned={key for u in selected for key in u['source_case_ids']}
    transfer=[k for k in dict.fromkeys(transfer) if k not in learned]
    return demands,selected,transfer


def material(unit, rows, atoms):
    keys=set(unit['source_case_ids'])
    sources=[]
    for row in rows:
        if row['task']['physical_key'] not in keys:continue
        sources.append({**deepcopy(row),'spans':list(c.public_spans(row['task'])),
            'process_qualification':'model_reported_reasoning_not_formal_proof'})
    return {'learning_material_version':c.MATERIAL,'unit':unit,'train_sources':sources,
        'atomic_source_hypotheses':[a for a in atoms if set(a['source_case_ids'])<=keys],
        'operation_registry':{unit['operation_kind']:c.interface(unit['operation_kind'])},
        'binding_parsers':c.PARSERS[unit['operation_kind']]}


def atomic(proposal, unit):
    a=proposal['atomic'];value={'goal':a['goal'],'applicability':a['applicability'],'guidance':a['operation'],
        'prerequisites':a['preconditions'],'operation':a['operation'],'stop_condition':'; '.join(a['stop_conditions']),
        'outputs':list(c.interface(proposal['program_spec']['operation_kind'])['output_schema']['properties']),
        'limitations':c.interface(proposal['program_spec']['operation_kind'])['proof_scope'],
        'method_version':c.VERSION,'state':'candidate','execution_intent':'program_candidate',
        'source_case_ids':proposal['source_case_ids'],'public_source_refs':proposal['public_source_refs'],
        'parent_outcomes':{d['physical_key']:d['parent_outcome'] for d in unit['public_demands']},
        'source_count_status':'multi_source_hypothesis' if len(proposal['source_case_ids'])>=2 else 'single_source_hypothesis',
        'origin':'train_trace_model_generated','qualification':'hypothesis','observed_execution':False}
    value['id']='skill_'+digest(value);return value


def program(answer, spec, environment):
    source=c.validate_source(answer)
    if environment.get('image_digest')!=c.IMAGE or environment.get('adapter_abi')!='simple.v2':raise ValueError('P6 locked Worker identity differs')
    value={'source':source,'entry':'run','input_schema':deepcopy(spec['input_schema']),
        'output_schema':deepcopy(spec['output_schema']),'allowed_tools':[],'environment':deepcopy(environment),
        'result_role':'intermediate','state':'candidate','method_version':c.VERSION,
        'entry_constraints':{'operation_kind':spec['operation_kind'],'contract_fingerprint':c.fingerprint(spec['operation_kind'])}}
    try:validate_program(value)
    except (ValueError,SyntaxError) as exc:raise c.ContractError([c.error('program_abi_invalid','source',str(exc),source)]) from exc
    value['id']='program_'+program_digest(value);return value


def inputs(spec, rows, sources):
    spans={s['span_id']:s for r in rows if r['task']['physical_key'] in sources for s in c.public_spans(r['task'])}
    real=c.bind(spec,spans)
    original=real if real is not None else spec['demonstration_input'];variant=deepcopy(original)
    kind=spec['operation_kind']
    if kind=='quantifier_dependency_audit':variant['statement']=r'\exists z '+original['statement']
    elif kind=='finite_relation_quantifiers':variant['relation']=[[not x for x in row] for row in original['relation']]
    else:variant['left']=[[a+1 if a<1000000 else a-1,b] for a,b in original['left']]
    missing={'statement':'No explicit formal quantifier supplied.'} if kind=='quantifier_dependency_audit' else (
        {'relation':[[True],[False,True]]} if kind=='finite_relation_quantifiers' else
        {'left':[[1,2]],'right':[[1,2],[1,3]]})
    return [('bound_source' if real is not None else 'synthetic_demonstration',original),('synthetic_variation',variant),('missing_premise',missing)]


def execute(worker, program, arguments, folder, role, *, grounding=None):
    """A persisted started claim without a receipt is an unresolved execution, never a retry."""
    folder=Path(folder);identity=digest([program_digest(program),arguments,role,grounding])
    path=folder/(identity+'.json');claim=folder/(identity+'.started.json')
    if path.exists():
        row=json.loads(path.read_text())
        if row['id']!=identity or row['receipt_hash']!=digest({k:v for k,v in row.items() if k!='receipt_hash'}):raise ValueError('P6 Worker receipt changed')
        return row
    if claim.exists():raise RuntimeError('Unresolved P6 Worker invocation: '+str(claim))
    save_json(claim,{'id':identity,'program_digest':program_digest(program),'input_hash':digest(arguments),'started_at':time.time()})
    adapter=AnswerAdapter('livemath',{identity:{}});adapter.reset(PublicTask(identity,identity,'Bounded local operation check'))
    broker=PureBroker(adapter,1,context=TaskContext({}));kind=program['entry_constraints']['operation_kind']
    try:
        result=worker.execute(program,arguments,broker);receipt=result.get('execution_receipt',{})
        if not receipt or receipt.get('image_digest')!=c.IMAGE or '--network=none' not in receipt.get('command',[]):
            raise RuntimeError('Missing locked Docker execution receipt')
        expected=None if role=='missing_premise' else c.oracle(kind,arguments)
        passed=(result['status']=='needs_input' if expected is None else result['status']=='ok' and result.get('outputs')==expected)
        passed=passed and receipt.get('exit_code')==0 and not broker.events
        row={'id':identity,'method_version':c.VERSION,'program_digest':program_digest(program),'role':role,
            'arguments':arguments,'input_hash':digest(arguments),'grounding':grounding,'result':result,
            'expected_local_outputs':expected,'passed':passed,'rpc_calls':broker.events,
            'evidence_kind':'ValidatedLocalResult' if passed else 'local_check_failed',
            'algorithm_test_scope':c.interface(kind)['proof_scope'],'task_correctness_claimed':False}
        row['receipt_hash']=digest(row);save_json(path,row);return row
    finally:adapter.close()


def qualify(worker, program, proposal, rows, transfer_keys, folder):
    folder=Path(folder);spec=proposal['program_spec'];sources=proposal['source_case_ids']
    receipts=[execute(worker,program,args,folder/'worker',role) for role,args in inputs(spec,rows,sources)]
    local=all(r['passed'] for r in receipts);transfers=[]
    if local:
        for row in rows:
            task=row['task']
            if task['physical_key'] not in transfer_keys or task['physical_key'] in sources:continue
            grounding=c.ready_bindings(task,spec['operation_kind'])
            if grounding:
                transfers.append(execute(worker,program,grounding['grounded_task_inputs'],folder/'worker',
                    'independent_train_transfer',grounding=grounding))
    bound=receipts[0]['role']=='bound_source' and receipts[0]['passed']
    return {'method_version':c.VERSION,'local_algorithm_checked':local,'task_binding_checked':bound,
        'task_transfer_checked':any(r['passed'] for r in transfers),'transfer_status':'checked' if any(r['passed'] for r in transfers) else 'transfer_failed' if transfers else 'coverage_insufficient',
        'source_case_ids':sources,'receipts':receipts,'transfer_receipts':transfers,
        'utility_observed':False,'reviewed_for_usable':False,'formal_usable':False,
        'qualification':'experimental_trial_only' if local else 'local_check_failed'}
